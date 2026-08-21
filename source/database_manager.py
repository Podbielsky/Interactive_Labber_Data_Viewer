"""Managed, optional measurement database support for the HDF5 viewer."""

from dataclasses import dataclass
from contextlib import contextmanager
import datetime
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile

import numpy as np

from HDF5Data import HDF5MapPreview, inspect_viewer_hdf5, load_hdf5_map_preview


DATABASE_METADATA_DIRECTORY = '.labber_viewer'
DATABASE_FILE_NAME = 'database.sqlite3'
DATABASE_IMPORT_DIRECTORY = 'Drag and Drop'
DATABASE_PREVIEW_DIRECTORY = 'previews'
DATABASE_SCHEMA_VERSION = 1
HDF5_FILE_EXTENSIONS = {'.h5', '.hdf5'}


def _utc_now():
    """Return a portable UTC timestamp for database metadata."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _sha256_file(path, chunk_size=1024 * 1024):
    """Hash a potentially large file without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, 'rb') as source_file:
        for chunk in iter(lambda: source_file.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class MeasurementRecord:
    """One indexed measurement and its user-owned metadata."""

    measurement_id: int
    relative_path: str
    file_name: str
    file_size: int
    modified_ns: int
    sha256: str
    data_channel: str
    step_dimensions: tuple
    has_traces: bool
    starred: bool
    comment: str
    added_at: str
    updated_at: str


@dataclass(frozen=True)
class DatabaseScanResult:
    """Summary returned after a recursive database scan."""

    discovered: int
    indexed: int
    invalid: tuple
    cancelled: bool = False


@dataclass(frozen=True)
class DatabaseImportResult:
    """Result of copying or matching a dropped measurement."""

    record: MeasurementRecord
    already_present: bool


class MeasurementDatabase:
    """Own the portable SQLite index stored inside a database directory."""

    def __init__(self, root_directory):
        root_directory = os.path.abspath(os.path.expanduser(root_directory))
        if not os.path.isabs(root_directory):
            raise ValueError('The database directory must be an absolute path.')
        self.root_directory = root_directory
        self.metadata_directory = os.path.join(
            root_directory, DATABASE_METADATA_DIRECTORY
        )
        self.database_path = os.path.join(
            self.metadata_directory, DATABASE_FILE_NAME
        )
        self.preview_directory = os.path.join(
            self.metadata_directory, DATABASE_PREVIEW_DIRECTORY
        )
        self.import_directory = os.path.join(
            root_directory, DATABASE_IMPORT_DIRECTORY
        )

    @staticmethod
    def is_initialized(root_directory):
        """Return whether a folder already contains a viewer database index."""
        root_directory = os.path.abspath(os.path.expanduser(root_directory))
        return os.path.isfile(
            os.path.join(
                root_directory,
                DATABASE_METADATA_DIRECTORY,
                DATABASE_FILE_NAME,
            )
        )

    def initialize(self):
        """Create the managed directories and schema without resetting metadata."""
        os.makedirs(self.root_directory, exist_ok=True)
        if not os.path.isdir(self.root_directory):
            raise ValueError('The selected database path is not a directory.')
        os.makedirs(self.metadata_directory, exist_ok=True)
        os.makedirs(self.preview_directory, exist_ok=True)
        os.makedirs(self.import_directory, exist_ok=True)

        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS database_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS measurements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    relative_path TEXT NOT NULL UNIQUE,
                    file_name TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    modified_ns INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    data_channel TEXT NOT NULL DEFAULT '',
                    step_dimensions TEXT NOT NULL DEFAULT '[]',
                    has_traces INTEGER NOT NULL DEFAULT 0,
                    starred INTEGER NOT NULL DEFAULT 0,
                    comment TEXT NOT NULL DEFAULT '',
                    added_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS measurements_sha256_index
                ON measurements (sha256);
                """
            )
            current_version = connection.execute(
                "SELECT value FROM database_metadata WHERE key = 'schema_version'"
            ).fetchone()
            if current_version is not None:
                try:
                    schema_version = int(current_version['value'])
                except (TypeError, ValueError) as error:
                    raise ValueError('The database schema version is invalid.') from error
                if schema_version > DATABASE_SCHEMA_VERSION:
                    raise ValueError(
                        'This database was created by a newer viewer version.'
                    )

            connection.execute(
                """
                INSERT INTO database_metadata (key, value)
                VALUES ('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(DATABASE_SCHEMA_VERSION),),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO database_metadata (key, value)
                VALUES ('created_at', ?)
                """,
                (_utc_now(),),
            )

    @contextmanager
    def _connect(self):
        """Open a worker-local SQLite connection with named result columns."""
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA foreign_keys = ON')
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _relative_path(self, absolute_path):
        """Return a normalized relative path guaranteed to remain in the root."""
        absolute_path = os.path.abspath(absolute_path)
        try:
            common_path = os.path.commonpath((self.root_directory, absolute_path))
        except ValueError as error:
            raise ValueError('The measurement is outside the database directory.') from error
        if os.path.normcase(common_path) != os.path.normcase(self.root_directory):
            raise ValueError('The measurement is outside the database directory.')
        relative_path = os.path.relpath(absolute_path, self.root_directory)
        return relative_path.replace(os.sep, '/')

    def get_absolute_path(self, record_or_relative_path):
        """Resolve a record path while preventing traversal outside the database."""
        if isinstance(record_or_relative_path, MeasurementRecord):
            relative_path = record_or_relative_path.relative_path
        else:
            relative_path = str(record_or_relative_path)
        components = [part for part in relative_path.split('/') if part]
        absolute_path = os.path.abspath(os.path.join(self.root_directory, *components))
        self._relative_path(absolute_path)
        return absolute_path

    @staticmethod
    def _record_from_row(row):
        """Convert a SQLite row into the public immutable record type."""
        try:
            step_dimensions = tuple(json.loads(row['step_dimensions']))
        except (TypeError, ValueError, json.JSONDecodeError):
            step_dimensions = ()
        return MeasurementRecord(
            measurement_id=int(row['id']),
            relative_path=row['relative_path'],
            file_name=row['file_name'],
            file_size=int(row['file_size']),
            modified_ns=int(row['modified_ns']),
            sha256=row['sha256'],
            data_channel=row['data_channel'],
            step_dimensions=step_dimensions,
            has_traces=bool(row['has_traces']),
            starred=bool(row['starred']),
            comment=row['comment'],
            added_at=row['added_at'],
            updated_at=row['updated_at'],
        )

    def list_measurements(self):
        """Return every measurement ordered by its portable relative path."""
        with self._connect() as connection:
            rows = connection.execute(
                'SELECT * FROM measurements ORDER BY relative_path COLLATE NOCASE'
            ).fetchall()
        return [self._record_from_row(row) for row in rows]

    def get_measurement(self, measurement_id):
        """Return one measurement or None when it was removed by a rescan."""
        with self._connect() as connection:
            row = connection.execute(
                'SELECT * FROM measurements WHERE id = ?',
                (int(measurement_id),),
            ).fetchone()
        return self._record_from_row(row) if row is not None else None

    def find_by_hash(self, file_hash):
        """Return the first existing database file with matching content."""
        with self._connect() as connection:
            rows = connection.execute(
                'SELECT * FROM measurements WHERE sha256 = ? ORDER BY id',
                (file_hash,),
            ).fetchall()
        for row in rows:
            record = self._record_from_row(row)
            if os.path.isfile(self.get_absolute_path(record)):
                return record
        return None

    def load_map_preview(self, record, maximum_points_per_axis=500):
        """Load a downsampled preview, using a content-addressed local cache."""
        maximum_points_per_axis = int(maximum_points_per_axis)
        if maximum_points_per_axis <= 0:
            raise ValueError('The preview size limit must be positive.')
        # The database is initialized when configured or scanned. Preview reads
        # only need their cache directory and must not contend with a rescan's
        # SQLite transaction by re-running schema initialization here.
        os.makedirs(self.preview_directory, exist_ok=True)
        cache_path = os.path.join(
            self.preview_directory,
            f'{record.sha256}-{maximum_points_per_axis}.npz',
        )
        measurement_path = self.get_absolute_path(record)
        file_stat = os.stat(measurement_path)
        cache_is_current = (
            int(file_stat.st_size) == record.file_size
            and int(file_stat.st_mtime_ns) == record.modified_ns
        )

        if cache_is_current and os.path.isfile(cache_path):
            try:
                with np.load(cache_path, allow_pickle=False) as cached_preview:
                    return HDF5MapPreview(
                        x=np.array(cached_preview['x'], copy=True),
                        y=np.array(cached_preview['y'], copy=True),
                        z=np.array(cached_preview['z'], copy=True),
                        x_label=str(cached_preview['x_label'].item()),
                        y_label=str(cached_preview['y_label'].item()),
                        z_label=str(cached_preview['z_label'].item()),
                    )
            except (OSError, KeyError, ValueError):
                try:
                    os.remove(cache_path)
                except OSError:
                    pass

        preview = load_hdf5_map_preview(
            measurement_path,
            maximum_points_per_axis=maximum_points_per_axis,
        )

        if not cache_is_current:
            return preview

        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f'.{record.sha256}.',
            suffix='.npz.tmp',
            dir=self.preview_directory,
        )
        try:
            with os.fdopen(descriptor, 'wb') as preview_file:
                np.savez_compressed(
                    preview_file,
                    x=preview.x,
                    y=preview.y,
                    z=preview.z,
                    x_label=np.asarray(preview.x_label),
                    y_label=np.asarray(preview.y_label),
                    z_label=np.asarray(preview.z_label),
                )
            os.replace(temporary_path, cache_path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
        return preview

    @staticmethod
    def _database_values(path, relative_path, metadata, file_hash, existing=None):
        """Build the stored values shared by scan and drag-and-drop imports."""
        file_stat = os.stat(path)
        now = _utc_now()
        return {
            'relative_path': relative_path,
            'file_name': os.path.basename(path),
            'file_size': int(file_stat.st_size),
            'modified_ns': int(file_stat.st_mtime_ns),
            'sha256': file_hash,
            'data_channel': metadata['log_names'][-1],
            'step_dimensions': json.dumps(list(metadata['step_dimensions'])),
            'has_traces': int(metadata['has_traces']),
            'added_at': existing['added_at'] if existing is not None else now,
            'updated_at': now,
        }

    @staticmethod
    def _upsert_measurement(connection, values):
        """Insert a measurement or refresh file-derived fields only."""
        connection.execute(
            """
            INSERT INTO measurements (
                relative_path, file_name, file_size, modified_ns, sha256,
                data_channel, step_dimensions, has_traces, added_at, updated_at
            ) VALUES (
                :relative_path, :file_name, :file_size, :modified_ns, :sha256,
                :data_channel, :step_dimensions, :has_traces, :added_at, :updated_at
            )
            ON CONFLICT(relative_path) DO UPDATE SET
                file_name = excluded.file_name,
                file_size = excluded.file_size,
                modified_ns = excluded.modified_ns,
                sha256 = excluded.sha256,
                data_channel = excluded.data_channel,
                step_dimensions = excluded.step_dimensions,
                has_traces = excluded.has_traces,
                updated_at = excluded.updated_at
            """,
            values,
        )

    def _candidate_files(self):
        """Collect regular, non-symlink HDF5 files below the database root."""
        candidates = []
        for directory, directory_names, file_names in os.walk(
            self.root_directory, followlinks=False
        ):
            directory_names[:] = [
                name
                for name in directory_names
                if name != DATABASE_METADATA_DIRECTORY
                and not os.path.islink(os.path.join(directory, name))
            ]
            for file_name in file_names:
                path = os.path.join(directory, file_name)
                if os.path.islink(path):
                    continue
                if os.path.splitext(file_name)[1].lower() in HDF5_FILE_EXTENSIONS:
                    candidates.append(path)
        return sorted(candidates, key=str.casefold)

    def scan(self, progress_callback=None, cancel_event=None):
        """Recursively validate and reconcile every measurement in the root."""
        self.initialize()
        candidates = self._candidate_files()
        valid_files = []
        invalid_files = []

        for index, path in enumerate(candidates, start=1):
            if cancel_event is not None and cancel_event.is_set():
                return DatabaseScanResult(
                    discovered=len(candidates),
                    indexed=len(valid_files),
                    invalid=tuple(invalid_files),
                    cancelled=True,
                )
            if progress_callback is not None:
                progress_callback(index - 1, len(candidates), path)
            try:
                metadata = inspect_viewer_hdf5(path)
                file_hash = _sha256_file(path)
                valid_files.append((path, metadata, file_hash))
            except (OSError, ValueError) as error:
                invalid_files.append((self._relative_path(path), str(error)))

        with self._connect() as connection:
            existing_rows = connection.execute('SELECT * FROM measurements').fetchall()
            existing_by_path = {row['relative_path']: row for row in existing_rows}
            valid_paths = {self._relative_path(path) for path, _, _ in valid_files}
            missing_rows = {
                path: row
                for path, row in existing_by_path.items()
                if path not in valid_paths
            }

            for path, metadata, file_hash in valid_files:
                relative_path = self._relative_path(path)
                existing = existing_by_path.get(relative_path)

                # Preserve comments and stars when a uniquely identifiable file
                # was renamed or moved within the managed database folder.
                if existing is None:
                    moved_matches = [
                        row
                        for row in missing_rows.values()
                        if row['sha256'] == file_hash
                    ]
                    if len(moved_matches) == 1:
                        moved_row = moved_matches[0]
                        connection.execute(
                            'UPDATE measurements SET relative_path = ? WHERE id = ?',
                            (relative_path, moved_row['id']),
                        )
                        missing_rows.pop(moved_row['relative_path'], None)
                        existing = moved_row

                values = self._database_values(
                    path, relative_path, metadata, file_hash, existing=existing
                )
                self._upsert_measurement(connection, values)

            for missing_row in missing_rows.values():
                connection.execute(
                    'DELETE FROM measurements WHERE id = ?',
                    (missing_row['id'],),
                )

        if progress_callback is not None:
            progress_callback(len(candidates), len(candidates), '')
        return DatabaseScanResult(
            discovered=len(candidates),
            indexed=len(valid_files),
            invalid=tuple(invalid_files),
        )

    def _unique_import_path(self, source_path):
        """Choose a collision-free destination in the managed import folder."""
        file_name = os.path.basename(source_path)
        stem, extension = os.path.splitext(file_name)
        candidate = os.path.join(self.import_directory, file_name)
        counter = 1
        while os.path.exists(candidate):
            candidate = os.path.join(
                self.import_directory, f'{stem} ({counter}){extension}'
            )
            counter += 1
        return candidate

    def import_dropped_file(self, source_path):
        """Copy a dropped file into the database, or return its existing copy."""
        self.initialize()
        source_path = os.path.abspath(os.path.expanduser(source_path))
        metadata = inspect_viewer_hdf5(source_path)
        source_hash = _sha256_file(source_path)
        existing_record = self.find_by_hash(source_hash)
        if existing_record is not None:
            return DatabaseImportResult(existing_record, already_present=True)

        destination_path = self._unique_import_path(source_path)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f'.{os.path.basename(destination_path)}.',
            suffix='.part',
            dir=self.import_directory,
        )
        os.close(descriptor)
        try:
            shutil.copy2(source_path, temporary_path)
            inspect_viewer_hdf5(temporary_path)
            if _sha256_file(temporary_path) != source_hash:
                raise OSError('The copied measurement failed its integrity check.')
            os.replace(temporary_path, destination_path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)

        relative_path = self._relative_path(destination_path)
        values = self._database_values(
            destination_path,
            relative_path,
            metadata,
            source_hash,
        )
        with self._connect() as connection:
            self._upsert_measurement(connection, values)
            row = connection.execute(
                'SELECT * FROM measurements WHERE relative_path = ?',
                (relative_path,),
            ).fetchone()
        return DatabaseImportResult(
            self._record_from_row(row), already_present=False
        )

    def set_starred(self, measurement_id, starred):
        """Persist the user's special-measurement marker."""
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE measurements
                SET starred = ?, updated_at = ?
                WHERE id = ?
                """,
                (int(bool(starred)), _utc_now(), int(measurement_id)),
            )
        if cursor.rowcount != 1:
            raise ValueError('The selected measurement is no longer in the database.')

    def set_comment(self, measurement_id, comment):
        """Persist the free-form comment belonging to a measurement."""
        if not isinstance(comment, str):
            raise ValueError('The measurement comment must be text.')
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE measurements
                SET comment = ?, updated_at = ?
                WHERE id = ?
                """,
                (comment, _utc_now(), int(measurement_id)),
            )
        if cursor.rowcount != 1:
            raise ValueError('The selected measurement is no longer in the database.')
