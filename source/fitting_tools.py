"""Shared custom-expression curve fitting for traces and line cuts."""

import ast
from dataclasses import dataclass
import re

import numpy as np
from scipy import constants as co
from scipy import optimize

from Data_analysis_and_transforms import gaussian, lorentzian


DEFAULT_FIT_EXPRESSION = 'a*x + b'
FIT_EXPRESSION_EXAMPLE = 'a*np.exp(-x/b) + c + G1(x) + L1(x)'
DEFAULT_MAXFEV = 2200

DISTRIBUTION_MODELS = {
    'G': (gaussian, ('a', 'mu', 'sigma')),
    'L': (lorentzian, ('a', 'x0', 'gamma', 'c')),
}


def _natural_name_key(value):
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r'(\d+)', value)
    )


def default_initial_value(parameter_name):
    """Return a practical generic starting value for a fit parameter."""
    base_name = parameter_name.rsplit('_', 1)[-1].casefold()
    if base_name in {'c', 'mu', 'x0', 'offset'}:
        return 0.0
    return 1.0


def _attribute_root_name(node):
    while isinstance(node, ast.Attribute):
        if node.attr.startswith('_'):
            raise ValueError('Private attributes are not allowed in fit expressions.')
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _parse_expression(expression):
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError('Enter a model expression before fitting.')

    try:
        expression_tree = ast.parse(expression, mode='eval')
    except SyntaxError as error:
        raise ValueError(f'Invalid model expression: {error.msg}.') from error

    all_names = {
        node.id for node in ast.walk(expression_tree) if isinstance(node, ast.Name)
    }
    called_names = {
        node.func.id
        for node in ast.walk(expression_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    distribution_names = sorted(
        (
            name for name in all_names
            if re.fullmatch(r'[GL]\d+', name)
        ),
        key=_natural_name_key,
    )
    for distribution_name in distribution_names:
        if distribution_name not in called_names:
            raise ValueError(
                f'{distribution_name} must be called as {distribution_name}(x).'
            )

    unsupported_calls = sorted(
        called_names.difference(distribution_names),
        key=_natural_name_key,
    )
    if unsupported_calls:
        raise ValueError(
            'Unsupported function name(s): '
            + ', '.join(unsupported_calls)
            + '. Use NumPy functions such as np.exp(x), or G1(x)/L1(x).'
        )

    for node in ast.walk(expression_tree):
        if isinstance(node, ast.Attribute):
            root_name = _attribute_root_name(node)
            if root_name not in {'np', 'co'}:
                raise ValueError(
                    'Only NumPy (np) and scipy.constants (co) attributes are '
                    'allowed in model expressions.'
                )
        if isinstance(node, ast.Name) and node.id.startswith('_'):
            raise ValueError('Private names are not allowed in fit expressions.')

    reserved_names = {'x', 'np', 'co', *distribution_names}
    direct_parameter_names = sorted(
        all_names.difference(reserved_names),
        key=_natural_name_key,
    )
    distribution_parameter_names = []
    for distribution_name in distribution_names:
        distribution_key = distribution_name[0]
        distribution_parameter_names.extend(
            f'{distribution_name}_{parameter_name}'
            for parameter_name in DISTRIBUTION_MODELS[distribution_key][1]
        )

    parameter_names = direct_parameter_names + distribution_parameter_names
    if not parameter_names:
        raise ValueError('The model expression does not contain any fit parameters.')

    return (
        compile(expression_tree, '<fit expression>', 'eval'),
        tuple(direct_parameter_names),
        tuple(distribution_names),
        tuple(parameter_names),
    )


@dataclass(frozen=True)
class CurveFitResult:
    """Result returned by a shared line-cut or trace fit."""

    parameter_names: tuple
    parameter_values: np.ndarray
    parameter_errors: np.ndarray
    covariance: np.ndarray
    fitted_y: np.ndarray

    @property
    def parameters(self):
        return dict(zip(self.parameter_names, self.parameter_values))


class FitModelDefinition:
    """Compiled custom model expression with stable parameter ordering."""

    def __init__(self, expression):
        self.expression = expression.strip()
        (
            self._compiled_expression,
            self.direct_parameter_names,
            self.distribution_names,
            self.parameter_names,
        ) = _parse_expression(self.expression)

    def initial_vector(self, initial_values):
        if isinstance(initial_values, dict):
            missing_names = [
                name for name in self.parameter_names if name not in initial_values
            ]
            if missing_names:
                raise ValueError(
                    'Missing initial value(s): ' + ', '.join(missing_names)
                )
            values = [initial_values[name] for name in self.parameter_names]
        else:
            values = initial_values

        vector = np.asarray(values, dtype=float).reshape(-1)
        if vector.size != len(self.parameter_names):
            raise ValueError(
                f'Expected {len(self.parameter_names)} initial values, got '
                f'{vector.size}.'
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError('All initial parameter values must be finite numbers.')
        return vector

    def __call__(self, x_data, *parameter_values):
        if len(parameter_values) != len(self.parameter_names):
            raise ValueError(
                f'Expected {len(self.parameter_names)} parameters, got '
                f'{len(parameter_values)}.'
            )

        parameter_lookup = dict(zip(self.parameter_names, parameter_values))
        namespace = {
            'x': x_data,
            'np': np,
            'co': co,
        }
        for parameter_name in self.direct_parameter_names:
            namespace[parameter_name] = parameter_lookup[parameter_name]

        for distribution_name in self.distribution_names:
            distribution_function, distribution_parameters = (
                DISTRIBUTION_MODELS[distribution_name[0]]
            )
            distribution_values = tuple(
                parameter_lookup[f'{distribution_name}_{parameter_name}']
                for parameter_name in distribution_parameters
            )
            namespace[distribution_name] = (
                lambda values=distribution_values, function=distribution_function:
                lambda x_value: function(x_value, *values)
            )()

        return eval(
            self._compiled_expression,
            {'__builtins__': {}},
            namespace,
        )

    def fit(self, x_data, y_data, initial_values, maxfev=DEFAULT_MAXFEV):
        """Fit this model to finite x/y samples and return a stable result."""
        x_array = np.asarray(x_data, dtype=float).reshape(-1)
        y_array = np.asarray(y_data, dtype=float).reshape(-1)
        if x_array.size != y_array.size:
            raise ValueError(
                f'x and y must contain the same number of samples; got '
                f'{x_array.size} and {y_array.size}.'
            )

        finite_mask = np.isfinite(x_array) & np.isfinite(y_array)
        finite_count = int(np.count_nonzero(finite_mask))
        if finite_count <= len(self.parameter_names):
            raise ValueError(
                f'The fit needs more finite samples than parameters; got '
                f'{finite_count} samples and {len(self.parameter_names)} parameters.'
            )

        try:
            max_function_evaluations = int(maxfev)
        except (TypeError, ValueError) as error:
            raise ValueError('maxfev must be a positive integer.') from error
        if max_function_evaluations <= 0:
            raise ValueError('maxfev must be a positive integer.')

        initial_vector = self.initial_vector(initial_values)
        fitted_values, covariance = optimize.curve_fit(
            self,
            x_array[finite_mask],
            y_array[finite_mask],
            p0=initial_vector,
            maxfev=max_function_evaluations,
        )

        fitted_y = np.full(y_array.shape, np.nan, dtype=float)
        fitted_y[finite_mask] = np.asarray(
            self(x_array[finite_mask], *fitted_values),
            dtype=float,
        )
        with np.errstate(invalid='ignore'):
            parameter_errors = np.sqrt(np.diag(covariance))

        return CurveFitResult(
            parameter_names=self.parameter_names,
            parameter_values=np.asarray(fitted_values, dtype=float),
            parameter_errors=np.asarray(parameter_errors, dtype=float),
            covariance=np.asarray(covariance, dtype=float),
            fitted_y=fitted_y,
        )


def format_fit_result(result, elapsed_seconds=None):
    """Return consistent parameter text for line-cut and trace fitters."""
    lines = ['Fitted parameters:']
    for name, value, error in zip(
        result.parameter_names,
        result.parameter_values,
        result.parameter_errors,
    ):
        if np.isfinite(error):
            lines.append(f'{name} = {value:.6g} ± {error:.3g}')
        else:
            lines.append(f'{name} = {value:.6g}')
    if elapsed_seconds is not None:
        lines.extend(('', f'Fit time: {elapsed_seconds:.3f} seconds'))
    return '\n'.join(lines)
