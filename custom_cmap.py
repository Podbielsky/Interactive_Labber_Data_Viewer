from matplotlib.colors import LinearSegmentedColormap

def make_neon_cyclic_colormap():
    # Define colors
    neon_pink = [1, 0.078, 0.576]  # Neon Pink
    neon_blue_green = [0.047, 0.776, 0.729]  # Neon Blue-Green
    # Create colormap dictionary
    cmap_dict = {
        'red':   [(0.0, 1.0, 1.0),
                  (0.25, neon_pink[0], neon_pink[0]),
                  (0.5, 0, 0),
                  (0.75, neon_blue_green[0], neon_blue_green[0]),
                  (1.0, 1.0, 1.0)],
        'green': [(0.0, 1.0, 1.0),
                  (0.25, neon_pink[1], neon_pink[1]),
                  (0.5, 0, 0),
                  (0.75, neon_blue_green[1], neon_blue_green[1]),
                  (1.0, 1.0, 1.0)],
        'blue':  [(0.0, 1.0, 1.0),
                  (0.25, neon_pink[2], neon_pink[2]),
                  (0.5, 0, 0),
                  (0.75, neon_blue_green[2], neon_blue_green[2]),
                  (1.0, 1.0, 1.0)]
    }
    return LinearSegmentedColormap('NeonPiCy', cmap_dict)

def make_bi_colormap():
    # Define colors
    bi_pride = LinearSegmentedColormap.from_list("", ["#D60270", "#9B4F96", "#0038A8"])
    return bi_pride