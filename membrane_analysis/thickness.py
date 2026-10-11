import numpy as np
from MDAnalysis.lib.mdamath import triclinic_vectors


def mean_thickness(upper_z, lower_z):
    """Bilayer thickness as the distance between mean upper and lower headgroup heights."""
    return float(np.mean(upper_z) - np.mean(lower_z))


def _fractional_xy(positions, dimensions):
    vecs = triclinic_vectors(dimensions)
    frac = positions @ np.linalg.inv(vecs)
    return frac[:, :2] % 1.0


def _binned_mean_z(frac_xy, z, nbins):
    edges = np.linspace(0.0, 1.0, nbins + 1)
    total, _, _ = np.histogram2d(frac_xy[:, 0], frac_xy[:, 1], bins=[edges, edges], weights=z)
    count, _, _ = np.histogram2d(frac_xy[:, 0], frac_xy[:, 1], bins=[edges, edges])
    return total, count


def thickness_map(upper_frames, lower_frames, dimensions, nbins=12):
    """Average 2D thickness map over several frames.

    ``upper_frames`` and ``lower_frames`` are lists of (N, 3) headgroup position
    arrays, one per frame. Binning is done in fractional box coordinates so
    triclinic (for example hexagonal) boxes work. Returns ``(X, Y, T)`` where X and
    Y are Cartesian bin-edge grids of shape (nbins+1, nbins+1), suitable for
    ``matplotlib.pyplot.pcolormesh``, and T is the (nbins, nbins) thickness in Å
    with NaN for bins that never contained headgroups from both leaflets.
    """
    up_sum = np.zeros((nbins, nbins))
    up_n = np.zeros((nbins, nbins))
    lo_sum = np.zeros((nbins, nbins))
    lo_n = np.zeros((nbins, nbins))
    for up, lo in zip(upper_frames, lower_frames):
        s, n = _binned_mean_z(_fractional_xy(up, dimensions), up[:, 2], nbins)
        up_sum += s
        up_n += n
        s, n = _binned_mean_z(_fractional_xy(lo, dimensions), lo[:, 2], nbins)
        lo_sum += s
        lo_n += n

    with np.errstate(invalid="ignore", divide="ignore"):
        thickness = up_sum / up_n - lo_sum / lo_n
    thickness[(up_n == 0) | (lo_n == 0)] = np.nan

    a, b, _ = triclinic_vectors(dimensions)
    grid = np.linspace(0.0, 1.0, nbins + 1)
    sa, sb = np.meshgrid(grid, grid, indexing="ij")
    X = sa * a[0] + sb * b[0]
    Y = sa * a[1] + sb * b[1]
    return X, Y, thickness
