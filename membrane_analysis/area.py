import numpy as np
from MDAnalysis.lib.mdamath import triclinic_vectors


def box_xy_area(dimensions):
    """Area of the box face spanned by the first two box vectors (Å²)."""
    a, b, _ = triclinic_vectors(dimensions)
    return float(abs(np.cross(a, b)[2]))


def area_per_lipid(dimensions, n_lipids_leaflet, excluded_area=0.0):
    """Area per lipid in one leaflet (Å²).

    ``excluded_area`` lets you subtract an estimate of the protein cross-section;
    without it the result is the apparent area per lipid.
    """
    if n_lipids_leaflet <= 0:
        raise ValueError("n_lipids_leaflet must be positive")
    return (box_xy_area(dimensions) - excluded_area) / n_lipids_leaflet
