import numpy as np


def assign_leaflets(head_z):
    """Return a boolean mask that is True for headgroups in the upper leaflet.

    Uses the mean headgroup height as the bilayer midplane, which is reliable for
    flat bilayers that are not split across the periodic boundary in z.
    """
    head_z = np.asarray(head_z, dtype=float)
    if head_z.size < 2:
        raise ValueError("Need at least two headgroups to split leaflets")
    return head_z > head_z.mean()
