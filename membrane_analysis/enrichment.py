from collections import Counter

import numpy as np
from MDAnalysis.lib.distances import capped_distance


def depletion_enrichment(lipids, protein, cutoff=6.0):
    """Lipid depletion-enrichment index around a protein for the current frame.

    A lipid counts as "near" if any of its atoms in ``lipids`` is within ``cutoff``
    Å of any atom in ``protein`` (periodic boundaries respected). For each lipid
    type the index is (fraction of that type among near lipids) divided by
    (fraction of that type in the whole membrane): above 1 means enrichment,
    below 1 means depletion.

    Returns a dict ``{resname: (index, n_near, n_total)}``.
    """
    pairs = capped_distance(
        protein.positions,
        lipids.positions,
        max_cutoff=cutoff,
        box=lipids.dimensions,
        return_distances=False,
    )
    near_resindices = np.unique(lipids.resindices[pairs[:, 1]]) if len(pairs) else np.array([], int)

    residues = lipids.residues
    total = Counter(residues.resnames)
    near_mask = np.isin(residues.resindices, near_resindices)
    near = Counter(residues.resnames[near_mask])

    n_total = sum(total.values())
    n_near = sum(near.values())
    result = {}
    for resname, count in total.items():
        bulk_frac = count / n_total
        near_frac = near.get(resname, 0) / n_near if n_near else np.nan
        result[resname] = (near_frac / bulk_frac, near.get(resname, 0), count)
    return result
