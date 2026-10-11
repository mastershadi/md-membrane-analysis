import numpy as np
from MDAnalysis.lib.distances import minimize_vectors


def charmm_sn1_pairs(n_carbons=16):
    """Carbon-hydrogen name pairs for the CHARMM36 sn-1 (palmitoyl, chain 3) tail.

    Returns ``{carbon_index: [(carbon_name, hydrogen_name), ...]}`` for carbons
    2..n_carbons, e.g. ``C32 -> H2X, H2Y`` and ``C316 -> H16X, H16Y, H16Z``.
    """
    pairs = {}
    for i in range(2, n_carbons + 1):
        hydrogens = ["X", "Y"] + (["Z"] if i == n_carbons else [])
        pairs[i] = [(f"C3{i}", f"H{i}{h}") for h in hydrogens]
    return pairs


def _p2(vectors, normal, box=None):
    if box is not None:
        vectors = minimize_vectors(vectors.astype(np.float32), box)
    unit = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    cos = unit @ normal
    return 0.5 * (3.0 * cos**2 - 1.0)


def scd_profile(universe, resname, pairs, normal=(0.0, 0.0, 1.0), frames=None):
    """Deuterium order parameter |S_CD| per carbon, averaged over lipids and frames.

    ``pairs`` is a mapping like the output of :func:`charmm_sn1_pairs`. Returns
    ``(carbon_indices, scd)`` as numpy arrays.
    """
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    selections = {}
    for idx, ch in pairs.items():
        carbons = [universe.select_atoms(f"resname {resname} and name {c}") for c, _ in ch]
        hydrogens = [universe.select_atoms(f"resname {resname} and name {h}") for _, h in ch]
        selections[idx] = list(zip(carbons, hydrogens))

    sums = {idx: 0.0 for idx in pairs}
    counts = {idx: 0 for idx in pairs}
    trajectory = universe.trajectory if frames is None else universe.trajectory[frames]
    for _ in trajectory:
        for idx, groups in selections.items():
            for carbons, hydrogens in groups:
                values = _p2(hydrogens.positions - carbons.positions, normal, universe.dimensions)
                sums[idx] += values.sum()
                counts[idx] += len(values)

    carbon_indices = np.array(sorted(pairs))
    scd = np.array([abs(sums[i] / counts[i]) for i in carbon_indices])
    return carbon_indices, scd


def martini_dppc_bonds():
    """Bead pairs along the two tails of a MARTINI 2 DPPC lipid."""
    return [
        ("PO4", "GL1"),
        ("GL1", "C1A"), ("C1A", "C2A"), ("C2A", "C3A"), ("C3A", "C4A"),
        ("GL2", "C1B"), ("C1B", "C2B"), ("C2B", "C3B"), ("C3B", "C4B"),
    ]


def bond_p2(universe, resname, bonds, normal=(0.0, 0.0, 1.0)):
    """Second-rank order parameter P2 = <(3cos²θ - 1)/2> for each bead pair.

    Typical for coarse-grained (MARTINI) lipids. 1 means aligned with the
    normal, -0.5 means perpendicular, 0 means random. Uses the current frame.
    Returns ``{"A-B": p2}``.
    """
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    result = {}
    for a, b in bonds:
        first = universe.select_atoms(f"resname {resname} and name {a}")
        second = universe.select_atoms(f"resname {resname} and name {b}")
        vectors = second.positions - first.positions
        result[f"{a}-{b}"] = float(_p2(vectors, normal, universe.dimensions).mean())
    return result
