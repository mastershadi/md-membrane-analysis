import MDAnalysis as mda
import numpy as np
import pytest

from membrane_analysis import (
    area_per_lipid,
    assign_leaflets,
    bond_p2,
    box_xy_area,
    charmm_sn1_pairs,
    depletion_enrichment,
    mean_thickness,
    scd_profile,
    thickness_map,
)

CUBIC = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0])
HEXAGONAL = np.array([100.0, 100.0, 120.0, 90.0, 90.0, 120.0])


def flat_bilayer(n_side=10, z_up=70.0, z_lo=30.0):
    grid = (np.arange(n_side) + 0.5) * CUBIC[0] / n_side
    xx, yy = np.meshgrid(grid, grid)
    xy = np.column_stack([xx.ravel(), yy.ravel()])
    up = np.column_stack([xy, np.full(len(xy), z_up)])
    lo = np.column_stack([xy, np.full(len(xy), z_lo)])
    return up, lo


def make_universe(resnames, atoms_per_res, names, positions, box=CUBIC):
    n_res = len(resnames)
    u = mda.Universe.empty(
        n_atoms=n_res * atoms_per_res,
        n_residues=n_res,
        atom_resindex=np.repeat(np.arange(n_res), atoms_per_res),
        trajectory=True,
    )
    u.add_TopologyAttr("resname", resnames)
    u.add_TopologyAttr("name", list(names) * n_res)
    u.atoms.positions = positions
    u.dimensions = box
    return u


def test_assign_leaflets_splits_two_planes():
    up, lo = flat_bilayer()
    mask = assign_leaflets(np.concatenate([up[:, 2], lo[:, 2]]))
    assert mask[: len(up)].all() and not mask[len(up):].any()


def test_assign_leaflets_rejects_single_lipid():
    with pytest.raises(ValueError):
        assign_leaflets([1.0])


def test_mean_thickness_of_flat_bilayer():
    up, lo = flat_bilayer(z_up=72.0, z_lo=32.0)
    assert mean_thickness(up[:, 2], lo[:, 2]) == pytest.approx(40.0)


def test_thickness_map_is_uniform_for_flat_bilayer():
    up, lo = flat_bilayer()
    X, Y, T = thickness_map([up], [lo], CUBIC, nbins=5)
    assert X.shape == Y.shape == (6, 6)
    assert np.allclose(T, 40.0)


def test_box_area_handles_hexagonal_box():
    assert box_xy_area(HEXAGONAL) == pytest.approx(100.0 * 100.0 * np.sin(np.radians(60)), rel=1e-5)


def test_area_per_lipid_with_excluded_area():
    assert area_per_lipid(CUBIC, 100, excluded_area=2000.0) == pytest.approx(80.0)


def test_scd_is_half_for_bonds_in_membrane_plane():
    pairs = charmm_sn1_pairs(n_carbons=3)
    names = ["C32", "H2X", "H2Y", "C33", "H3X", "H3Y", "H3Z"]
    pos = np.array([
        [50, 50, 50], [51, 50, 50], [50, 51, 50],
        [50, 50, 52], [51, 50, 52], [49, 50, 52], [50, 49, 52],
    ], dtype=float)
    u = make_universe(["TST"], len(names), names, pos)
    carbons, scd = scd_profile(u, "TST", pairs)
    assert list(carbons) == [2, 3]
    assert np.allclose(scd, 0.5)


def test_bond_p2_respects_periodic_boundaries():
    # Bond crosses the z boundary: raw difference is -99 Å, minimum image is +1 Å.
    pos = np.array([[50, 50, 99.5], [50, 50, 0.5]], dtype=float)
    u = make_universe(["TST"], 2, ["A", "B"], pos)
    assert bond_p2(u, "TST", [("A", "B")])["A-B"] == pytest.approx(1.0)


def test_depletion_enrichment_detects_enriched_lipid():
    # One protein atom at the centre; both PG lipids touch it, the PE lipids are far away.
    resnames = ["PRT", "PG", "PG", "PE", "PE", "PE", "PE"]
    pos = np.array([
        [50, 50, 50],
        [53, 50, 50], [47, 50, 50],
        [10, 10, 50], [90, 10, 50], [10, 90, 50], [90, 90, 50],
    ], dtype=float)
    u = make_universe(resnames, 1, ["X"], pos)
    lipids = u.select_atoms("resname PG PE")
    protein = u.select_atoms("resname PRT")
    result = depletion_enrichment(lipids, protein, cutoff=6.0)
    index_pg, n_near_pg, n_total_pg = result["PG"]
    assert (n_near_pg, n_total_pg) == (2, 2)
    assert index_pg == pytest.approx(3.0)
    assert result["PE"][0] == pytest.approx(0.0)
