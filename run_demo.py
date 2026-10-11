"""Run the full analysis on two public test systems shipped with MDAnalysisTests.

1. YiiP zinc transporter in a POPE/POPG bilayer (all-atom CHARMM36, 5 frames).
2. DPPC/cholesterol bilayer (MARTINI coarse-grained, 1 frame).

Writes CSV tables to ``results/`` and figures to ``figures/``.
"""

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysisTests.datafiles import GRO_MEMPROT, XTC_MEMPROT, Martini_membrane_gro

from membrane_analysis import (
    area_per_lipid,
    assign_leaflets,
    bond_p2,
    charmm_sn1_pairs,
    depletion_enrichment,
    martini_dppc_bonds,
    mean_thickness,
    scd_profile,
    thickness_map,
)

warnings.filterwarnings("ignore", category=UserWarning)


def analyse_yiip(results, figures, cutoff):
    u = mda.Universe(GRO_MEMPROT, XTC_MEMPROT)
    heads = u.select_atoms("resname POPE POPG and name P")
    lipids = u.select_atoms("resname POPE POPG and not name H*")
    protein = u.select_atoms("protein and not name H*")

    rows, enrich_rows = [], []
    upper_frames, lower_frames = [], []
    for ts in u.trajectory:
        upper = assign_leaflets(heads.positions[:, 2])
        up_pos, lo_pos = heads.positions[upper], heads.positions[~upper]
        upper_frames.append(up_pos.copy())
        lower_frames.append(lo_pos.copy())
        rows.append({
            "time_ns": ts.time / 1000.0,
            "thickness_PP_A": mean_thickness(up_pos[:, 2], lo_pos[:, 2]),
            "apparent_APL_upper_A2": area_per_lipid(ts.dimensions, int(upper.sum())),
            "apparent_APL_lower_A2": area_per_lipid(ts.dimensions, int((~upper).sum())),
        })
        for resname, (index, n_near, n_total) in depletion_enrichment(lipids, protein, cutoff).items():
            enrich_rows.append({"time_ns": ts.time / 1000.0, "lipid": resname,
                                "DE_index": index, "n_near": n_near, "n_total": n_total})

    per_frame = pd.DataFrame(rows)
    per_frame.to_csv(results / "yiip_per_frame.csv", index=False)
    enrichment = pd.DataFrame(enrich_rows)
    enrichment.to_csv(results / "yiip_enrichment.csv", index=False)

    X, Y, T = thickness_map(upper_frames, lower_frames, u.dimensions, nbins=12)
    ca = u.select_atoms("protein and name CA").wrap(inplace=False)
    fig, ax = plt.subplots(figsize=(6.2, 5))
    mesh = ax.pcolormesh(X, Y, T, cmap="viridis", shading="flat")
    ax.scatter(ca[:, 0], ca[:, 1], s=2, c="crimson", alpha=0.5, label="protein Cα")
    fig.colorbar(mesh, ax=ax, label="P–P thickness (Å)")
    ax.set_aspect("equal")
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_title("YiiP in POPE/POPG: bilayer thickness map")
    ax.legend(loc="upper left", fontsize=8, markerscale=4)
    fig.tight_layout()
    fig.savefig(figures / "thickness_map.png", dpi=150)
    plt.close(fig)

    summary = enrichment.groupby("lipid")["DE_index"].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.bar(summary.index, summary["mean"], yerr=summary["std"], capsize=6,
           color=["#4C72B0", "#DD8452"])
    ax.axhline(1.0, color="k", lw=1, ls="--")
    ax.set_ylabel("Depletion-enrichment index")
    ax.set_title(f"Lipids within {cutoff:g} Å of YiiP")
    fig.tight_layout()
    fig.savefig(figures / "lipid_enrichment.png", dpi=150)
    plt.close(fig)

    order = {}
    for resname in ("POPE", "POPG"):
        carbons, scd = scd_profile(u, resname, charmm_sn1_pairs())
        order[resname] = scd
    order_df = pd.DataFrame({"carbon": carbons, **{f"SCD_{k}_sn1": v for k, v in order.items()}})
    order_df.to_csv(results / "yiip_scd_sn1.csv", index=False)
    return per_frame, summary, order_df


def analyse_martini(results):
    u = mda.Universe(Martini_membrane_gro)
    heads = u.select_atoms("resname DPPC and name PO4")
    upper = assign_leaflets(heads.positions[:, 2])
    p2 = bond_p2(u, "DPPC", martini_dppc_bonds())
    pd.DataFrame({"bond": list(p2), "P2": list(p2.values())}).to_csv(
        results / "martini_dppc_p2.csv", index=False)
    stats = {
        "thickness_PO4_A": mean_thickness(heads.positions[upper, 2], heads.positions[~upper, 2]),
        "APL_DPPC_plus_CHOL_A2": area_per_lipid(
            u.dimensions, len(u.select_atoms("resname DPPC CHOL").residues) / 2),
    }
    return p2, stats


def plot_order(order_df, p2, figures):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for column, label in [("SCD_POPE_sn1", "POPE"), ("SCD_POPG_sn1", "POPG")]:
        ax1.plot(order_df["carbon"], order_df[column], "o-", label=label)
    ax1.set_xlabel("Carbon number (sn-1 palmitoyl)")
    ax1.set_ylabel(r"$|S_{CD}|$")
    ax1.set_title("All-atom: acyl chain order")
    ax1.legend()

    ax2.bar(list(p2), list(p2.values()), color="#55A868")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_ylabel(r"$P_2$")
    ax2.set_title("MARTINI DPPC/CHOL: bond order")
    ax2.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(figures / "order_parameters.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("."), help="output directory")
    parser.add_argument("--cutoff", type=float, default=6.0, help="enrichment cutoff in Å")
    args = parser.parse_args()

    results = args.out / "results"
    figures = args.out / "figures"
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    per_frame, enrichment, order_df = analyse_yiip(results, figures, args.cutoff)
    p2, martini_stats = analyse_martini(results)
    plot_order(order_df, p2, figures)

    print("YiiP / POPE:POPG, per frame")
    print(per_frame.round(2).to_string(index=False))
    print("\nDepletion-enrichment index (mean ± std over frames)")
    print(enrichment.round(2).to_string())
    print("\nMARTINI DPPC/CHOL")
    for key, value in martini_stats.items():
        print(f"  {key}: {value:.2f}")


if __name__ == "__main__":
    main()
