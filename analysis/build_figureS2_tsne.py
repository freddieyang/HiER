"""Render Supplementary Figure S2 from the fixed all-entity t-SNE coordinates."""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MPLCONFIGDIR"] = "/tmp/hier_fig3_tsne"
from pathlib import Path
import argparse
from io_utils import prepare_output
import json
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = prepare_output(args.output_dir, [args.data_dir])
    points = pd.read_csv(args.data_dir / "figure3_tsne_coordinates.csv")
    assert len(points) > 0
    assert not points.duplicated(["entity_type", "entity_id"]).any()
    assert set(points.entity_type) == {"Enzyme", "Reaction"}
    assert points.displayed.eq(True).all()
    points["reference_L1"] = points.reference_L1.astype(str)
    colors = {"1": "#0072B2", "2": "#D99E30", "3": "#00866A", "4": "#9069A9",
              "5": "#C45131", "6": "#46A6BE", "7": "#CC79A7", "Unresolved": "#BAC1C5"}
    assert set(points.reference_L1) <= set(colors)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7,
        "axes.titlesize": 7.3, "axes.labelsize": 6.5, "axes.linewidth": .55,
        "text.color": "#263A43", "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 2, figsize=(3.44, 1.88))
    display_audit = []
    for ax, space, title in zip(axes, ["latent", "joint"],
                              ["a  Latent space", "b  Full retrieval space"]):
        # All enzyme points and all reaction points are drawn, even if they overlap.
        for modality, marker, size, linewidth, edge, zorder in [
            ("Enzyme", "o", 1.5, 0, "none", 2),
            ("Reaction", "^", 10, .20, "white", 3),
        ]:
            selected = points[points.entity_type == modality]
            collection = ax.scatter(
                selected[f"{space}_tsne_1"], selected[f"{space}_tsne_2"],
                c=selected.reference_L1.map(colors), s=size, alpha=.85,
                linewidths=linewidth, edgecolors=edge, marker=marker,
                rasterized=False, zorder=zorder,
            )
            assert len(collection.get_offsets()) == len(selected)
            display_audit.append({"space": space, "entity_type": modality,
                                  "displayed": len(collection.get_offsets())})
        ax.set_title(title, loc="left", pad=5, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(1)
        ax.margins(.055)
        for spine in ax.spines.values():
            spine.set_color("#5F6B70")
    class_handles = [Line2D([], [], marker="o", ls="", color=color, ms=3,
                           label="EC " + label if label != "Unresolved" else "Unresolved")
                     for label, color in colors.items()]
    class_legend = fig.legend(
        handles=class_handles, ncol=8, loc="upper center", bbox_to_anchor=(.505, 1.01),
        frameon=False, fontsize=6.1, handlelength=.55, handletextpad=.28,
        columnspacing=.63, borderaxespad=0,
    )
    # Marker meanings and entity counts are given in the manuscript caption.
    # Keeping that text outside the axes prevents it from obscuring any point.
    fig.subplots_adjust(left=.025, right=.989, bottom=.02, top=.82, wspace=.10)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(args.output_dir / f"figureS2_tsne.{ext}",
                    dpi=400, bbox_inches="tight", pad_inches=.025)
    plt.close(fig)
    summary = {
        "projection": "t-SNE", "displayed_entities_per_panel": len(points),
        "displayed_reactions_per_panel": int(points.entity_type.eq("Reaction").sum()),
        "displayed_enzymes_per_panel": int(points.entity_type.eq("Enzyme").sum()),
        "subsampling_during_rendering": False,
        "all_coordinate_rows_drawn": True, "class_counts": points.reference_L1.value_counts().to_dict(),
        "reference_labels_used_only_for_color": True,
        "source_coordinates": "figure3_tsne_coordinates.csv", "panel_counts": display_audit,
    }
    (args.output_dir / "figure3_display_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
