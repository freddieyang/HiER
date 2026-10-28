"""Shared, physical-unit layouts for the combined and standalone Figure 4 panels."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter
import numpy as np

BLUE, GREEN, ORANGE, GREY = "#0072B2", "#009E73", "#D55E00", "#8A8F95"
WIDTH = 7.2
HEIGHTS = {"a": 2.95, "b": 2.40, "c": 3.03}
GUTTER = .18
STEMS = {"a": "figure4a_ec_probability_trees",
         "b": "figure4b_candidate_score_components",
         "c": "figure4c_level_score_contributions"}


class Panel:
    """Place artists in inches measured from the panel's upper-left corner."""

    def __init__(self, fig, top, height):
        self.fig, self.top, self.height = fig, top, height
        self.figure_height = fig.get_figheight()

    def text(self, x, y, text, **kwargs):
        kwargs.setdefault("va", "center")
        return self.fig.text(x / WIDTH, 1 - (self.top + y) / self.figure_height,
                             text, **kwargs)

    def axes(self, x, y, width, height):
        return self.fig.add_axes([x / WIDTH, 1 - (self.top + y + height) / self.figure_height,
                                  width / WIDTH, height / self.figure_height])

    def header(self, letter, title):
        self.text(.10, .12, letter, fontsize=13, weight="bold")
        self.text(.37, .12, title, fontsize=9.3, weight="bold")


def percent(p):
    return "<0.01%" if p < .0001 else (f"{100*p:.2f}%" if p < .01 else f"{100*p:.1f}%")


def draw_a(panel, trees, scores, metrics):
    panel.header("a", "Shared EC predictions, different probability distributions")
    entities = trees.entity_id.drop_duplicates().tolist()
    assert len(entities) == 3, "Tree panel needs one query and two candidates in display order"
    assert entities[0] not in set(scores.candidate_id)
    assert set(entities[1:]) <= set(scores.candidate_id)
    first = trees[trees.entity_id == entities[0]]
    branches = []
    for level in [1, 2, 3]:
        nodes = first[(first.level == level) & ~first.other].node.tolist()
        assert len(nodes) == 1, "Select one shared branch at each of L1-L3"
        branches.append(nodes[0])
    leaves = first[first.level == 4].node.tolist()
    assert len(leaves) == 5, "This layout shows four L4 leaves and their remaining mass"
    positive_ids = set(scores.loc[scores.recorded_positive, "candidate_id"])
    titles = ["Reaction query"] + [entity + ("*" if entity in positive_ids else "")
                                     for entity in entities[1:]]
    panel.text(.37, .35, f"L4 leaf labels show suffixes of EC {branches[-1]}.",
               fontsize=7.3, color="#555B61")
    panel.text(7.03, .35, "* Recorded positive", ha="right", fontsize=7.1, color="#555B61")
    for i, (entity, title, color) in enumerate(zip(entities, titles, [BLUE, GREEN, ORANGE])):
        left = .10 + i*2.36
        panel.text(left+1.11, .59, title, ha="center", fontsize=9, weight="bold", color=color)
        ax = panel.axes(left, .78, 2.22, 2.0)
        ax.set(xlim=(0, 2.22), ylim=(2.0, 0))
        ax.axis("off")
        frame = trees[trees.entity_id == entity]
        probability = frame.set_index("node").probability.to_dict()
        assert set(frame.node) == set(first.node), "Use the same displayed tree for all entities"
        positions = {"root": (.78, .07)}
        for level, (branch, y) in enumerate(zip(branches, [.38, .74, 1.10]), 1):
            positions[branch] = (.78, y)
            other = frame[(frame.level == level) & frame.other].node.tolist()
            assert len(other) == 1
            positions[other[0]] = (1.66, y)
        positions.update({node: (x, 1.54) for node, x in zip(leaves, np.linspace(.13, 2.05, 5))})
        for r in frame.itertuples():
            x0, y0 = positions[r.parent]
            x1, y1 = positions[r.node]
            c = GREY if r.other else color
            ax.plot([x0, x1], [y0, y1], color=c, linewidth=.8, alpha=.65, zorder=1)
            ax.scatter(x1, y1, s=140*r.probability, facecolor=c, edgecolor=c, linewidth=.4, zorder=3)
            if r.level < 4 and not r.other:
                ax.text(x1-.17, y1, r.node, ha="right", va="center", fontsize=8, weight="bold")
                ax.text(x1+.16, y1-.04, percent(r.probability), ha="left", va="center", fontsize=7.2)
            elif r.level < 4:
                ax.text(x1+.10, y1, f"Other\n{percent(r.probability)}", ha="left", va="center",
                        fontsize=6.7, color="#555B61", linespacing=1.25)
        ax.text(.78, -.02, "All ECs", ha="center", va="center", fontsize=7)
        for node in leaves:
            x, _ = positions[node]
            label = "Other" if node.startswith("Other") else "." + node.split(".")[-1]
            ax.text(x, 1.74, label, ha="center", va="top", fontsize=7.4)
            ax.text(x, 1.92, percent(probability[node]), ha="center", va="top", fontsize=7.1)
        if i == 0:
            for y, label in [(.38, "L1"), (.74, "L2"), (1.10, "L3")]:
                ax.text(.02, y, label, ha="left", va="center", fontsize=6.5, color="#555B61")


def draw_b(panel, trees, scores, metrics):
    panel.header("b", "Candidate-level contributions to the joint score")
    ax = panel.axes(1.21, .60, 5.83, 1.14)
    values = scores[["latent", "L1", "L2", "L3", "L4"]].to_numpy()
    norms = [Normalize(0, 13), Normalize(0, 2.1)]
    maps = [plt.get_cmap("Blues"), plt.get_cmap("GnBu")]
    ax.set(xlim=(-.5, 6.95), ylim=(len(scores)-.5, -.5))
    ax.axis("off")
    for row, candidate in enumerate(scores.itertuples()):
        for col in range(5):
            index = int(col > 0)
            color = maps[index](norms[index](values[row, col]))
            ax.add_patch(Rectangle((col-.5, row-.5), 1, 1, facecolor=color,
                                   edgecolor="white", linewidth=.45))
            ax.text(col, row, f"{values[row,col]:.2f}", ha="center", va="center", color="white", fontsize=8)
        ax.text(5.45, row, f"{candidate.latent_rank} → {candidate.full_rank}", ha="center", va="center", fontsize=8)
        ax.text(6.55, row, f"{candidate.archived_full_score:.2f}", ha="center", va="center", fontsize=8)
        panel.text(1.12, .60 + (row+.5)*1.14/len(scores),
                   candidate.candidate_id + ("*" if candidate.recorded_positive else ""),
                   ha="right", fontsize=8)
    ax.axvline(.5, color="white", linewidth=3)
    for col, label in zip([0, 1, 2, 3, 4, 5.45, 6.55],
                          ["Latent", "L1", "L2", "L3", "L4", "Rank\nlatent → full", "Full\nscore"]):
        x = 1.21 + (col+.5)*5.83/7.45
        panel.text(x, .43, label, ha="center", fontsize=7.5, linespacing=1.3)
    for left, width, norm, cmap, label in [(1.55, 1.10, norms[0], maps[0], "Latent contribution"),
                                          (3.50, 1.60, norms[1], maps[1], "EC contribution (shared scale)")]:
        panel.text(left+width/2, 1.97, label, ha="center", fontsize=7)
        cbax = panel.axes(left, 2.10, width, .055)
        cb = panel.fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbax,
                               orientation="horizontal")
        cb.solids.set_rasterized(False)
        cb.solids.set_edgecolor("face")
        cb.set_ticks([norm.vmin, norm.vmax])
        cb.ax.tick_params(labelsize=6.5, length=1.5, pad=3)
        cb.outline.set_linewidth(.4)
    panel.text(6.85, 2.08, "* Recorded positive", ha="right", fontsize=7)


def draw_c(panel, trees, scores, metrics):
    panel.header("c", "Contribution of individual EC score blocks")
    handles = [Line2D([], [], color=BLUE, marker="o", lw=0, ms=4),
               Line2D([], [], color=GREEN, marker="^", lw=0, ms=4)]
    panel.fig.legend(handles, ["R → E", "E → R"], loc="center right",
                     bbox_to_anchor=(.98, 1-(panel.top+.12)/panel.figure_height),
                     frameon=False, fontsize=7.5, ncol=2, columnspacing=1.2, handletextpad=.4)
    for col, (split, title) in enumerate([("time", "Time"), ("seq_smi", "Enzyme similarity"), ("mol_smi", "Reaction similarity")]):
        left = .65 + col*2.19
        panel.text(left+.91, .42, title, ha="center", fontsize=8.3)
        for row, (metric, multiplier, ylim, yticks) in enumerate([
            ("delta_hit1", 100, (-.65, 3.1), [0, 1, 2, 3]),
            ("delta_mrr", 1, (-.0035, .0135), [0, .005, .010]),
        ]):
            ax = panel.axes(left, .59 if row == 0 else 1.72, 1.82, .82)
            ax.axhline(0, color="#9A9A9A", linewidth=.6, linestyle=(0, (3, 2)))
            for direction, color, marker, offset in [("R-E", BLUE, "o", -.15), ("E-R", GREEN, "^", .15)]:
                for level in range(1, 5):
                    data = metrics[(metrics.split == split) & (metrics.direction == direction) &
                                   (metrics.condition == f"without_L{level}")].sort_values("seed")
                    assert data.seed.tolist() == [42, 43, 44]
                    y = data[metric].to_numpy()*multiplier
                    x = level + offset
                    ax.scatter(x + np.array([-.052, 0, .052]), y, s=12, color=color,
                               marker=marker, alpha=.85, linewidths=.25, zorder=3)
                    ax.plot([x-.115, x+.115], [y.mean(), y.mean()], color=color, lw=1.6, zorder=4)
            ax.set(xlim=(.5, 4.5), ylim=ylim, xticks=range(1, 5),
                   xticklabels=["L1", "L2", "L3", "L4"], yticks=yticks)
            ax.tick_params(labelsize=7, length=2, pad=3)
            if row == 1:
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else f"{v:.3f}"))
            if col == 0:
                ax.set_ylabel("ΔHit@1 (pp)" if row == 0 else "ΔMRR", fontsize=8, labelpad=5)
            ax.spines["left"].set_color("#BBBBBB")
            ax.spines["bottom"].set_color("#BBBBBB")
    panel.text(WIDTH/2, 2.91, "Small symbols: three seeds; bars: means.  Δ = full − without the indicated EC level.",
               ha="center", fontsize=7.1)


def audit_text(fig):
    """Measure actual rendered text, including ticks and axis labels."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    objects = []
    for artist in fig.findobj(Text):
        if artist.get_visible() and artist.get_text().strip():
            bbox = artist.get_window_extent(renderer)
            if bbox.width and bbox.height:
                objects.append((artist.get_text(), bbox))
    overlaps = []
    for i, (text, box) in enumerate(objects):
        for other, b in objects[i+1:]:
            w = min(box.x1, b.x1) - max(box.x0, b.x0)
            h = min(box.y1, b.y1) - max(box.y0, b.y0)
            if w > .75*fig.dpi/72 and h > .75*fig.dpi/72:
                overlaps.append([text, other])
    outside = [text for text, box in objects if box.x0 < 0 or box.y0 < 0 or
               box.x1 > fig.bbox.width or box.y1 > fig.bbox.height]
    return {"text_count": len(objects), "overlapping_text_pairs": overlaps,
            "text_outside_canvas": outside}


def render_hierarchy_panels(trees, scores, metrics, output_dir):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.linewidth": .6, "axes.spines.top": False,
                         "axes.spines.right": False, "xtick.major.width": .6,
                         "ytick.major.width": .6, "text.color": "#222222",
                         "axes.labelcolor": "#222222", "xtick.color": "#333333",
                         "ytick.color": "#333333"})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    draw = {"a": draw_a, "b": draw_b, "c": draw_c}
    layouts = {"figure4_hierarchical_matching": ["a", "b", "c"]}
    layouts.update({STEMS[k]: [k] for k in "abc"})
    report = {}
    for stem, letters in layouts.items():
        height = sum(HEIGHTS[k] for k in letters) + GUTTER*(len(letters)-1) + .08
        fig = plt.figure(figsize=(WIDTH, height), facecolor="white")
        top = .04
        bands = {}
        for letter in letters:
            bands[letter] = {"top_inches": top, "height_inches": HEIGHTS[letter]}
            draw[letter](Panel(fig, top, HEIGHTS[letter]), trees, scores, metrics)
            top += HEIGHTS[letter] + GUTTER
        report[stem] = {**audit_text(fig), "width_inches": WIDTH, "height_inches": height,
                        "panel_bands": bands}
        for extension in ["svg", "pdf", "png"]:
            fig.savefig(output_dir / f"{stem}.{extension}", dpi=300, facecolor="white")
        plt.close(fig)
    (output_dir / "figure4_layout_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
