"""Render Figure 4, its standalone panels and Table S11 from scientific data."""
from __future__ import annotations

import argparse
from io_utils import prepare_output
from pathlib import Path

import numpy as np
import pandas as pd
from io_utils import require_columns

from hierarchy_figure_layout import render_hierarchy_panels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = prepare_output(args.output_dir, [args.data_dir])
    trees = pd.read_csv(args.data_dir / "case_tree_nodes.csv")
    scores = pd.read_csv(args.data_dir / "case_score_components.csv")
    metrics = pd.read_csv(args.data_dir / "metrics_per_seed.csv")
    summary = pd.read_csv(args.data_dir / "metrics_summary.csv")
    require_columns(trees, ["entity_id", "level", "node", "parent", "probability", "other"], "case_tree_nodes.csv")
    require_columns(scores, ["candidate_id", "recorded_positive", "latent_rank", "full_rank",
                            "archived_full_score", "latent", "L1", "L2", "L3", "L4"], "case_score_components.csv")
    require_columns(metrics, ["split", "seed", "direction", "condition", "delta_hit1", "delta_mrr"], "metrics_per_seed.csv")
    assert trees.other.isin([True, False]).all() and scores.recorded_positive.isin([True, False]).all()
    trees["other"] = trees.other.astype(bool)
    scores["recorded_positive"] = scores.recorded_positive.astype(bool)
    assert not trees.duplicated(["entity_id", "node"]).any()
    assert not scores.candidate_id.duplicated().any()
    assert trees.probability.between(0, 1).all()
    np.testing.assert_allclose(scores[["latent", "L1", "L2", "L3", "L4"]].sum(axis=1),
                               scores.archived_full_score, rtol=0, atol=1e-4)
    keys = ["split", "direction", "condition"]
    assert not metrics.duplicated(keys + ["seed"]).any()
    assert not summary.duplicated(keys).any()
    # Summaries must describe the same paired differences that the figure plots.
    for split in ["time", "seq_smi", "mol_smi"]:
        for direction in ["E-R", "R-E"]:
            for level in range(1, 5):
                condition = f"without_L{level}"
                rows = metrics[(metrics.split == split) & (metrics.direction == direction)
                               & (metrics.condition == condition)]
                assert sorted(rows.seed.tolist()) == [42, 43, 44]
                stored = summary[(summary.split == split) & (summary.direction == direction)
                                 & (summary.condition == condition)]
                assert len(stored) == 1
                for metric in ["delta_hit1", "delta_mrr"]:
                    np.testing.assert_allclose([rows[metric].mean(), rows[metric].std(ddof=1)],
                        [stored.iloc[0][metric+"_mean"], stored.iloc[0][metric+"_sd"]], rtol=0, atol=1e-10)
    out = args.output_dir
    render_hierarchy_panels(trees, scores, metrics, out)
    table = [r"\begin{table}[!htbp]\centering\fontsize{9}{10.5}\selectfont",
             r"\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.05}",
             r"\caption{Paired changes after removing one EC score level from a fixed model.}\label{tab:hierarchy_scores}",
             r"\begin{tabular}{@{}lrrrr@{}}",
             r"\toprule",
             r"& \multicolumn{2}{c}{$E\to R$} & \multicolumn{2}{c}{$R\to E$}\\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Removed level & $\Delta$Hit@1 (pp) & $\Delta$MRR & $\Delta$Hit@1 (pp) & $\Delta$MRR\\",
             r"\midrule"]
    for split, title in [("time", "Time"), ("seq_smi", "Enzyme similarity"), ("mol_smi", "Reaction similarity")]:
        table.append(r"\multicolumn{5}{@{}l}{\textbf{" + title + r"}}\\")
        for level in range(1, 5):
            cells = [f"L{level}"]
            for direction in ["E-R", "R-E"]:
                record = summary[(summary.split == split) & (summary.direction == direction) &
                                 (summary.condition == f"without_L{level}")].iloc[0]
                for metric, scale, digits in [("delta_hit1", 100, 2), ("delta_mrr", 1, 4)]:
                    mean, sd = record[f"{metric}_mean"]*scale, record[f"{metric}_sd"]*scale
                    cells.append(f"${mean:+.{digits}f}\\pm{sd:.{digits}f}$")
            table.append(" & ".join(cells) + r" \\")
        table.append(r"\midrule" if split != "mol_smi" else r"\bottomrule")
    table.extend([r"\end{tabular}",
                  r"\par\smallskip\begin{minipage}{\linewidth}\footnotesize Values are means $\pm$ sample SD of the three paired seed differences (42--44). Each difference is the full score's metric minus the metric after removing the indicated level. Positive values favour the full score. Hit@1 changes are in percentage points; MRR averages reciprocal ranks over all recorded positives within each query, then over queries. The original test libraries and stable tie order are retained. Remaining score coefficients are fixed, without renormalization or retraining.\end{minipage}",
                  r"\end{table}"])
    (args.output_dir / "table_hierarchy_scores.tex").write_text("\n".join(table) + "\n")
    print(out / "figure4_hierarchical_matching.pdf")


if __name__ == "__main__":
    main()
