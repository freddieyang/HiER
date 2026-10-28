"""Render retrieval-depth means from saved HiER/TIGER summary metrics."""
import argparse
from io_utils import prepare_output
import csv
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/hier_retrieval_figure_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = prepare_output(args.output_dir, [args.data_dir])
    with (args.data_dir / 'local_baseline_comparison.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    splits = [('time', 'Time'), ('seq_smi', 'Enzyme similarity'), ('mol_smi', 'Reaction similarity')]
    methods = [('HiER', '#4E8778', 'o', '-', 'HiER'),
               ('TIGER_ESM2Text_local', '#AA96C4', 's', '--', r'TIGER$^i$'),
               ('TIGER_ProtT3_local', '#7199B6', '^', ':', r'TIGER$^{ii}$')]
    ink = '#213547'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
        'axes.titlesize': 8.5, 'axes.labelsize': 8, 'xtick.labelsize': 7,
        'ytick.labelsize': 7, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .55, 'text.color': ink, 'axes.labelcolor': ink,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(3, 2, figsize=(3.36, 3.75))
    values = []
    for i, (split, title) in enumerate(splits):
        for j, direction in enumerate(['E-R', 'R-E']):
            ax = axes[i, j]
            for method, colour, marker, line, _ in methods:
                row = next(r for r in rows if (r['split'], r['direction'], r['method']) == (split, direction, method))
                assert int(row['seed_count']) == 3 and int(row['training_epochs']) == 100
                means = [100 * float(row[f'hit{k}_mean']) for k in [1, 5, 10, 20]]
                ax.plot(range(4), means, color=colour, marker=marker, linestyle=line,
                        markersize=2.8, linewidth=1,
                        markerfacecolor=colour if method == 'HiER' else 'white')
                for k, mean in zip([1, 5, 10, 20], means):
                    values.append(dict(split=split, direction=direction, method=method, k=k,
                        hit_mean_percent=mean, hit_sd_percent=100 * float(row[f'hit{k}_std']),
                        seed_count=3, query_unit=row['query_unit'], query_count=row['query_count'],
                        candidate_count=row['candidate_count'], training_epochs=100,
                        source_file='local_baseline_comparison.csv'))
            ax.set_ylim(0, 104)
            ax.set_xlim(-.12, 3.14)
            ax.set_yticks([0, 50, 100])
            ax.set_xticks(range(4), ['1', '5', '10', '20'])
            if j == 0:
                ax.set_ylabel('Hit@k (%)', labelpad=1.5)
            else:
                ax.tick_params(labelleft=False)
            if i == 2:
                ax.set_xlabel('Retrieval cutoff k', labelpad=2)
            ax.grid(axis='y', alpha=.15)
            ax.tick_params(length=2, pad=1.5)
            ax.set_title(f'{chr(97 + 2*i + j)}  {title}', loc='left', fontweight='bold', fontsize=7.5, pad=5)
            ax.text(.965, .10, f"N={int(row['query_count']):,}\nC={int(row['candidate_count']):,}",
                    transform=ax.transAxes, fontsize=6.5, ha='right')
    handles = [Line2D([], [], color=c, marker=m, linestyle=ls, markersize=3,
                     markerfacecolor=c if key == 'HiER' else 'white', linewidth=1, label=label)
               for key, c, m, ls, label in methods]
    fig.legend(handles=handles, ncol=3, loc='upper center', bbox_to_anchor=(.54, 1.02),
               frameon=False, fontsize=7, handlelength=1.3, columnspacing=1.0, handletextpad=.4)
    fig.text(.32, .921, 'Enzyme → reaction', ha='center', fontsize=7.5)
    fig.text(.78, .921, 'Reaction → enzyme', ha='center', fontsize=7.5)
    fig.subplots_adjust(left=.13, right=.985, bottom=.08, top=.86, hspace=.64, wspace=.25)
    for extension in ['pdf', 'svg', 'png']:
        fig.savefig(args.output_dir / f'figure2_retrieval_depth_column.{extension}',
                    dpi=350, bbox_inches='tight', pad_inches=.04)
    plt.close(fig)
    with (args.output_dir / 'figure2_values.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)
    print(f'Rendered six panels from {len(values)} saved mean values.')


if __name__ == '__main__':
    main()
