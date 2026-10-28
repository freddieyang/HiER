"""Plot the three validation loss-weight scans from supplied per-seed metrics.

No checkpoint is loaded and no training, test evaluation or model selection is
performed. The reported coefficient settings remain fixed display references.
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('MPLCONFIGDIR', '/tmp/hier_loss_sensitivity_mpl')

import argparse
from io_utils import prepare_output
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd

GRIDS = {
    'beta': [0., .1, .2, .5, 1.],
    'lambda_ec': [0., .05, .1, .2, .5, .6, .7, .8, .9],
    'lambda_align': [0., .01, .02, .04, .1, .2, .5, 1.],
}
ADOPTED = {'beta': .2, 'lambda_ec': .1, 'lambda_align': .02}
SPLITS = ['time', 'seq_smi', 'mol_smi']
SPLIT_NAMES = ['Time', 'Enzyme similarity', 'Reaction similarity']
SEEDS = [42, 43, 44]
METRICS = ['Hit1_percent', 'MRR', 'MR']


def load_records(data):
    df = pd.read_csv(data / 'validation_per_seed.csv')
    from io_utils import require_columns
    require_columns(df, ['axis', 'value', 'split', 'seed', 'subset', 'direction',
                         'query_unit', 'N', 'C', 'source', *METRICS], 'validation_per_seed.csv')
    assert set(df.axis) == set(GRIDS) and df.subset.eq('VAL').all()
    assert not df.duplicated(['axis', 'value', 'split', 'seed', 'direction']).any()
    assert np.isfinite(df[['value', 'N', 'C', *METRICS]].to_numpy()).all()
    assert df.Hit1_percent.between(0, 100).all() and df.MRR.between(0, 1).all()
    assert df.MR.ge(1).all() and df.N.ge(1).all() and df.C.ge(1).all()
    assert not df.source.isna().any()
    for axis, grid in GRIDS.items():
        part = df[df.axis.eq(axis)]
        assert len(part) == len(grid)*len(SPLITS)*len(SEEDS)*2
        assert sorted(part.value.unique()) == grid
        assert set(part.split) == set(SPLITS) and set(part.seed) == set(SEEDS)
        for (value, split, seed), group in part.groupby(['value', 'split', 'seed']):
            assert set(group.direction) == {'E-R', 'R-E'}
            assert group.source.nunique() == 1
            e, r = group.set_index('direction').loc[['E-R', 'R-E']].to_dict('records')
            assert (e['N'], e['C']) == (r['C'], r['N'])
            assert e['query_unit'] == 'enzyme' and r['query_unit'] == 'reaction'
    for split in SPLITS:
        part = df[df.split.eq(split)]
        assert part[part.direction.eq('E-R')][['N', 'C']].drop_duplicates().shape[0] == 1
        for seed in SEEDS:
            selected = part[part.seed.eq(seed)]
            reference = selected[selected.axis.eq('beta') & selected.value.eq(ADOPTED['beta'])].set_index('direction')
            for axis in ['lambda_ec', 'lambda_align']:
                current = selected[selected.axis.eq(axis) & selected.value.eq(ADOPTED[axis])].set_index('direction')
                pd.testing.assert_frame_equal(
                    reference[['source', 'N', 'C', *METRICS]].sort_index(),
                    current[['source', 'N', 'C', *METRICS]].sort_index())
    # The common reported configuration is reused across all three axes.
    expected_runs = (sum(map(len, GRIDS.values())) - len(GRIDS) + 1)*len(SPLITS)*len(SEEDS)
    assert df.source.nunique() == expected_runs
    return df


def summarize(df, dest):
    runs = []
    for (axis, value, split, seed), group in df.groupby(['axis', 'value', 'split', 'seed']):
        e, r = group.set_index('direction').loc[['E-R', 'R-E']].to_dict('records')
        runs.append(dict(axis=axis, value=value, split=split, seed=seed,
                         subset='VAL', enzyme_queries=e['N'], reaction_queries=r['N'],
                         bidirectional_hit1=np.sqrt(e['Hit1_percent']*r['Hit1_percent'])/100,
                         bidirectional_mrr=np.sqrt(e['MRR']*r['MRR']), source=e['source']))
    runs = pd.DataFrame(runs)
    score_columns = ['bidirectional_hit1', 'bidirectional_mrr']
    # Average the three partitions within each seed; error bars are sample SD
    # across those three seed averages, not across the nine individual runs.
    per_seed = runs.groupby(['axis', 'value', 'seed'])[score_columns].mean()
    aggregate = per_seed.groupby(['axis', 'value']).mean()
    seed_sd = per_seed.groupby(['axis', 'value']).std(ddof=1)
    for metric in score_columns:
        aggregate[metric+'_sd_ddof1'] = seed_sd[metric]
    means = df.groupby(['axis', 'value', 'split', 'direction'])[METRICS].mean()
    runs.to_csv(dest/'bidirectional_per_run.csv', index=False)
    per_seed.to_csv(dest/'aggregate_validation_per_seed.csv')
    output = aggregate.reset_index()
    output['run_count'] = len(SEEDS)*len(SPLITS)
    output['seed_count'] = len(SEEDS)
    output['split_count'] = len(SPLITS)
    output['subset'] = 'VAL'
    output['reported_setting'] = [row.value == ADOPTED[row.axis] for row in output.itertuples()]
    output.to_csv(dest/'aggregate_validation.csv', index=False)
    return aggregate, means


def render(aggregate, output):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.labelsize': 7.5, 'xtick.labelsize': 7.2, 'ytick.labelsize': 7.2,
                         'axes.linewidth': .65, 'text.color': '#263A43',
                         'axes.labelcolor': '#263A43', 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    # Set the canvas near its final one-column width to preserve legible type.
    fig, axes = plt.subplots(3, 2, figsize=(3.5, 3.5))
    titles = ['a  Retrieval mixture', 'b  EC supervision', 'c  EC distribution alignment']
    labels = [r'$\beta$', r'$\lambda_{\mathrm{EC}}$', r'$\lambda_{\mathrm{align}}$']
    for row, (axis, grid) in enumerate(GRIDS.items()):
        for col, metric, scale, limits, ticks, fmt in [
            (0, 'bidirectional_hit1', 100, (83.6, 85.0), [83.8, 84.2, 84.6, 85.0], '%.1f'),
            (1, 'bidirectional_mrr', 1, (.744, .7533), [.744, .747, .750, .753], '%.3f')]:
            ax = axes[row, col]
            y = aggregate.loc[axis].reindex(grid)[metric].to_numpy()*scale
            sd = aggregate.loc[axis].reindex(grid)[metric+'_sd_ddof1'].to_numpy()*scale
            current = grid.index(ADOPTED[axis])
            ax.axvspan(current-.22, current+.22, color='#E5EFE9', lw=0, zorder=0)
            ax.errorbar(np.arange(len(grid)), y, yerr=sd, color='#395C6D',
                        marker='o', ms=3, lw=1.1, elinewidth=.7, capsize=1.8,
                        capthick=.7, zorder=3)
            ax.scatter([current], [y[current]], marker='s', s=31, color='#347F69',
                       edgecolor='white', lw=.6, zorder=4)
            ax.set_xlim(-.35, len(grid)-.65)
            ax.set_ylim(*limits)
            ax.set_xticks(range(len(grid)), labels=[f'{v:g}' for v in grid])
            if len(grid) > 5:
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
                         rotation_mode='anchor', fontsize=6.5)
            ax.set_yticks(ticks)
            ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))
            ax.tick_params(length=2.5, width=.6, pad=2)
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', lw=.4, color='#E4E8E9', zorder=0)
            ax.set_ylabel('Hit@1 (%)' if col == 0 else 'MRR', labelpad=3)
            ax.set_xlabel(labels[row], labelpad=2)
    fig.subplots_adjust(left=.16, right=.985, bottom=.08, top=.93,
                        wspace=.56, hspace=1.30)
    title_artists = []
    for row, title in enumerate(titles):
        box = axes[row, 0].get_position()
        title_artists.append(fig.text(box.x0, box.y1 + .025, title, fontsize=8,
                                      fontweight='bold', ha='left', va='bottom'))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for row in [1, 2]:
        title_top = title_artists[row].get_window_extent(renderer).y1
        label_bottom = axes[row-1, 0].xaxis.label.get_window_extent(renderer).y0
        assert label_bottom > title_top + 2, 'Panel title overlaps previous x-axis label'
    output.mkdir(parents=True, exist_ok=True)
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(output/f'figure6_loss_sensitivity.{ext}', dpi=400,
                    bbox_inches='tight', pad_inches=.025)
    plt.close(fig)


def latex_table(axis, df, means, dest):
    symbol = r'\lambda_{\mathrm{EC}}' if axis == 'lambda_ec' else r'\lambda_{\mathrm{align}}'
    name = 'EC-supervision weight' if axis == 'lambda_ec' else 'EC-distribution alignment weight'
    fixed = r'$\beta=0.2$ and $\lambda_{\mathrm{align}}=0.02$' if axis == 'lambda_ec' else r'$\beta=0.2$ and $\lambda_{\mathrm{EC}}=0.1$'
    lines = [r'\begin{table}[!htbp]\centering\fontsize{9}{10.5}\selectfont',
             r'\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.08}',
             r'\caption{Validation sensitivity to the '+name+r'.}\label{tab:'+axis+r'_sensitivity}',
             r'\begin{tabular}{@{}lrrrrrr@{}}',
             r'\toprule', r'& \multicolumn{3}{c}{$E\to R$} & \multicolumn{3}{c}{$R\to E$}\\',
             r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
             '$'+symbol+r'$ & Hit@1 $\uparrow$ & MRR $\uparrow$ & MR $\downarrow$ & Hit@1 $\uparrow$ & MRR $\uparrow$ & MR $\downarrow$\\',
             r'\midrule']
    for split, title in zip(SPLITS, SPLIT_NAMES):
        counts = df[df.split.eq(split) & df.direction.eq('E-R')].iloc[0]
        n_e, n_r = int(counts.N), int(counts.C)
        lines.append(r'\multicolumn{7}{@{}l}{\textbf{'+title+r' validation} ($N_E='+
                     f'{n_e:,}'+r'$, $N_R='+f'{n_r:,}'+r'$)}\\')
        for value in GRIDS[axis]:
            cells = [f'{value:g}']
            for direction in ['E-R', 'R-E']:
                for metric, precision in [('Hit1_percent', 2), ('MRR', 4), ('MR', 2)]:
                    numbers = [round(means.loc[(axis, v, split, direction), metric], precision) for v in GRIDS[axis]]
                    order = sorted(set(numbers), reverse=metric != 'MR')
                    number = round(means.loc[(axis, value, split, direction), metric], precision)
                    cell = f'{number:.{precision}f}'
                    if number == order[0]:
                        cell = r'\mathbf{'+cell+'}'
                    elif len(order) > 1 and number == order[1]:
                        cell = r'\underline{'+cell+'}'
                    cells.append('$'+cell+'$')
            lines.append(' & '.join(cells)+r' \\')
        if split != SPLITS[-1]:
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}',
              r'\par\smallskip\begin{minipage}{\linewidth}\footnotesize Three-seed means (42--44); Hit@1 is in percent. '+
              fixed+r' remain fixed. Bold and underline mark the best and second-best distinct values at the displayed precision. '+
              r'Zero disables only the indicated loss; the EC predictor and retrieval score remain present. '+
              r'MRR averages reciprocal ranks over all recorded positives within each query, then over queries.\end{minipage}',
              r'\end{table}']
    (dest/f'table_{axis}.tex').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True,
                        help='Directory containing validation_per_seed.csv')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    dest = prepare_output(args.output_dir, [args.data_dir])
    df = load_records(args.data_dir)
    aggregate, means = summarize(df, dest)
    render(aggregate, dest)
    for axis in ['lambda_ec', 'lambda_align']:
        latex_table(axis, df, means, dest)
    environment = dict(created_at=datetime.now(timezone.utc).isoformat(),
                       python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                       matplotlib=matplotlib.__version__, device='CPU', training=False,
                       model_forward=False, test_evaluation=False, new_model_selection=False,
                       unique_source_runs=int(df.source.nunique()), directional_rows=len(df),
                       grids=GRIDS, seeds=SEEDS, reported_settings=ADOPTED,
                       error_bars='sample SD across three seed means, each averaged over three partitions',
                       source_run_artifacts_verified=False)
    (dest/'figure_environment.json').write_text(json.dumps(environment, indent=2)+'\n')
    print(f'Rendered three-parameter Figure 5 from {len(df)} supplied directional rows; '
          'source run artifacts were not opened.')


if __name__ == '__main__':
    main()
