"""Draw recorded Top-1 corrections from archived CPU score decompositions."""
from pathlib import Path
import argparse
from io_utils import prepare_output
import html
import json
import xml.etree.ElementTree as ET
import cairosvg
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from molecular_display import molecule_for_display

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-data', type=Path, required=True)
    parser.add_argument('--display-data', type=Path,
                        required=True,
                        help='Source molecular inventories for depiction; scores still use --case-data.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--stem', default='figure4_rank_correction')
    parser.add_argument('--layout', choices=['main', 'supplement'], default='main')
    args = parser.parse_args()
    args.output_dir = prepare_output(args.output_dir, [args.case_data, args.display_data])
    if Path(args.stem).name != args.stem or args.stem in {".", ".."}:
        raise ValueError("--stem must be a filename stem, not a path")
    data = json.loads(args.case_data.read_text())
    display_data = json.loads(args.display_data.read_text())['cases']
    assert 1 <= len(data['cases']) <= 3
    row_height = 280 if args.layout == 'main' else 360
    height = row_height * len(data['cases']) + 35
    INK, TEAL = '#263A43', '#347765'
    COLORS = ['#AAB9C2', '#C4DFD2', '#95C1AC', '#609B85', '#306E5D']
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" viewBox="0 0 1000 {height}">',
             f'<rect width="1000" height="{height}" fill="white"/>']


    def text(x, y, s, size=15, color=INK, bold=False, anchor='start'):
        parts.append(f'<text x="{x}" y="{y}" font-family="DejaVu Sans" font-size="{size}" '
                     f'fill="{color}" font-weight="{"bold" if bold else "normal"}" text-anchor="{anchor}">{html.escape(str(s))}</text>')


    def molecule(smi, index, x, y, w, h):
        mol = molecule_for_display(smi)
        rdDepictor.Compute2DCoords(mol)
        if args.layout == 'supplement':
            x = 8 + (x - 8) * 1.18
            w *= 1.18
        large = mol.GetNumHeavyAtoms() > 35
        # A flexible canvas includes every label on extended cofactors before fitting.
        drawer = rdMolDraw2D.MolDraw2DSVG(-1, -1) if large else rdMolDraw2D.MolDraw2DSVG(int(w * 2), int(h * 2))
        opts = drawer.drawOptions()
        opts.bondLineWidth = 2.4
        opts.minFontSize = 25
        opts.maxFontSize = 32
        opts.padding = .08
        if large:
            opts.fixedBondLength = 28
            opts.fixedFontSize = 20
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        root = ET.fromstring(drawer.GetDrawingText())
        root.set('x', str(x)); root.set('y', str(y))
        root.set('width', str(w)); root.set('height', str(h))
        parts.append(ET.tostring(root, encoding='unicode'))
        text(x + w / 2, y + h + 13, f'M{index + 1}', 12, '#697D86', anchor='middle')


    for case_index, case in enumerate(data['cases']):
        y0 = row_height * case_index
        positive = next(row for row in case['candidates'] if row['full_rank'] == 1)
        previous_top = next(row for row in case['candidates'] if row['latent_rank'] == 1)
        assert positive['recorded_positive'] and positive['full_rank'] == 1 and positive['latent_rank'] > 1
        assert not previous_top['recorded_positive']
        assert set(positive['recorded_ec']) == set(case['recorded_ec'])
        assert not set(previous_top['recorded_ec']).intersection(case['recorded_ec'])
        text(8, y0 + 21, f"{case['case']}  Reaction EC {case['recorded_ec'][0]}", 18, bold=True)
        text(8, y0 + 44, f"Predicted L3: {case['query_predictions'][2]['label']}", 14)
        display = display_data[case['reaction_id']]
        assert display['source_molecular_inventory'].replace('*', 'C') == case['raw_molecular_input']
        mols = display['display_molecules']
        assert len(mols) == len(case['canonical_molecules'])
        available_height = row_height - 85
        if len(mols) == 5:
            # Molecular size controls layout only; each original component is retained.
            sizes = [Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in mols]
            large = sorted(range(5), key=lambda j: (-sizes[j], j))[:2]
            small = [j for j in range(5) if j not in large]
            large_height = available_height * .66
            small_height = available_height - large_height - 24
            for j, x in zip(large, [8, 225]):
                molecule(mols[j], j, x, y0 + 60, 211, large_height)
            for j, x, w in zip(small, [8, 154, 281], [130, 110, 151]):
                molecule(mols[j], j, x, y0 + 60 + large_height + 24, w, small_height)
        elif len(mols) == 2:
            for j, x in enumerate([8, 226]):
                molecule(mols[j], j, x, y0 + 65, 206, min(180, available_height - 20))
        elif len(mols) in [3, 4]:
            cell_height = (available_height - 24) / 2
            for j, smi in enumerate(mols):
                x = 117 if len(mols) == 3 and j == 2 else 8 + (j % 2) * 218
                molecule(smi, j, x, y0 + 60 + (j // 2) * (cell_height + 24), 206, cell_height)
        else:
            raise ValueError('Provide a legible inventory layout for this component count.')
        # Candidate rows and ranking changes all answer the same question.
        candidate_x = 477 if args.layout == 'main' else 540
        bar_x = 619 if args.layout == 'main' else 680
        bar_w = 276 if args.layout == 'main' else 215
        text(candidate_x, y0 + 24, 'Enzyme / ref. EC', 13, bold=True)
        text(bar_x + 19, y0 + 24, 'Score contributions', 13, bold=True)
        text(946, y0 + 24, 'Rank', 13, bold=True, anchor='middle')
        text(946, y0 + 43, 'Latent → Full', 12, anchor='middle')
        score_max = 22
        centers = [y0 + 99, y0 + 190] if args.layout == 'main' else [y0 + 120, y0 + 255]
        axis_bottom = row_height - 48
        for tick in [0, 10, 20]:
            x = bar_x + tick / score_max * bar_w
            parts.append(f'<path d="M{x} {y0 + 59} V{y0 + axis_bottom}" stroke="#E5EAEC" stroke-width="1"/>')
            text(x, y0 + axis_bottom + 17, tick, 12, '#627680', anchor='middle')
        # Show the actual first-ranked candidate before and after adding EC agreement.
        for row, yc, status in zip([previous_top, positive], centers, ['Latent Top-1', 'Full Top-1']):
            mark = ' *' if row['recorded_positive'] else ''
            text(candidate_x, yc - 24, status, 12, TEAL if mark else '#697D86', bool(mark))
            text(candidate_x, yc - 4, row['enzyme_id'] + mark, 15, TEAL if mark else INK, bool(mark))
            text(candidate_x, yc + 16, 'EC: ' + '; '.join(row['recorded_ec']), 12,
                 TEAL if mark else '#627680', bool(mark))
            values = [row['latent_score'], *row['ec_scores']]
            assert abs(sum(values) - row['total_score']) < 1e-4
            assert 0 <= sum(values) <= score_max
            x = bar_x
            for val, color in zip(values, COLORS):
                width = val / score_max * bar_w
                parts.append(f'<rect x="{x:.4f}" y="{yc - 12}" width="{width:.4f}" height="18" fill="{color}" stroke="white" stroke-width=".55"/>')
                x += width
            text(946, yc + 2, f"{row['latent_rank']} → {row['full_rank']}", 15,
                 TEAL if mark else INK, bool(mark), 'middle')
        if case_index + 1 < len(data['cases']):
            parts.append(f'<path d="M8 {y0 + row_height - 7} H988" stroke="#C5D1D6" stroke-width=".9"/>')

    legend_y = row_height * len(data['cases']) + 8
    for x, label, color in zip([235, 414, 486, 558, 630], ['Latent', 'L1', 'L2', 'L3', 'L4'], COLORS):
        parts.append(f'<rect x="{x}" y="{legend_y}" width="18" height="12" fill="{color}"/>')
        text(x + 24, legend_y + 12, label, 14)
    text(763, legend_y + 12, '* Recorded positive', 13, TEAL)
    parts.append('</svg>')
    svg = '\n'.join(parts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / args.stem
    stem.with_suffix('.svg').write_text(svg)
    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(stem.with_suffix('.pdf')))
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(stem.with_suffix('.png')), scale=2)
    print(stem)


if __name__ == "__main__":
    main()
