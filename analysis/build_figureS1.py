"""Render the complete R2/R3 molecular inventories from saved case records."""
import argparse
from io_utils import prepare_output
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cairosvg
from rdkit.Chem.Draw import rdMolDraw2D
from molecular_display import molecule_for_display



def text(x, y, value, size=20, bold=False):
    return (f'<text x="{x}" y="{y}" font-family="DejaVu Sans" '
            f'font-size="{size}" font-weight="{700 if bold else 400}" '
            f'fill="#213547" text-anchor="start">'
            f'{html.escape(str(value))}</text>')


def molecule(smiles, x, y, width, height):
    mol = molecule_for_display(smiles)
    drawer = rdMolDraw2D.MolDraw2DSVG(int(width), int(height))
    options = drawer.drawOptions()
    options.bondLineWidth = 1.65
    options.fixedBondLength = 22
    options.fixedFontSize = 17
    options.padding = .045
    options.clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    element = ET.fromstring(drawer.GetDrawingText())
    element.attrib.update(x=str(x), y=str(y), width=str(width), height=str(height))
    return ET.tostring(element, encoding='unicode')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-data', type=Path, required=True)
    parser.add_argument('--display-data', type=Path,
                        required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = prepare_output(args.output_dir, [args.case_data, args.display_data])
    data = json.loads(args.case_data.read_text())
    display_data = json.loads(args.display_data.read_text())['cases']
    parts = []
    for label, record, start, width in [
        ('a  Reaction R2', data['reactions'][1], 12, 900),
        ('b  Reaction R3', data['reactions'][2], 942, 496),
    ]:
        parts += [text(start, 30, label, 26, True),
                  text(start, 66, 'Annotated EC: ' + ', '.join(record['annotated_EC_union']), 23)]
        display = display_data[record['reaction_id']]
        assert display['source_molecular_inventory'].replace('*', 'C') == record['raw_input']
        components = display['display_molecules']
        if len(components) != 2:
            raise ValueError('This registered layout requires two complete components per reaction.')
        for index, smiles in enumerate(components):
            parts.append(molecule(smiles, start + index * width / 2, 95, width / 2 - 10, 220))
    parts.append('<path d="M 924,14 V 329" stroke="#D7DFE3" stroke-width="1.2"/>')
    width, height = 1450, 345
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="7.15in" '
           f'height="{7.15 * height / width}in" viewBox="0 0 {width} {height}">'
           f'<rect width="{width}" height="{height}" fill="white"/>' + ''.join(parts) + '</svg>')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / 'figureS1_case_inventories'
    stem.with_suffix('.svg').write_text(svg)
    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(stem.with_suffix('.pdf')))
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(stem.with_suffix('.png')), output_width=2700)


if __name__ == '__main__':
    main()
