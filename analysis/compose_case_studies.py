"""Compose Figure 6 and Figure S4 from the editable case figures.

The correction source depicts original unspecified atoms from the supplied
display records. The repeated score comparison in S4a is replaced by its
molecular inventory; main Figure 4 contains that comparison. No numerical
results, case selections or model predictions are changed.
"""

from pathlib import Path
import argparse
from io_utils import prepare_output
from copy import deepcopy
import json
import xml.etree.ElementTree as ET

import cairosvg
from build_figureS1 import molecule as molecule_svg


SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)


def tag(name):
    return f"{{{SVG}}}{name}"


def canvas(width, height, title):
    root = ET.Element(tag("svg"), {
        "width": "7.15in", "height": f"{7.15 * height / width:.8f}in",
        "viewBox": f"0 0 {width} {height}",
    })
    ET.SubElement(root, tag("title")).text = title
    ET.SubElement(root, tag("rect"), {
        "width": str(width), "height": str(height), "fill": "white",
    })
    return root


def correction_panels(root):
    """Separate the three original rows and their common legend."""
    assert root.attrib["viewBox"] == "0 0 1000 875"
    children = list(root)
    starts = [
        next(i for i, e in enumerate(children)
             if e.tag == tag("text") and (e.text or "").startswith(f"{letter}  Reaction EC "))
        for letter in "abc"
    ]
    legend_start = next(
        i for i, e in enumerate(children)
        if e.tag == tag("rect") and e.get("x") == "235" and e.get("y") == "848"
    )
    panels = []
    for start, end in zip(starts, [*starts[1:], legend_start]):
        # Replace row dividers at the composition level, outside the content.
        panels.append([
            deepcopy(e) for e in children[start:end]
            if not (e.tag == tag("path") and e.get("stroke") == "#C5D1D6")
        ])
    return panels, [deepcopy(e) for e in children[legend_start:]]


def append_group(root, elements, transform, group_id):
    group = ET.SubElement(root, tag("g"), {"id": group_id, "transform": transform})
    group.extend(deepcopy(elements))
    return group


def relabel(group, before, after):
    matches = [e for e in group.iter(tag("text")) if e.text == before]
    assert len(matches) == 1, (before, len(matches))
    matches[0].text = after


def write_figure(root, stem):
    data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    stem.with_suffix(".svg").write_bytes(data)
    cairosvg.svg2pdf(bytestring=data, write_to=str(stem.with_suffix(".pdf")))
    cairosvg.svg2png(bytestring=data, write_to=str(stem.with_suffix(".png")),
                    output_width=2400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = prepare_output(args.output_dir, [args.source_dir, args.data_dir])
    correction = ET.parse(args.source_dir / "figure4_rank_correction.svg").getroot()
    examples = ET.parse(args.source_dir / "figure5_bidirectional_examples.svg").getroot()
    assert examples.attrib["viewBox"] == "0 0 1450 588"
    records = json.loads((args.data_dir / "successful_cases.json").read_text())
    display_data = json.loads((args.data_dir / "case_molecular_display.json").read_text())['cases']
    reaction = next(row for row in records['reactions'] if row['is_RE_query'])
    display = display_data[reaction['reaction_id']]
    assert display['source_molecular_inventory'].replace('*', 'C') == reaction['raw_input']
    slots = [e for e in examples if e.tag == tag('svg') and e.get('y') == '133']
    assert len(slots) == len(display['display_molecules']) == 2
    for slot, smiles in zip(slots, display['display_molecules']):
        if '*' in smiles:
            replacement = ET.fromstring(molecule_svg(
                smiles, float(slot.get('x')), float(slot.get('y')),
                float(slot.get('width')), float(slot.get('height'))))
            position = list(examples).index(slot)
            examples.remove(slot)
            examples.insert(position, replacement)
    relabel(examples, "R1 · complete molecular input", "R1 · molecular inventory")
    relabel(examples, "Query R4 · complete molecular input", "Query R4 · molecular inventory")
    panels, legend = correction_panels(correction)
    headings = [next(e.text for e in panel if e.tag == tag("text") and "Reaction EC" in (e.text or ""))
                for panel in panels]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # The original correction row spans the full width. The two reciprocal
    # retrieval examples retain their original side-by-side arrangement.
    merged = canvas(1450, 1065, "Functional matching and bidirectional retrieval")
    upper = ET.SubElement(merged, tag("g"), {
        "id": "panel-a-rank-correction", "transform": "scale(1.45)",
    })
    upper.extend(deepcopy(panels[0]))
    append_group(upper, legend, "translate(0,-560)", "score-legend")
    ET.SubElement(merged, tag("path"), {
        "d": "M12 465 H1438", "stroke": "#C5D1D6", "stroke-width": "1.2",
    })
    lower = append_group(merged, list(examples), "translate(0,477)",
                         "panels-b-c-bidirectional-retrieval")
    relabel(lower, "a  Enzyme → reaction", "b  Enzyme → reaction")
    relabel(lower, "b  Reaction → enzyme", "c  Reaction → enzyme")
    # The original vertical separator extended two units below its viewport.
    # Keep its visible extent within the lower row of the combined canvas.
    for e in lower.iter(tag("path")):
        if e.get("d") == "M775 47 V590":
            e.set("d", "M775 47 V588")
    write_figure(merged, args.output_dir / "figure4_case_studies")

    supplementary = canvas(1000, 525, "Case molecular inventory and ranking correction")
    first = ET.SubElement(supplementary, tag("g"), {"id": "panel-a-molecular-inventory"})
    for y, size, label in [
        (23, 18, "a" + headings[1][1:] + ": molecular inventory"),
        (45, 14, "EC distributions and score contributions: main Figure 4a,b"),
    ]:
        ET.SubElement(first, tag("text"), {
            "x": "8", "y": str(y), "font-family": "DejaVu Sans",
            "font-size": str(size), "fill": "#263A43",
            "font-weight": "bold" if size == 18 else "normal",
        }).text = label
    molecules = [deepcopy(e) for e in panels[1] if e.tag == tag("svg")]
    assert len(molecules) == 2
    for index, molecule in enumerate(molecules):
        x = 20 + index * 490
        molecule.attrib.update(x=str(x), y="54", width="460", height="118")
        first.append(molecule)
        ET.SubElement(first, tag("text"), {
            "x": str(x + 230), "y": "190", "font-family": "DejaVu Sans",
            "font-size": "12", "fill": "#697D86", "text-anchor": "middle",
        }).text = f"M{index + 1}"
    second = append_group(supplementary, panels[2], "translate(0,-350)",
                          "panel-b-rank-correction")
    relabel(second, headings[2], "b" + headings[2][1:])
    ET.SubElement(supplementary, tag("path"), {
        "d": "M8 202 H988", "stroke": "#C5D1D6", "stroke-width": ".9",
    })
    append_group(supplementary, legend, "translate(0,-350)", "score-legend")
    write_figure(supplementary, args.output_dir / "figureS4_rank_corrections")

    manifest = {
        "operation": "Source-faithful molecular depictions; S4a retains inventory only; numerical results unchanged",
        "current_main_figure": 6,
        "filename_retained": "figure4_case_studies",
        "sources": ["figure4_rank_correction.svg", "figure5_bidirectional_examples.svg"],
        "panel_map": {
            "Figure 6a": "original Figure 4a",
            "Figure 6b": "original Figure 5a",
            "Figure 6c": "original Figure 5b",
            "Figure S4a": "molecular inventory of the second correction-source case",
            "Figure S4b": "original Figure 4c",
        },
        "formats": ["svg", "pdf", "png"],
        "editability": "Source SVG text, vector molecules and bars; supplied structure cartoons retained",
        "source_data": ["case_expansion_main.json", "successful_cases.json", "case_molecular_display.json"],
    }
    (args.output_dir / "figure4_composition.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print("Created Figure 6 and compact Figure S4; all retained scores and ranks are unchanged.")


if __name__ == "__main__":
    main()
