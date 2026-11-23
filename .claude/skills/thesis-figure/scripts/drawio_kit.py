"""Helpers for writing a thesis diagram as a .drawio file from Python.

Taken from the generators of the approved figures (see ../examples/). A generator imports this,
places shapes in draw.io px (1 px ~ 0.256 mm in the exported PDF; 440 px ~ 113 mm), and calls
write(). Coordinates are absolute; x, y of a shape or text box is its top-left corner.

    import sys; sys.path.insert(0, "<skill>/scripts")
    from drawio_kit import *
    shape(0, 0, 60, 16, "GNN", TRAIN, sw=1.5)
    edge([(60, 8), (90, 8)])
    write("out.drawio", name="my_figure")
"""
import math
from xml.sax.saxutils import escape

FONT = "fontFamily=Latin Modern Roman;fontColor=#000000;"

# Okabe-Ito tints used as fills (light), and the full colours for lines and coloured text.
WHITE, GREY = "#FFFFFF", "#EEEEEE"          # neutral box / frozen module
SKY = "#DDEFFA"                             # ogbn-arxiv paper, retrieved node, 1-hop region
ORANGE = "#FAECCC"                          # added paper
TRAIN = "#F6D5C2"                           # trainable module (vermillion tint), 1.5 px outline
YELLOW = "#FCF9D9"                          # selected region / 2-hop region
BLUE, VERMILLION, GREEN, FAINT = "#0072B2", "#D55E00", "#009E73", "#8C8C8C"
DASHED = "dashed=1;dashPattern=3 2;"
DOTTED = "dashed=1;dashPattern=1 2;"

_cells, _n = [], [1]


def _id():
    _n[0] += 1
    return f"c{_n[0]}"


def _esc(t):
    return escape(t, {'"': "&quot;"})


def shape(x, y, w, h, t, fill, stroke="#000000", sw=1, size=9, kind="rounded=0;", extra=""):
    """Box (or any draw.io shape via `kind`, e.g. "ellipse;"). Label centred; <br> breaks lines."""
    _cells.append(f'<mxCell id="{_id()}" value="{_esc(t)}" style="{kind}whiteSpace=wrap;html=1;{FONT}'
                  f'fontSize={size};fillColor={fill};strokeColor={stroke};strokeWidth={sw};spacing=0;{extra}" '
                  f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')


def circ(cx, cy, d, fill, **kw):
    """Node of diameter d centred at (cx, cy)."""
    shape(cx - d / 2, cy - d / 2, d, d, "", fill, kind="ellipse;", **kw)


def text(x, y, w, h, t, size=9, extra=""):
    """Free text in a w x h box, centred unless extra="align=left;". Math: r"\\(x_v\\)" with math=True."""
    _cells.append(f'<mxCell id="{_id()}" value="{_esc(t)}" style="text;html=1;{FONT}fontSize={size};'
                  f'align=center;verticalAlign=middle;strokeColor=none;fillColor=none;spacing=0;{extra}" '
                  f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')


def edge(pts, arrow=True, color="#000000", width=1, extra=""):
    """Polyline through pts [(x, y), ...]; block arrowhead at the last point."""
    (sx, sy), (tx, ty) = pts[0], pts[-1]
    way = "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in pts[1:-1])
    end = "endArrow=block;endFill=1;endSize=4;" if arrow else "endArrow=none;"
    _cells.append(f'<mxCell id="{_id()}" style="edgeStyle=none;html=1;rounded=0;strokeColor={color};'
                  f'strokeWidth={width};{end}{extra}" edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
                  f'<mxPoint x="{sx}" y="{sy}" as="sourcePoint"/><mxPoint x="{tx}" y="{ty}" as="targetPoint"/>'
                  + (f'<Array as="points">{way}</Array>' if way else "") + '</mxGeometry></mxCell>')


def link(p, q, r=6, **kw):
    """Straight arrow between two node centres, from border to border (nodes of radius r)."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    edge([(p[0] + r * ux, p[1] + r * uy), (q[0] - r * ux, q[1] - r * uy)], **kw)


def panel_label(cx, y, letter):
    """"(a)", "(b)" centred under a panel, 11 px."""
    text(cx - 15, y, 30, 14, f"({letter})", size=11)


def write(path, name="figure", math=False):
    """Write the .drawio file. math=True enables MathJax (\\( ... \\) labels), exported as vector paths."""
    model = '<mxGraphModel math="1">' if math else "<mxGraphModel>"
    xml = (f'<mxfile><diagram name="{name}">{model}<root><mxCell id="0"/><mxCell id="1" parent="0"/>'
           + "".join(_cells) + "</root></mxGraphModel></diagram></mxfile>")
    with open(path, "w") as f:
        f.write(xml)
