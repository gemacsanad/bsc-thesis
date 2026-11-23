"""Generator for enrichment.drawio (thesis Fig., Ch. 4 §4.2: building the joint citation graph).
Usage: python make_enrich_fig.py OUT.drawio"""
import sys, math
from xml.sax.saxutils import escape
out = sys.argv[1]
F = "fontFamily=Latin Modern Roman;fontColor=#000000;"
cells, n = [], [1]
def nid():
    n[0] += 1; return f"c{n[0]}"
esc = lambda t: escape(t, {'"': "&quot;"})
def shape(x, y, w, h, t, fill, stroke="#000000", sw=1, size=9, kind="rounded=0;", extra=""):
    i = nid()
    cells.append(f'<mxCell id="{i}" value="{esc(t)}" style="{kind}whiteSpace=wrap;html=1;{F}fontSize={size};fillColor={fill};strokeColor={stroke};strokeWidth={sw};spacing=0;{extra}" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
def circ(cx, cy, d, fill, **kw):
    shape(cx - d / 2, cy - d / 2, d, d, "", fill, kind="ellipse;", **kw)
def text(x, y, w, h, t, size=9, extra=""):  # x, y = top-left
    i = nid()
    cells.append(f'<mxCell id="{i}" value="{esc(t)}" style="text;html=1;{F}fontSize={size};align=center;verticalAlign=middle;strokeColor=none;fillColor=none;spacing=0;{extra}" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
def edge(pts, arrow=True, color="#000000", extra=""):
    i = nid(); (sx, sy), (tx, ty) = pts[0], pts[-1]
    way = "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in pts[1:-1])
    end = "endArrow=block;endFill=1;endSize=4;" if arrow else "endArrow=none;"
    cells.append(f'<mxCell id="{i}" style="edgeStyle=none;html=1;rounded=0;strokeColor={color};strokeWidth=1;{end}{extra}" edge="1" parent="1"><mxGeometry relative="1" as="geometry"><mxPoint x="{sx}" y="{sy}" as="sourcePoint"/><mxPoint x="{tx}" y="{ty}" as="targetPoint"/>' + (f'<Array as="points">{way}</Array>' if way else "") + '</mxGeometry></mxCell>')
def link(p, q, r=5.5, **kw):               # straight arrow between two node centres, border to border
    dx, dy = q[0] - p[0], q[1] - p[1]; L = math.hypot(dx, dy); ux, uy = dx / L, dy / L
    edge([(p[0] + r * ux, p[1] + r * uy), (q[0] - r * ux, q[1] - r * uy)], **kw)

WHITE, GREY, SKY, ORANGE = "#FFFFFF", "#EEEEEE", "#DDEFFA", "#FAECCC"
CID, MAG = "#D55E00", "#009E73"            # Okabe-Ito vermillion (Corpus ID links), bluish green (MAG id links)
BLUE, FAINT = "#0072B2", "#8C8C8C"         # added citation; paper / citation not added
ADDED = "dashed=1;dashPattern=3 2;"
NOTADDED = "dashed=1;dashPattern=1 2;"
RH = 14                                    # table row height

def table(x, y, label, cols, rows, hh=RH):
    """Ruled table: label above, grey header row (height hh), data rows. cols = [(header, width, header colour)]."""
    W = sum(w for _, w, _ in cols)
    text(x - 10, y - 13, W + 20, 12, label)
    cx = x
    for (hd, w, col), *vals in zip(cols, *rows):
        shape(cx, y, w, hh, hd, GREY, extra=f"fontColor={col};")
        for k, v in enumerate(vals):
            shape(cx, y + hh + RH * k, w, RH, v, WHITE)
        cx += w
    return W

# ---------------- panel (a): identifier matching ----------------
Y = 12                                     # header row top of the upper tables
ya, yb = Y + 1.5 * RH, Y + 2.5 * RH        # centres of the 700 and 900 rows
AX, PX = 0, 108                             # abstracts / paper records left edges
table(AX, Y, "S2 abstracts", [("Corpus ID", 48, CID), ("abstract", 40, "#000000")], [["700", "…"], ["900", "…"]])
table(PX, Y, "S2 paper records", [("Corpus ID", 48, CID), ("MAG", 32, MAG)], [["700", "100"], ["900", "200"]])
PR = PX + 80                               # right edge of the paper records
for y in (ya, yb):                         # abstracts <-> paper records by Corpus ID
    edge([(AX + 88, y), (PX, y)], arrow=False, color=CID)
OGX = PR + 18                              # ogbn-arxiv paper, matched by MAG id
text(OGX - 4, Y - 13, 60, 12, "ogbn-arxiv")
shape(OGX, ya - 8, 52, 16, "MAG 100", SKY)
edge([(PR, ya), (OGX, ya)], arrow=False, color=MAG)
# citation 700 -> 900, translated through the paper records into MAG 100 -> MAG 200
HC = 2 * RH - 4                             # two-line header of the citations table
YC = Y + 3 * RH + 40 - (HC - RH)            # header row top of the citations table (data row kept in place)
yc = YC + HC + 0.5 * RH
table(AX, YC, "S2 citations", [("citing<br>Corpus ID", 48, CID), ("cited<br>Corpus ID", 48, CID)], [["700", "900"]], hh=HC)
xin, xout = PX + 24, PX + 48 + 16          # Corpus ID / MAG column centres
edge([(AX + 96, yc), (xin, yc), (xin, Y + 3 * RH)], color=CID)
RX = xin + 6                               # result box starts just right of the Corpus ID arrow
edge([(xout, Y + 3 * RH), (xout, yc - 8)], color=MAG)
shape(RX, yc - 8, 134, 16, "citation: MAG 100 → MAG 200", WHITE)
AW = max(OGX + 52, RX + 134)                # panel (a) width

# ---------------- panel (b): graph expansion ----------------
BX, BY = AW + 24, 6
D = 12                                     # node diameter
S = 1.3                                    # scale of the graph drawing
P = {"O1": (24, 24), "O2": (55, 12), "O3": (82, 38), "O4": (32, 58), "O5": (64, 68),
     "A1": (6, 44), "A2": (98, 18), "A3": (98, 70), "X": (8, 86)}
P = {k: (BX + S * x, BY + S * y) for k, (x, y) in P.items()}
for a, b in [("O1", "O2"), ("O2", "O3"), ("O1", "O4"), ("O4", "O5"), ("O5", "O3")]:
    link(P[a], P[b], r=D / 2)              # original ogbn-arxiv citations
for a, b in [("O1", "A1"), ("O3", "A2"), ("O5", "A3"), ("A2", "A3"), ("O2", "O5")]:
    link(P[a], P[b], r=D / 2, color=BLUE, extra=ADDED)   # added S2 citations
link(P["A1"], P["X"], r=D / 2, color=FAINT, extra=NOTADDED)
for k, p in P.items():
    if k == "X":
        circ(*p, D, WHITE, stroke=FAINT, extra=NOTADDED)
    else:
        circ(*p, D, SKY if k[0] == "O" else ORANGE)
BW = 104 * S

# ---------------- labels and key ----------------
YL = YC + HC + RH + 10                       # top of the (a)/(b) labels
text(AW / 2 - 15, YL, 30, 14, "(a)", size=11)
text(BX + BW / 2 - 15, YL, 30, 14, "(b)", size=11)
FW = BX + BW                               # full width
def key_row(y, items):
    total = sum(sw + 4 + w for _, sw, w, _ in items) + 14 * (len(items) - 1)
    x = FW / 2 - total / 2
    for draw, sw, w, lab in items:
        draw(x, y, sw)
        text(x + sw + 4, y - 6, w, 12, lab, extra="align=left;")
        x += sw + 4 + w + 14
def kc(fill, stroke="#000000", extra=""):
    return lambda x, y, sw: circ(x + sw / 2, y, 9, fill, stroke=stroke, extra=extra)
def ka(color, extra=""):
    return lambda x, y, sw: edge([(x, y), (x + sw, y)], color=color, extra=extra)
def kx(x, y, sw):                          # not added: dotted arrow into a dotted circle
    edge([(x, y), (x + sw - 9, y)], color=FAINT, extra=NOTADDED)
    circ(x + sw - 4.5, y, 9, WHITE, stroke=FAINT, extra=NOTADDED)
y1 = YL + 26
key_row(y1, [(kc(SKY), 9, 62, "ogbn-arxiv paper"), (kc(ORANGE), 9, 44, "added paper"),
             (kx, 24, 36, "not added")])
key_row(y1 + 15, [(ka("#000000"), 18, 72, "ogbn-arxiv citation"), (ka(BLUE, ADDED), 18, 52, "added citation")])

xml = ('<mxfile><diagram name="enrichment"><mxGraphModel><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
       + "".join(cells) + '</root></mxGraphModel></diagram></mxfile>')
open(out, "w").write(xml)
