"""Generator for g_retriever.drawio, simple version v2 (thesis Fig., Ch. 2 §2.6). Usage: python make_gr_fig.py OUT.drawio"""
import sys, math
from xml.sax.saxutils import escape
out = sys.argv[1]
F = "fontFamily=Latin Modern Roman;fontColor=#000000;"
cells, n = [], [1]
def nid():
    n[0] += 1; return f"c{n[0]}"
esc = lambda t: escape(t, {'"': "&quot;"})
def shape(x, y, w, h, t, fill, stroke="#000000", sw=1, size=10, kind="rounded=0;", extra=""):
    i = nid()
    cells.append(f'<mxCell id="{i}" value="{esc(t)}" style="{kind}whiteSpace=wrap;html=1;{F}fontSize={size};fillColor={fill};strokeColor={stroke};strokeWidth={sw};{extra}" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
def box(cx, cy, w, h, t, fill, **kw):
    shape(cx - w / 2, cy - h / 2, w, h, t, fill, **kw)
def circ(cx, cy, d, fill, **kw):
    shape(cx - d / 2, cy - d / 2, d, d, "", fill, kind="ellipse;", **kw)
def edge(pts, arrow=True, color="#000000"):
    i = nid(); (sx, sy), (tx, ty) = pts[0], pts[-1]
    way = "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in pts[1:-1])
    end = "endArrow=block;endFill=1;endSize=4;" if arrow else "endArrow=none;"
    cells.append(f'<mxCell id="{i}" style="edgeStyle=none;html=1;rounded=0;strokeColor={color};strokeWidth=1;{end}" edge="1" parent="1"><mxGeometry relative="1" as="geometry"><mxPoint x="{sx}" y="{sy}" as="sourcePoint"/><mxPoint x="{tx}" y="{ty}" as="targetPoint"/>' + (f'<Array as="points">{way}</Array>' if way else "") + '</mxGeometry></mxCell>')
def link(p, q, r=5):                       # plain line between two node centres, border to border
    dx, dy = q[0] - p[0], q[1] - p[1]; L = math.hypot(dx, dy); ux, uy = dx / L, dy / L
    edge([(p[0] + r * ux, p[1] + r * uy), (q[0] - r * ux, q[1] - r * uy)], arrow=False)

WHITE, GREY, VERM, SKY, YELLOW = "#FFFFFF", "#EEEEEE", "#F6D5C2", "#DDEFFA", "#FCF9D9"
GSTROKE = "#7F7F7F"
TW = 1.5                                   # outline width of trainable modules / highlighted nodes
D = 11                                     # node diameter

# ---------------- graph (local coordinates) ----------------
NODES = {1: (8, 8), 2: (30, 4), 3: (54, 10), 4: (80, 6),
         5: (12, 32), 6: (34, 28), "P": (58, 32),
         7: (4, 56), 8: (28, 52), "C": (60, 54), "Q": (82, 48),
         9: (8, 82), 10: (30, 80), "R": (48, 74), 11: (84, 88)}
EDGES = [(1, 2), (2, 3), (3, 4), (1, 5), (2, 6), (5, 6), (6, "P"), (3, "P"), (4, "Q"), (5, 7), (7, 8),
         (6, 8), (8, "C"), ("P", "C"), ("C", "Q"), ("C", "R"), ("Q", 11), (8, 10), (9, 10), (7, 9),
         (10, "R"), ("R", 11)]
SEL, SUB = ("P", "Q", "R"), ("P", "C", "Q", "R")
EC, ERX, ERY = (65, 53), 30, 34            # yellow region around the subgraph (local)
SX_, SY = 0.95, 1.25                        # horizontal / vertical stretch of the graph drawing
NODES = {k: (x * SX_, y * SY) for k, (x, y) in NODES.items()}
EC, ERX, ERY = (EC[0] * SX_, EC[1] * SY), ERX * SX_, ERY * SY

def draw_graph(ox, oy, keep, rot=0, rx=None, dx=0):
    g = lambda k: (ox + NODES[k][0], oy + NODES[k][1])
    rx = rx or ERX
    shape(ox + EC[0] + dx - rx, oy + EC[1] - ERY, 2 * rx, 2 * ERY, "", YELLOW, stroke="none", kind="ellipse;",
          extra=f"rotation={rot};")
    for a, b in EDGES:
        if a in keep and b in keep:
            link(g(a), g(b))
    for k in NODES:
        if k in keep:
            sel = k in SEL
            circ(*g(k), D, SKY if sel else WHITE, sw=TW if sel else 1)

# ---------------- layout ----------------
QB = (0, 0, 68, 44)                       # question box x, y, w, h
GX, GY = 87, 22                            # big graph origin
FR = (GX - 8, GY - 8, GX + 90, GY + 88 * SY + 8)   # dashed frame around the big graph
MY = (FR[1] + FR[3]) / 2                   # main axis (question, graph, subgraph)
SX = FR[2] + 15 + ERX                      # subgraph: x of its region centre
SOX, SOY = SX - EC[0], MY - EC[1]          # subgraph origin (same relative positions)
GL = SX + 10                               # left edge of the GNN box
XC, C, P = GL + 114 + 6.5, 13, 16          # input column: x centre, cell size, pitch
Y0 = 24                                    # soft-prompt cell (first in the sequence)
NDOC, NQ = 6, 3                            # token cells from the subgraph text and from the question
ys = [Y0 + P * i for i in range(1 + NDOC + NQ)]
LLM = (XC + 6.5 + 8, XC + 6.5 + 34)                           # LLM block x range

# question -> big graph
QB = (QB[0], MY - QB[3] / 2, QB[2], QB[3])
shape(*QB, "Where was Pierre Curie’s wife born?", WHITE)
edge([(QB[0] + QB[2], MY), (FR[0], MY)])
shape(FR[0], FR[1], FR[2] - FR[0], FR[3] - FR[1], "", "none", stroke=GSTROKE, extra="dashed=1;dashPattern=3 3;")
draw_graph(GX, GY, list(NODES), rot=18, rx=ERX - 3, dx=0)
# big graph -> subgraph
edge([(FR[2], MY), (SX - ERX, MY)])
draw_graph(SOX, SOY, SUB)

# graph path: subgraph -> GNN -> pooled vector -> projection -> soft prompt
edge([(SX, MY - ERY), (SX, Y0), (GL, Y0)])
box(GL + 14, Y0, 28, 16, "GNN", VERM, sw=TW)
edge([(GL + 28, Y0), (GL + 35, Y0)])
box(GL + 35 + C / 2, Y0, C, C, "", WHITE)  # the pooled graph vector
edge([(GL + 48, Y0), (GL + 55, Y0)])
box(GL + 81, Y0, 52, 16, "projection", VERM, sw=TW)
edge([(GL + 107, Y0), (XC - C / 2, Y0)])
box(XC, Y0, C, C, "", VERM, sw=TW)

# text path: subgraph as text -> token cells; question -> token cells
def bracket(y0, y1, x=XC - C / 2 - 6):
    edge([(x + 3, y0), (x, y0), (x, y1), (x + 3, y1)], arrow=False)
    return x, (y0 + y1) / 2
ydoc = (ys[1] + ys[NDOC]) / 2
xb, _ = bracket(ys[1] - 5, ys[NDOC] + 5)
DOC = (GL + 36, ydoc - 14, 22, 28)
xe = SX + ERX * math.sqrt(1 - ((ydoc - MY) / ERY) ** 2)
edge([(xe, ydoc), (DOC[0], ydoc)])
shape(*DOC, "", WHITE, kind="shape=note;size=6;")
shape(DOC[0] + DOC[2] / 2 - 32, DOC[1] + DOC[3] + 2, 64, 22, "nodes and edges<br>as text", "none",
      stroke="none", size=9, extra="spacing=0;")
for k in range(4):
    yl = DOC[1] + 9 + 5 * k
    edge([(DOC[0] + 4, yl), (DOC[0] + DOC[2] - (4 if k < 3 else 9), yl)], arrow=False, color=GSTROKE)
edge([(DOC[0] + DOC[2], ydoc), (xb, ydoc)])
yq = (ys[NDOC + 1] + ys[-1]) / 2
xb, _ = bracket(ys[NDOC + 1] - 5, ys[-1] + 5)
edge([(QB[2] / 2, QB[1] + QB[3]), (QB[2] / 2, yq), (xb, yq)])

# input column -> LLM -> answer
for i, y in enumerate(ys):
    if i:
        box(XC, y, C, C, "", WHITE)
    edge([(XC + C / 2, y), (LLM[0], y)])
shape(LLM[0], ys[0] - 8, LLM[1] - LLM[0], ys[-1] - ys[0] + 16, "LLM", GREY)
ym = (ys[0] + ys[-1]) / 2
edge([(LLM[1], ym), (LLM[1] + 9, ym)])
shape(LLM[1] + 9, ym - 10, 38, 20, "Warsaw", WHITE)

xml = ('<mxfile><diagram name="g_retriever"><mxGraphModel><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
       + "".join(cells) + '</root></mxGraphModel></diagram></mxfile>')
open(out, "w").write(xml)
