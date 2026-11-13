"""Generator for gnn_estimator.drawio (thesis Fig. 5.2, Ch. 5 §5.1). Usage: python make_gnn_estimator.py OUT.drawio

Left to right: the same neighbourhood drawn once per attention head (edge width = that head's
attention weight; the [MASK] target has a self-loop) -> four 256-value outputs stacked into one
1024-value vector (concatenation) -> linear layer -> L2 normalisation -> GNN estimate; dashed
training branch: 1 - cos(estimate, target's own embedding).
"""
import sys
import math
sys.path.insert(0, "/Users/gemacsanad/egyetem/DyTAG/baseline/.claude/skills/thesis-figure/scripts")
from drawio_kit import *

TW = 1.5                                    # outline of trainable elements
C = 13                                      # vector cell size
PITCH = 44                                  # vertical distance between the head rows
Y0 = 42                                     # centre of the first head row
ROWS = [Y0 + PITCH * k for k in range(4)]
MY = (ROWS[0] + ROWS[-1]) / 2               # main axis of the right half

# reference positions relative to the target, and each head's attention widths (r1, r2, r3, self)
M = (54, 0)
REFS = [(M[0] - 28, -14), (M[0] - 36, 3), (M[0] - 20, 16)]   # arrive at distinct points on the left half
WIDTHS = [(2.6, 0.7, 1.0, 0.7),
          (0.7, 2.6, 1.0, 0.7),
          (1.0, 0.7, 2.6, 0.7),
          (1.6, 0.7, 0.7, 2.0)]
RM, RR = 7, 5                               # radius of the target, of a reference


def head_size(w):
    """draw.io scales arrowheads with the stroke width; keep them about the size of a 1 px arrow's."""
    return f"endSize={max(1.5, 3.5 / w):.1f};"

def head(cy, widths):
    mx, my = M[0], cy + M[1]
    for (rx, ry), w in zip(REFS, widths):
        px, py = rx, cy + ry
        dx, dy = mx - px, my - py
        L = math.hypot(dx, dy)
        edge([(px + RR * dx / L, py + RR * dy / L), (mx - (RM + 0.5) * dx / L, my - (RM + 0.5) * dy / L)],
             width=w, extra=head_size(w))
    # self-loop on the target, from its upper left to its upper right
    ws = widths[3]
    a0, a1 = math.radians(130), math.radians(50)
    sx, sy = mx + RM * math.cos(a0), my - RM * math.sin(a0)
    ex, ey = mx + (RM + 0.5) * math.cos(a1), my - (RM + 0.5) * math.sin(a1)
    edge([(sx, sy), (sx - 3, my - RM - 10), (ex + 3, my - RM - 10), (ex, ey)], width=ws,
         extra="curved=1;" + f"endSize={max(1.5, 2.5 / ws):.1f};")
    for rx, ry in REFS:
        circ(rx, cy + ry, 2 * RR, WHITE)
    circ(mx, my, 2 * RM, WHITE, sw=TW, extra=DASHED)    # the target; its row holds the learned [MASK] vector

# the GATv2 layer (trainable): one box behind the four head rows, labelled above
GX0, GY0 = 8, ROWS[0] - PITCH / 2
shape(GX0, GY0, M[0] + RM + 8 - GX0, ROWS[-1] - ROWS[0] + PITCH, "", TRAIN, sw=TW)
text(GX0, GY0 - 14, M[0] + RM + 8 - GX0, 12, "GATv2")
for cy, w in zip(ROWS, WIDTHS):
    head(cy, w)

# concatenation: the four 256-value head outputs form one 1024-value vector
BX, BW = 84, 20
for k, cy in enumerate(ROWS):
    edge([(M[0] + RM + 1, cy), (BX, cy)])
    shape(BX, cy - PITCH / 2, BW, PITCH, "256", WHITE, size=8)
text(BX - 15, ROWS[-1] + PITCH / 2 + 2, BW + 30, 12, "1024", size=8)

# linear layer -> normalisation -> estimate
X = BX + BW
edge([(X, MY), (X + 12, MY)])
shape(X + 12, MY - 9, 38, 18, "linear", TRAIN, sw=TW)
X += 50
edge([(X, MY), (X + 12, MY)])
shape(X + 12, MY - 9, 50, 18, "normalise", WHITE)
X += 62
edge([(X, MY), (X + 12, MY)])
X += 12
shape(X, MY - C / 2, C, C, "", WHITE)                       # the GNN estimate
text(X + C / 2 - 25, MY + C / 2 + 2, 50, 12, "estimate", size=8)

# training only: 1 - cos(estimate, own embedding)
LX = X + C + 22
edge([(X + C, MY), (LX, MY)], extra=DASHED)
shape(LX, MY - 9, 40, 18, "1 − cos", WHITE)
OX = LX + 40 + 22
edge([(OX, MY), (LX + 40, MY)], extra=DASHED)
shape(OX, MY - C / 2, C, C, "", WHITE)                      # the target's own embedding
text(OX + C / 2 - 30, MY + C / 2 + 2, 60, 22, "own<br>embedding", size=8)

# about 4 px of margin around the drawing (the PDF export crops to the content)
shape(-4, ROWS[0] - PITCH / 2 - 17, OX + C + 25, ROWS[-1] - ROWS[0] + PITCH + 37, "", "none", stroke="none")
write(sys.argv[1], name="gnn_estimator")
