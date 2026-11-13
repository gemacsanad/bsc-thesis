"""Generator for prefix_adapter.drawio (thesis Fig. 5.3, Ch. 5 §5.2). Usage: python make_prefix_adapter.py OUT.drawio

Left to right: GNN estimate -> trunk -> one projection -> 80 codes (sliced; K and V rows of
layers 40 and 1 drawn) -> key / value decoder (shared across layers) -> B + g * Delta -> the
key or value prefix of that layer (8 x 32 x 128).
"""
import sys
sys.path.insert(0, "/Users/gemacsanad/egyetem/DyTAG/baseline/.claude/skills/thesis-figure/scripts")
from drawio_kit import *

TW = 1.5
C = 13
P = 17                                       # row pitch
# rows top to bottom: layer 40 K, V, (dots), layer 1 K, V
ROWS = [("K", 40), ("V", 40), None, ("K", 1), ("V", 1)]
Y0 = 14
YS = [Y0 + P * i for i in range(len(ROWS))]
MY = (YS[0] + YS[-1]) / 2

# estimate -> trunk -> projection
shape(0, MY - C / 2, C, C, "", WHITE)
text(-12, MY + C / 2 + 2, 37, 12, "estimate", size=8)
edge([(C, MY), (C + 9, MY)])
X = C + 9
shape(X, MY - 9, 34, 18, "trunk", TRAIN, sw=TW)
X += 34
edge([(X, MY), (X + 9, MY)])
X += 9
shape(X, MY - 9, 52, 18, "projection", TRAIN, sw=TW)
X += 52

# the sliced projection output: one column of codes
CX = X + 12
edge([(X, MY), (CX, MY)])
for (row, y) in zip(ROWS, YS):
    if row is None:                          # the codes of layers 2-39, so the column stays one piece
        shape(CX, y - P / 2, 16, P, r"\(\vdots\)", WHITE, size=9)
        continue
    shape(CX, y - P / 2, 16, P, row[0], WHITE)
text(CX - 22, YS[-1] + P / 2 + 2, 60, 12, "80 codes", size=8)

# decoders (shared), gate and base, output prefix per row
DX = CX + 16 + 12
DW, GW, OW = 68, 46, 60                     # decoder, gate/base, output box widths
MATH_MID = "spacingBottom=1;"              # MathJax labels sit low; lift them to the centre
for (row, y) in zip(ROWS, YS):
    if row is None:
        for x, w in ((DX, DW), (DX + DW + 12, GW), (DX + DW + GW + 24, OW)):
            text(x, y - P / 2, w, P, r"\(\vdots\)", size=9)
        continue
    role, layer = row
    edge([(CX + 16, y), (DX, y)])
    shape(DX, y - 6.5, DW, 13, "key decoder" if role == "K" else "value decoder", TRAIN, sw=TW)
    x = DX + DW
    edge([(x, y), (x + 12, y)])
    shape(x + 12, y - 6.5, GW, 13, r"\(\mathbf{B} + g\,\boldsymbol{\Delta}\)", TRAIN, sw=TW, extra=MATH_MID)
    x += 12 + GW
    edge([(x, y), (x + 12, y)])
    shape(x + 12, y - 6.5, OW, 13, r"\(8 \times 32 \times 128\)", WHITE, extra=MATH_MID)

# layer labels to the right of each K/V pair
RX = DX + DW + GW + OW + 24 + 6
for (a, b) in ((0, 1), (3, 4)):
    y0, y1 = YS[a] - 6.5, YS[b] + 6.5
    edge([(RX, y0), (RX + 3, y0), (RX + 3, y1), (RX, y1)], arrow=False)
    text(RX + 6, (y0 + y1) / 2 - 6, 40, 12, f"layer {ROWS[a][1]}", size=8, extra="align=left;")

write(sys.argv[1], name="prefix_adapter", math=True)
