"""Generator for method_overview.drawio (thesis Fig. 5.1, Ch. 5 intro). Usage: python make_method_overview.py OUT.drawio

Left to right: the target's references (1-hop region) -> GNN -> one vector -> prefix adapter
-> key/value prefixes in every layer of the LLM (layers carry LoRA) -> generated abstract.
"""
import sys
sys.path.insert(0, "/Users/gemacsanad/egyetem/DyTAG/baseline/.claude/skills/thesis-figure/scripts")
from drawio_kit import *

TW = 1.5                                    # outline of trainable modules
C = 13                                      # token / vector cell size
MY = 63                                     # main axis of the left half

# ---------------- graph: references -> target ----------------
import math
T = (52, MY)
REFS = [(T[0] + 38 * math.cos(math.radians(a)), MY - 38 * math.sin(math.radians(a)))
        for a in (60, 125, 180, 235, 300)]
shape(T[0] - 48, MY - 48, 96, 96, "", SKY, stroke="none", kind="ellipse;")     # 1-hop region
for r in REFS:                              # border of a reference (r 6) to border of the target (r 9)
    dx, dy = T[0] - r[0], T[1] - r[1]; L = math.hypot(dx, dy)
    edge([(r[0] + 6 * dx / L, r[1] + 6 * dy / L), (T[0] - 9.5 * dx / L, T[1] - 9.5 * dy / L)])
for r in REFS:
    circ(*r, 12, WHITE)
circ(*T, 18, WHITE, sw=TW, extra=DASHED)     # target: its own embedding is not an input

# ---------------- GNN -> vector -> prefix adapter ----------------
edge([(T[0] + 45, MY), (110, MY)])           # from the edge of the 1-hop region
shape(110, MY - 8, 30, 16, "GNN", TRAIN, sw=TW)
edge([(140, MY), (148, MY)])
shape(148, MY - C / 2, C, C, "", WHITE)                       # the estimated embedding
edge([(161, MY), (169, MY)])
shape(169, MY - 13, 44, 26, "prefix<br>adapter", TRAIN, sw=TW)

# ---------------- LLM: layer 1, dots, layer 40 ----------------
BX0, BX1 = 240, 420                          # layer bars
BH = 18
Y40, Y1 = 30, 92                             # top of layer-40 and layer-1 bars
COLS = [362, 382, 402]                            # token columns (cell left edges)
BUS = 225

def layer(y, name):
    shape(BX0, y, BX1 - BX0, BH, "", GREY)
    for k in range(2):                                         # key and value prefix
        shape(BX0 + 5 + 16 * k, y + (BH - C) / 2, C, C, "", TRAIN, sw=TW)
    text(BX0 + 37, y, 40, BH, name)
    shape(BX0 + 80, y + 3, 30, BH - 6, "LoRA", TRAIN, sw=TW, extra="verticalAlign=middle;")
    edge([(BUS, y + BH / 2), (BX0 + 5, y + BH / 2)])

layer(Y40, "layer 40")
layer(Y1, "layer 1")
text(BX0 + 37, Y40 + BH + 2, 40, Y1 - Y40 - BH - 4, r"\(\vdots\)", size=9)
text(BX0 + 5, Y40 + BH + 2, 29, Y1 - Y40 - BH - 4, r"\(\vdots\)", size=9)   # prefixes of the layers between
for x in COLS:
    text(x, Y40 + BH + 2, C, Y1 - Y40 - BH - 4, r"\(\vdots\)", size=9)
edge([(213, MY), (BUS, MY)], arrow=False)
edge([(BUS, Y40 + BH / 2), (BUS, Y1 + BH / 2)], arrow=False)

# prompt tokens in, abstract tokens out
YIN, YOUT = Y1 + BH + 10, Y40 - 10 - C
for x in COLS:
    shape(x, YIN, C, C, "", WHITE)
    edge([(x + C / 2, YIN), (x + C / 2, Y1 + BH)])
    shape(x, YOUT, C, C, "", WHITE)
    edge([(x + C / 2, Y40), (x + C / 2, YOUT + C)])
mid = (COLS[0] + COLS[-1] + C) / 2
text(mid - 50, YIN + C + 1, 100, 13, "Write the abstract.")
text(mid - 50, YOUT - 14, 100, 13, "abstract")

write(sys.argv[1], name="method_overview", math=True)
