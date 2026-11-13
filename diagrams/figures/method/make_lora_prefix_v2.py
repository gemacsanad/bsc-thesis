"""v2 (one row, superseded 2026-11-04). Generator for lora_prefix.drawio (thesis Fig. 5.4, Ch. 5 §5.3), v2. Usage: python make_lora_prefix.py OUT.drawio

One attention head, left to right (the author's layout; v1, two panels with projection branches,
is kept as make_lora_prefix_v1.py):
Q_LoRA x [P^K; K]^T -> scale, causal mask, softmax -> A x [P^V; V_LoRA] = O.
Prefix blocks (filled) bypass the token projections; the LoRA-adapted blocks (Q, token values)
have a thick outline. The shape of every block is written under it; a two-line legend below.
"""
import sys
sys.path.insert(0, "/Users/gemacsanad/egyetem/DyTAG/baseline/.claude/skills/thesis-figure/scripts")
from drawio_kit import *

TW = 1.5                                    # outline of the LoRA-adapted blocks
PRE = TRAIN                                 # fill of the prefix blocks
C = 9                                       # cell: one drawn position
NP, NT = 3, 4                               # drawn prefix and token positions (stand for 32 and n)
U = 34                                      # the head dimension, 128
CY = 50                                     # main axis
SY = 84                                     # row of the shape labels
LP, LT = NP * C, NT * C                     # lengths of the prefix and token parts


def op(x, t):
    """Multiplication or equals sign centred at x on the main axis."""
    text(x - 6, CY - 6, 12, 12, t, size=10)


def shape_label(cx, t, w=64):
    text(cx - w / 2, SY, w, 12, t, size=8)


# 1. queries
x = 0
shape(x, CY - LT / 2, U, LT, r"\(\mathbf{Q}_{\text{LoRA}}\)", WHITE, sw=TW, size=10)
shape_label(x + U / 2, r"\(n \times 128\)")
text(x + U / 2 - 30, SY + 12, 60, 22, "LoRA-adapted<br>projection", size=8)
op(x + U + 8, "×")

# 2. keys, transposed: prefix block, then token block
x = U + 16
shape(x, CY - U / 2, LP, U, r"\(\mathbf{P}^{K\top}\)", PRE, size=10)
shape(x + LP, CY - U / 2, LT, U, r"\(\mathbf{K}^\top\)", WHITE, size=10)
text(x - 4, CY + U / 2 + 1, LP + 8, 11, "32", size=8)
text(x + LP, CY + U / 2 + 1, LT, 11, r"\(n\)", size=8)
shape_label(x + (LP + LT) / 2, r"\(128 \times (32+n)\)")
x += LP + LT

# 3. the product Q [P^K; K]^T (dashed frame) -> scale, causal mask, softmax -> A
shape(-4, CY - LT / 2 - 4, x + 8, LT / 2 + 4 + U / 2 + 14, "", "none", stroke=FAINT, extra=DASHED)
edge([(x + 4, CY), (x + 13, CY)])
shape(x + 13, CY - 14, 62, 28, "scale, causal<br>mask, softmax", WHITE, size=8)
edge([(x + 75, CY), (x + 85, CY)])
x += 85
for i in range(NT):
    for j in range(NP + NT):
        if j < NP:
            fill = PRE                      # prefix column: visible to every token
        elif j - NP <= i:
            fill = SKY                      # earlier or same token position: visible
        else:
            fill = WHITE                    # later token position: masked
        shape(x + j * C, CY - LT / 2 + i * C, C, C, "", fill, stroke=FAINT, sw=0.5)
shape(x, CY - LT / 2, LP + LT, LT, "", "none")
edge([(x + LP, CY - LT / 2), (x + LP, CY + LT / 2)], arrow=False)
text(x, CY - LT / 2 - 13, LP + LT, 12, r"\(\mathbf{A}\)")
text(x - 4, CY + LT / 2 + 1, LP + 8, 11, "32", size=8)
text(x + LP, CY + LT / 2 + 1, LT, 11, r"\(n\)", size=8)
shape_label(x + (LP + LT) / 2, r"\(n \times (32+n)\)")
x += LP + LT
op(x + 9, "×")

# 4. values: prefix block on top, token block below
x += 30
top = CY - (LP + LT) / 2
shape(x, top, U, LP, r"\(\mathbf{P}^V\)", PRE, size=10)
shape(x, top + LP, U, LT, r"\(\mathbf{V}_{\text{LoRA}}\)", WHITE, sw=TW, size=10)
text(x - 16, top + LP / 2 - 6, 13, 12, "32", size=8, extra="align=right;")
text(x - 16, top + LP + LT / 2 - 6, 13, 12, r"\(n\)", size=8, extra="align=right;")
shape_label(x + U / 2, r"\((32+n) \times 128\)")
text(x + U / 2 - 30, SY + 12, 60, 22, "LoRA-adapted<br>projection", size=8)
x += U
op(x + 12, "=")

# 5. output
x += 24
shape(x, CY - LT / 2, U, LT, r"\(\mathbf{O}\)", WHITE, size=10)
shape_label(x + U / 2, r"\(n \times 128\)")

# legend
LY = SY + 40
LX = (x + U) / 2 - 106
shape(LX, LY, 10, 10, "", PRE)
text(LX + 15, LY - 1, 196, 12, "Prefix: computed separately for each paper.", size=8, extra="align=left;")
shape(LX, LY + 16, 10, 10, "", WHITE, sw=TW)
text(LX + 15, LY + 15, 196, 12, "LoRA: learned projection updates shared across papers.", size=8, extra="align=left;")

write(sys.argv[1], name="lora_prefix", math=True)
