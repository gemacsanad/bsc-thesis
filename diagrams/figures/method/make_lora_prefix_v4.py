"""v4 (one corner + linear value product, superseded 2026-11-04). Generator for lora_prefix.drawio (thesis Fig. 5.4, Ch. 5 §5.3), v4. Usage: python make_lora_prefix.py OUT.drawio

The corner-shaped matrix product of one attention head, simplified (author, 2026-11-04; v1-v3
are kept as make_lora_prefix_v1.py ... _v3.py): Q left of A, [P^K; K]^T above A, A x [P^V; V] = O.
Q and the token block V each hold a small inset "frozen projection + LoRA update" (how they are
produced, not an operand). The prefix blocks are tinted and labelled as outputs of the prefix
adapter. No branches, no X, no adapter network; the only number is the 32 prefix positions.
"""
import sys
sys.path.insert(0, "/Users/gemacsanad/egyetem/DyTAG/baseline/.claude/skills/thesis-figure/scripts")
from drawio_kit import *

C = 14                                      # cell: one drawn position
NP, NT = 3, 4                               # drawn prefix and token positions (stand for 32 and n)
LP, LT = NP * C, NT * C
W = 88                                      # width of Q and [P^V; V] (room for the inset)
KH = 34                                     # height of [P^K; K]^T
AX, AY = 110, 60                            # attention matrix, top left
AW = LP + LT
ACY = AY + LT / 2


def produced(x, y, name):
    """A block named `name` with the inset: frozen projection + LoRA update."""
    shape(x, y, W, LT, "", WHITE)
    text(x, y + 2, W, 12, rf"\(\mathbf{{{name}}}\)", size=10)
    shape(x + 5, y + 15, W - 10, 14, "frozen projection", GREY, size=8)
    text(x, y + 29, W, 10, "+", size=8)
    shape(x + 5, y + 40, W - 10, 14, "LoRA update", TRAIN, sw=1.5, size=8)


# attention matrix A: prefix columns, then causally masked token columns
for i in range(NT):
    for j in range(NP + NT):
        if j < NP:
            fill = TRAIN                    # prefix column: visible to every token
        elif j - NP <= i:
            fill = SKY                      # earlier or same token position: visible
        else:
            fill = WHITE                    # later token position: masked
        shape(AX + j * C, AY + i * C, C, C, "", fill, stroke=FAINT, sw=0.5)
shape(AX, AY, AW, LT, "", "none")
edge([(AX + LP, AY), (AX + LP, AY + LT)], arrow=False)
text(AX, AY + LT + 1, LP, 11, "32", size=8)
text(AX, AY + LT + 12, AW, 12, r"\(\mathbf{A}\)", size=10)

# Q to the left of A
produced(AX - 8 - W, AY, "Q")

# keys above A
KY = AY - 6 - KH
shape(AX, KY, LP, KH, r"\(\mathbf{P}^{K\top}\)", TRAIN, size=10)
shape(AX + LP, KY, LT, KH, r"\(\mathbf{K}^\top\)", WHITE, size=10)
text(AX + LP / 2 - 30, KY - 24, 60, 22, "from the<br>prefix adapter", size=8)

# values to the right of A
text(AX + AW + 4, ACY - 6, 12, 12, "×", size=10)
VX = AX + AW + 22
VY = ACY - (LP + LT) / 2
shape(VX, VY, W, LP, r"\(\mathbf{P}^V\)", TRAIN, size=10)
produced(VX, VY + LP, "V")
text(VX - 16, VY + LP / 2 - 6, 13, 12, "32", size=8, extra="align=right;")
text(VX + W / 2 - 50, VY - 13, 100, 12, "from the prefix adapter", size=8)

# output
text(VX + W + 4, ACY - 6, 12, 12, "=", size=10)
shape(VX + W + 20, AY, 34, LT, r"\(\mathbf{O}\)", WHITE, size=10)

write(sys.argv[1], name="lora_prefix", math=True)
