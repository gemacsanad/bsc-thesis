"""Generator for lora_prefix.drawio (thesis Fig. 5.4, Ch. 5 §5.3), v5. Usage: python make_lora_prefix.py OUT.drawio

Both products of one attention head as matrix-multiplication corners (author, 2026-11-04; v1-v4
are kept as make_lora_prefix_v1.py ... _v4.py):
- Q left of A, [P^K; K]^T above A: Q [P^K; K]^T -> A;
- A left of O, [P^V; V] above O:  A [P^V; V] = O.
One length unit per position (C) and one per 128 features (W), so Q and O are identical, Q, A, O
share the height n, the keys share A's width 32 + n, and the values share O's width 128 and A's
width as their height. Q and the token block V hold the inset "frozen projection + LoRA update";
the prefix blocks are tinted and labelled as outputs of the prefix adapter. The only dimensions
drawn are the 32 prefix positions and the expression for the second product below.
"""
import sys
sys.path.insert(0, "/Users/gemacsanad/egyetem/DyTAG/baseline/.claude/skills/thesis-figure/scripts")
from drawio_kit import *

C = 18                                      # one drawn position
NP, NT = 3, 4                               # drawn prefix and token positions (stand for 32 and n)
LP, LT = NP * C, NT * C
W = 116                                     # 128 features (room for the inset)
G = 10                                      # gap between neighbouring matrices
AX, AY = W + G, W + 2 * C + G + 30          # attention matrix, top left
AW = LP + LT
OX = AX + AW + G                            # output, right of A


def produced(x, y, name):
    """A block named `name` with the inset: frozen projection + LoRA update."""
    shape(x, y, W, LT, "", WHITE)
    text(x, y + 3, W, 12, rf"\(\mathbf{{{name}}}\)", size=10)
    shape(x + 7, y + 18, W - 14, 17, "frozen projection", GREY, size=9)
    text(x, y + 35, W, 11, "+", size=9)
    shape(x + 7, y + 47, W - 14, 17, "LoRA update", TRAIN, sw=1.5, size=9)


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
text(AX, AY + LT + 2, LP, 12, "32", size=9)
text(AX, AY + LT + 2, AW, 12, r"\(\mathbf{A}\)", size=10)

# first corner: Q left of A, keys above A (height: 128 features)
produced(AX - G - W, AY, "Q")
KY = AY - G - W
shape(AX, KY, LP, W, r"\(\mathbf{P}^{K\top}\)", TRAIN, size=10)
shape(AX + LP, KY, LT, W, r"\(\mathbf{K}^\top\)", WHITE, size=10)
text(AX + LP / 2 - 40, KY - 28, 80, 26, "from the<br>prefix adapter", size=9)

# second corner: O right of A, values above O (height: 32 + n positions)
shape(OX, AY, W, LT, r"\(\mathbf{O}\)", WHITE, size=10)
VY = AY - G - AW
shape(OX, VY, W, LP, r"\(\mathbf{P}^V\)", TRAIN, size=10)
produced(OX, VY + LP, "V")
text(OX + W + 4, VY + LP / 2 - 6, 16, 12, "32", size=9, extra="align=left;")
text(OX + W / 2 - 40, VY - 28, 80, 26, "from the<br>prefix adapter", size=9)

# guides: thin grey lines in the gaps marking each product's operands and result
GC = "#B0B0B0"
gx1, gx2, gy = AX - G / 2, OX - G / 2, AY - G / 2
edge([(AX - G - W, gy), (AX + AW, gy)], arrow=False, color=GC, width=0.75)      # 1: keys | A, over Q
edge([(gx1, KY), (gx1, AY + LT)], arrow=False, color=GC, width=0.75)             # 1: Q | A
edge([(gx2, gy), (OX + W, gy)], arrow=False, color=GC, width=0.75)               # 2: values | O
edge([(gx2, VY), (gx2, AY + LT)], arrow=False, color=GC, width=0.75)             # 2: A | O


def mark(cx, cy, k):
    """Circled number: a corner and its explanation."""
    shape(cx - 6.5, cy - 6.5, 13, 13, str(k), WHITE, stroke=GC, size=9, kind="ellipse;",
          extra="verticalAlign=middle;")


mark(gx1, AY + LT + 10, 1)
mark(gx2, AY + LT + 10, 2)

# the second product, written out
text(AX - 10, AY + LT + 30, AW + G + W + 20, 24,
     r"\(\underbrace{n \times (32+n)}_{\textstyle\mathbf{A}} \cdot \underbrace{(32+n) \times 128}_{\textstyle[\mathbf{P}^V;\mathbf{V}]}"
     r" = \underbrace{n \times 128}_{\textstyle\mathbf{O}}\)", size=10)

# what the two corners compute
EY = AY + LT + 64
for k, t in ((1, "Query–key multiplication, followed by scaling, causal masking and softmax."),
             (2, "Attention weights multiplied by the combined prefix and token values.")):
    mark(AX - G - W + 7, EY + 7 + (k - 1) * 18, k)
    text(AX - G - W + 18, EY + 1 + (k - 1) * 18, 340, 12, t, size=9, extra="align=left;")

write(sys.argv[1], name="lora_prefix", math=True)
