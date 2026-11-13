"""v3 (corner layout with branches, superseded 2026-11-04). Generator for lora_prefix.drawio (thesis Fig. 5.4, Ch. 5 §5.3), v3. Usage: python make_lora_prefix.py OUT.drawio

One attention head as a corner-shaped matrix product (the author's sketch; v1 two panels and
v2 one row are kept as make_lora_prefix_v1.py / _v2.py):
Q left of A (rows aligned), [P^K; K]^T above A (columns aligned), "scale, causal mask, softmax"
in the empty corner; A x [P^V; V] = O to the right. Q is built from X by a frozen query
projection plus a LoRA query update; the token block of V likewise, from below. The prefix
adapter feeds P^K and P^V directly, bypassing both additions.
"""
import sys
sys.path.insert(0, "/Users/gemacsanad/egyetem/DyTAG/baseline/.claude/skills/thesis-figure/scripts")
from drawio_kit import *

TW = 1.5                                    # outline of trainable modules
C = 9                                       # cell: one drawn position
NP, NT = 3, 4                               # drawn prefix and token positions (stand for 32 and n)
U = 34                                      # the head dimension, 128
LP, LT = NP * C, NT * C
AX, AY = 170, 100                           # attention matrix, top left
AW = LP + LT
ACY = AY + LT / 2                           # centre row of A
BW, BH = 62, 22                             # projection boxes
SD = 12                                     # plus sign


def plus(cx, cy):
    shape(cx - SD / 2, cy - SD / 2, SD, SD, "+", WHITE, kind="ellipse;", extra="verticalAlign=middle;")


# ---------------- attention matrix A ----------------
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
text(AX - 4, AY + LT + 1, LP + 8, 11, "32", size=8)
text(AX + LP, AY + LT + 1, LT, 11, r"\(n\)", size=8)
text(AX - 20, AY + LT + 12, AW + 40, 12, r"\(\mathbf{A}\): \(n \times (32+n)\)", size=8)

# ---------------- keys above A, the corner annotation ----------------
KY = AY - 6 - U
shape(AX, KY, LP, U, r"\(\mathbf{P}^{K\top}\)", TRAIN, size=10)
shape(AX + LP, KY, LT, U, r"\(\mathbf{K}^\top\)", WHITE, size=10)
text(AX + LP / 2 - 116, KY - 17, 104, 12, r"\([\mathbf{P}^K;\mathbf{K}]^\top\): \(128 \times (32+n)\)", size=8,
     extra="align=right;")
text(AX - 86, KY + 4, 80, 24, "scale, causal mask,<br>softmax", size=8, extra="align=right;")

# ---------------- Q left of A, built from X ----------------
QX = AX - 8 - U
shape(QX, AY, U, LT, r"\(\mathbf{Q}\)", WHITE, size=10)
text(QX - 8, AY + LT + 1, U + 16, 11, r"\(n \times 128\)", size=8)
PX = QX - 14                                # centre of the plus sign
plus(PX, ACY)
edge([(PX + SD / 2, ACY), (QX, ACY)])
BX = PX - 14 - BW
for dy, t, fill, sw in ((-15, "frozen query<br>projection", GREY, 1), (15, "LoRA query<br>update", TRAIN, TW)):
    shape(BX, ACY + dy - BH / 2, BW, BH, t, fill, sw=sw, size=8)
    edge([(BX + BW, ACY + dy), (PX, ACY + dy), (PX, ACY - (SD / 2 if dy < 0 else -SD / 2))])
XB = BX - 12                                # bus from X
edge([(XB, ACY - 15), (XB, ACY + 15)], arrow=False)
for dy in (-15, 15):
    edge([(XB, ACY + dy), (BX, ACY + dy)])
edge([(XB - 10, ACY), (XB, ACY)], arrow=False)
text(XB - 24, ACY - 6, 14, 12, r"\(\mathbf{X}\)", size=10)

# ---------------- [P^V; V] right of A, V built from X below ----------------
text(AX + AW + 4, ACY - 6, 12, 12, "×", size=10)
VX = AX + AW + 30
VY = ACY - (LP + LT) / 2
shape(VX, VY, U, LP, r"\(\mathbf{P}^V\)", TRAIN, size=10)
shape(VX, VY + LP, U, LT, r"\(\mathbf{V}\)", WHITE, size=10)
text(VX - 14, VY + LP / 2 - 6, 12, 12, "32", size=8, extra="align=right;")
text(VX - 14, VY + LP + LT / 2 - 6, 12, 12, r"\(n\)", size=8, extra="align=right;")
text(VX + U + 4, VY - 2, 100, 12, r"\([\mathbf{P}^V;\mathbf{V}]\): \((32+n) \times 128\)", size=8,
     extra="align=left;")
VB = VY + LP + LT                           # bottom of V
VCX = VX + U / 2
plus(VCX, VB + 14)
edge([(VCX, VB + 14 - SD / 2), (VCX, VB)])
BY = VB + 34                                # top of the value projection boxes
for dx, t, fill, sw in ((-BW / 2 - 6, "frozen value<br>projection", GREY, 1),
                        (BW / 2 + 6, "LoRA value<br>update", TRAIN, TW)):
    cx = VCX + dx
    shape(cx - BW / 2, BY, BW, BH, t, fill, sw=sw, size=8)
    edge([(cx, BY), (cx, VB + 14), (VCX + (-SD / 2 if dx < 0 else SD / 2), VB + 14)])
XY = BY + BH + 12                           # bus from X
edge([(VCX - BW / 2 - 6, XY), (VCX + BW / 2 + 6, XY)], arrow=False)
for dx in (-BW / 2 - 6, BW / 2 + 6):
    edge([(VCX + dx, XY), (VCX + dx, BY + BH)])
edge([(VCX, XY), (VCX, XY + 8)], arrow=False)
text(VCX - 7, XY + 8, 14, 12, r"\(\mathbf{X}\)", size=10)

# ---------------- output ----------------
text(VX + U + 6, ACY - 6, 12, 12, "=", size=10)
OX = VX + U + 24
shape(OX, AY, U, LT, r"\(\mathbf{O}\)", WHITE, size=10)
text(OX - 8, AY + LT + 1, U + 16, 11, r"\(n \times 128\)", size=8)

# ---------------- the prefix path ----------------
DY = KY - 46
DX, DW = AX - 22, 72
shape(DX, DY, DW, 18, "prefix adapter", TRAIN, sw=TW)
text(DX - 82, DY, 68, 18, "GNN estimate", extra="align=right;")
edge([(DX - 12, DY + 9), (DX, DY + 9)])
edge([(AX + LP / 2, DY + 18), (AX + LP / 2, KY)])
edge([(DX + DW, DY + 9), (VCX, DY + 9), (VCX, VY)])

write(sys.argv[1], name="lora_prefix", math=True)
