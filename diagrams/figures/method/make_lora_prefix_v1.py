"""v1 (two panels, superseded 2026-11-04). Generator for lora_prefix.drawio (thesis Fig. 5.4, Ch. 5 §5.3). Usage: python make_lora_prefix.py OUT.drawio

One query head and its key-value head in one layer, two panels.
(a) where LoRA acts: token hidden states X -> frozen W^Q, W^K, W^V; a trainable LoRA branch
    beside W^Q and W^V (outputs added), none beside W^K -> Q, K, V.
(b) where the prefix enters: GNN estimate -> prefix adapter -> P^K, P^V, which bypass the
    projections; drawn as a matrix product: Q (left) times [P^K; K]^T (above) gives the
    attention matrix A, whose first columns (prefix) are visible to every token and whose token
    columns are causally masked; A times [P^V; V] gives O. The equations are written below.
"""
import sys
sys.path.insert(0, "/Users/gemacsanad/egyetem/DyTAG/baseline/.claude/skills/thesis-figure/scripts")
from drawio_kit import *

TW = 1.5                                    # outline of trainable elements
C = 9                                       # cell of the attention matrix
NP, NT = 3, 4                               # prefix and token columns drawn (32 and n)


def math(x, y, w, h, t, size=9, extra=""):
    text(x, y, w, h, t, size=size, extra=extra)


# ---------------- (a) where LoRA acts ----------------
YQ, YK, YV = 54, 92, 130                    # rows of the three projections
LQ, LV = YQ - 26, YV + 26                   # rows of the LoRA branches
shape(0, YK - 18, 16, 36, r"\(\mathbf{X}\)", WHITE)                 # token hidden states
math(-12, YK + 20, 40, 12, r"\(n \times 5120\)", size=8)
BUS = 28
edge([(16, YK), (BUS, YK)], arrow=False)
edge([(BUS, LQ), (BUS, LV)], arrow=False)
PX, PW = 40, 40                             # projection boxes
for y, w in ((YQ, "Q"), (YK, "K"), (YV, "V")):
    edge([(BUS, y), (PX, y)])
    shape(PX, y - 9, PW, 18, rf"\(\mathbf{{W}}^{w}\)", GREY)
for y in (LQ, LV):
    edge([(BUS, y), (PX, y)])
    shape(PX, y - 9, PW, 18, "LoRA", TRAIN, sw=TW)
math(PX - 30, LV + 11, PW + 60, 12, "LoRA weights: shared by all papers", size=8)

SX, SD = 100, 12                            # the sums
OX = 122                                    # output matrices
for y, ly, sign in ((YQ, LQ, 1), (YV, LV, -1)):
    edge([(PX + PW, y), (SX - SD / 2, y)])
    edge([(PX + PW, ly), (SX, ly), (SX, y - sign * SD / 2)])
    shape(SX - SD / 2, y - SD / 2, SD, SD, "+", WHITE, kind="ellipse;", extra="verticalAlign=middle;")
    edge([(SX + SD / 2, y), (OX, y)])
edge([(PX + PW, YK), (OX, YK)])
for y, w in ((YQ, "Q"), (YK, "K"), (YV, "V")):
    shape(OX, y - 9, 14, 18, rf"\(\mathbf{{{w}}}\)", WHITE)
math(OX - 13, YV + 11, 80, 12, r"\(\mathbf{Q}, \mathbf{K}, \mathbf{V}\): \(n \times 128\)", size=8)
panel_label(70, 193, "a")

# ---------------- (b) where the prefix enters ----------------
AX, AY = 230, 82                            # attention matrix, top left
AW, AH = (NP + NT) * C, NT * C
# Q to the left of A, rows aligned
QX = AX - 24
shape(QX, AY, 14, AH, r"\(\mathbf{Q}\)", WHITE)
# [P^K; K]^T above A, columns aligned
KY = AY - 20
shape(AX, KY, NP * C, 14, r"\(\mathbf{P}^K\)", TRAIN, size=8)
shape(AX + NP * C, KY, NT * C, 14, r"\(\mathbf{K}\)", WHITE, size=8)
math(AX - 62, KY + 1, 58, 12, r"\([\mathbf{P}^K;\mathbf{K}]^\top\)", size=8, extra="align=right;")
# the attention matrix: prefix columns, then causally masked token columns
for i in range(NT):
    for j in range(NP + NT):
        x, y = AX + j * C, AY + i * C
        if j < NP:
            shape(x, y, C, C, "", TRAIN, stroke=FAINT, sw=0.5)
        elif j - NP <= i:
            shape(x, y, C, C, "", SKY, stroke=FAINT, sw=0.5)
        else:
            shape(x, y, C, C, "", WHITE, stroke=FAINT, sw=0.5)
shape(AX, AY, AW, AH, "", "none")
edge([(AX + NP * C, AY), (AX + NP * C, AY + AH)], arrow=False)
# brackets under A: 32 prefix columns, n token columns
math(AX - 6, AY + AH + 1, NP * C + 12, 11, "32", size=8)
math(AX + NP * C, AY + AH + 1, NT * C, 11, r"\(n\)", size=8)
math(AX - 10, AY + AH + 12, AW + 20, 12, r"\(\mathbf{A}\): \(n \times (32+n)\)", size=8)
# [P^V; V] to the right of A, then O
VX, VY = AX + AW + 12, AY - (NP * C) / 2 - 2
shape(VX, VY, 14, NP * C, r"\(\mathbf{P}^V\)", TRAIN, size=8)
shape(VX, VY + NP * C, 14, NT * C, r"\(\mathbf{V}\)", WHITE, size=8)
math(VX - 23, VY + (NP + NT) * C + 2, 60, 12, r"\([\mathbf{P}^V;\mathbf{V}]\)", size=8)
math(VX - 23, VY + (NP + NT) * C + 13, 60, 12, r"\((32+n) \times 128\)", size=8)
math(VX + 14, AY + AH / 2 - 6, 16, 12, "=")
OXB = VX + 30
shape(OXB, AY, 14, AH, r"\(\mathbf{O}\)", WHITE)

# the prefix path: GNN estimate -> prefix adapter -> P^K, P^V (bypassing the projections)
DX, DY, DW = AX - 4, 18, 64
math(DX - 82, DY, 68, 18, "GNN estimate", size=9, extra="align=right;")
edge([(DX - 12, DY + 9), (DX, DY + 9)])
shape(DX, DY, DW, 18, "prefix adapter", TRAIN, sw=TW)
edge([(AX + NP * C / 2, DY + 18), (AX + NP * C / 2, KY)])
edge([(DX + DW, DY + 9), (VX + 7, DY + 9), (VX + 7, VY)])
math(AX - 100, DY + 24, AX + NP * C / 2 - 7 - (AX - 100), 12, r"\(\mathbf{P}^K, \mathbf{P}^V\): paper-dependent", size=8, extra="align=right;")

# the computation, written out
EY = 161
math(QX - 20, EY, 200, 14,
     r"\(\mathbf{A} = \mathrm{softmax}\big(\mathbf{Q}[\mathbf{P}^K;\mathbf{K}]^\top / \sqrt{d_k} + \text{mask}\big)\)",
     size=8)
math(QX - 20, EY + 15, 200, 14, r"\(\mathbf{O} = \mathbf{A}\,[\mathbf{P}^V;\mathbf{V}]\)", size=8)
panel_label((QX + OXB + 14) / 2, 193, "b")

write(sys.argv[1], name="lora_prefix", math=True)
