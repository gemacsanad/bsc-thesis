"""Generator for prefix_adapter.drawio (thesis Fig. 5.3, Ch. 5 §5.2), v2. Usage: python make_prefix_adapter.py OUT.drawio

Top to bottom, with the shape of every tensor beside the arrow that carries it (the author's
sketch; v1, a left-to-right row per layer, is kept as make_prefix_adapter_v1.py):
GNN estimate (1024) -> trunk -> projection, sliced into 80 codes of 192 -> 40 key codes to the
key decoder, 40 value codes to the value decoder (each shared by all layers) -> B + g Delta, with
the base prefix and the gates as side inputs -> prefix for 40 layers x {key, value}.
"""
import sys
sys.path.insert(0, "/Users/gemacsanad/egyetem/DyTAG/baseline/.claude/skills/thesis-figure/scripts")
from drawio_kit import *

TW = 1.5
CX = 160                                     # main axis
BW, BH = 170, 30                             # width and height of the main boxes


def label(title, detail):
    """Two-line box label: module name, then what it computes in smaller type."""
    return f'{title}<br><font style="font-size:8px">{detail}</font>'


def note(x, y, t, align="left"):
    """Shape annotation beside an arrow."""
    w = 110
    x0 = x if align == "left" else x - w
    text(x0, y, w, 12, t, size=8, extra=f"align={align};")


# input
text(CX - 60, 0, 120, 12, "GNN estimate")
edge([(CX, 13), (CX, 30)])
note(CX + 5, 15, "1024")

# trunk
shape(CX - BW / 2, 30, BW, BH, label("trunk", "linear, GELU, layer normalisation"), TRAIN, sw=TW)
edge([(CX, 60), (CX, 78)])
note(CX + 5, 63, "1024")

# projection, sliced into codes
shape(CX - BW / 2, 78, BW, BH, label("projection", "linear 1024 → 15,360, sliced"), TRAIN, sw=TW)
XK, XV = CX - 47, CX + 47                    # key and value branches
edge([(CX, 108), (CX, 118)], arrow=False)
edge([(XK, 118), (XV, 118)], arrow=False)
edge([(XK, 118), (XK, 138)])
edge([(XV, 118), (XV, 138)])
note(XK - 5, 122, "40 key codes × 192", align="right")
note(XV + 5, 122, "40 value codes × 192")

# two decoders, each shared by the 40 layers of its role
DW, DH = 88, 40
for x, role in ((XK, "key"), (XV, "value")):
    shape(x - DW / 2, 138, DW, DH, label(f"{role} decoder", "192 → 8 × 32 × 128<br>shared by all 40 layers"),
          TRAIN, sw=TW)
    edge([(x, 138 + DH), (x, 204)])
note(XK - 5, 185, r"\(\boldsymbol{\Delta}\): 40 × 8 × 32 × 128", align="right")
note(XV + 5, 185, r"\(\boldsymbol{\Delta}\): 40 × 8 × 32 × 128")

# combine with the gates and the base prefix (trainable side inputs)
CW = 130
shape(CX - CW / 2, 204, CW, BH, r"\(\mathbf{B} + g\,\boldsymbol{\Delta}\)", WHITE)
GW = 66
shape(CX - CW / 2 - 14 - GW, 204, GW, BH, label("gates \\(g\\)", "40 × 2 scalars"), TRAIN, sw=TW)
edge([(CX - CW / 2 - 14, 219), (CX - CW / 2, 219)])
RW = 96
shape(CX + CW / 2 + 14, 204, RW, BH, label("base \\(\\mathbf{B}\\)", "40 × 2 × 8 × 32 × 128"), TRAIN, sw=TW)
edge([(CX + CW / 2 + 14, 219), (CX + CW / 2, 219)])

# output
edge([(CX, 234), (CX, 252)])
text(CX - 120, 253, 240, 24,
     "prefix: 40 × 2 × 8 × 32 × 128<br>(layers × key/value × heads × positions × head dimension)")

# about 4 px of margin around the drawing (the PDF export crops to the content)
shape(CX - CW / 2 - 14 - GW - 4, -4, GW + RW + CW + 28 + 8, 285, "", "none", stroke="none")
write(sys.argv[1], name="prefix_adapter", math=True)
