#!/bin/sh
# Generate a .drawio from its generator, export the PDF, and check it at print size.
# Usage: build_drawio.sh GENERATOR.py OUT_DIR NAME [PREVIEW_DIR]
#   -> OUT_DIR/NAME.drawio, OUT_DIR/NAME.pdf, PREVIEW_DIR/NAME_preview.png (300 dpi)
set -e
GEN=$1; OUT=$2; NAME=$3; PREV=${4:-$OUT}
DRAWIO=/Applications/draw.io.app/Contents/MacOS/draw.io
mkdir -p "$OUT" "$PREV"
python3 "$GEN" "$OUT/$NAME.drawio"
"$DRAWIO" -x -f pdf --crop -o "$OUT/$NAME.pdf" "$OUT/$NAME.drawio" 2>&1 | grep -v ERROR || true
pdftoppm -png -r 300 -singlefile "$OUT/$NAME.pdf" "$PREV/${NAME}_preview"
# 1 pt = 0.3528 mm; the width must stay <= 113 mm (320 pt)
pdfinfo "$OUT/$NAME.pdf" | awk '/Page size/ {printf "size: %.1f x %.1f mm\n", $3*0.3528, $5*0.3528}'
pdffonts "$OUT/$NAME.pdf" | tail -n +3 | awk '{print "font:", $1, "emb:", $(NF-4)}'
