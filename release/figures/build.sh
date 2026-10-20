#!/usr/bin/env bash
# Rebuild the paper's figures and the animated routing schematic.
# Needs pdflatex (TeX Live 2023 or later, with pgfplots), poppler-utils
# (pdftocairo, pdftoppm), ffmpeg with libx264, and Python 3.
set -euo pipefail
cd "$(dirname "$0")"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

latex() {  # latex FILE.tex: compile into $work, show the log on failure
  local log="$work/$(basename "$1" .tex).build.log"
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory "$work" "$1" >"$log" \
    || { tail -n 40 "$log"; exit 1; }
}

# Figures 1 and 2: PDF (vector), SVG (vector), PNG (300 dpi).
for fig in fig1-routing fig2-curves; do
  latex "src/$fig.tex"
  cp "$work/$fig.pdf" "$fig.pdf"
  pdftocairo -svg "$fig.pdf" "$fig.svg"
  pdftoppm -png -r 300 -singlefile "$fig.pdf" "$fig"
done

# Animated schematic: one PDF page per frame, rasterized, then encoded.
python3 src/routing_animation.py "$work/routing-animation.tex"
latex "$work/routing-animation.tex"
mkdir -p "$work/frames"
pdftoppm -png -gray -r 200 "$work/routing-animation.pdf" "$work/frames/f"
python3 src/routing_animation.py --encode "$work/frames" .
echo "built: $(ls fig1-routing.* fig2-curves.* routing-schematic.* | tr '\n' ' ')"
