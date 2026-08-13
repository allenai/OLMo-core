MAIN := iclr2026/iclr2026_conference

.PHONY: pdf watch clean

# Build from THIS directory: the .tex files reference `iclr2026/sections/...` and
# `figures/...` relative to it, and TEXINPUTS lets graphicx/\input also find the
# .sty/.bst files that live inside iclr2026/.
pdf:
	TEXINPUTS="iclr2026:" latexmk -pdf -interaction=nonstopmode -output-directory=iclr2026 $(MAIN).tex

watch:
	TEXINPUTS="iclr2026:" latexmk -pdf -pvc -interaction=nonstopmode -output-directory=iclr2026 $(MAIN).tex

clean:
	latexmk -C -output-directory=iclr2026 $(MAIN).tex
