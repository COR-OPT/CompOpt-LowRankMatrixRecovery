#!/bin/bash

python matcomp_phase.py --noheader --i matcomp_subgrad.csv \
	--out_file matcomp_subgrad.svg
inkscape -D -z --file=matcomp_subgrad.svg \
	--export-pdf=matcomp_subgrad.pdf --export-latex
