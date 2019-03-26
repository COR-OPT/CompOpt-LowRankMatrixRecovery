#!/bin/bash

IMG_BILIN="matsense_bilinear"
IMG_SQUAD="matsense_symquad"

for img in $IMG_BILIN $IMG_SQUAD; do
	echo "Generating $img..."
	python mtxsensing_phase.py --noheader -i ${img}_rank_{1,5,10}.csv \
		--out_file ${img}.svg
	inkscape -D -z --file="${img}.svg" --export-pdf="${img}.pdf" --export-latex
done
