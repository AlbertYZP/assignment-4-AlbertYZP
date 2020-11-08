#!/bin/bash
#NOTE: DO NOT EDIT THIS FILE-- MAY RESULT IN INCOMPLETE SUBMISSIONS
set -euo pipefail


CODE=(
	"cs231n/rnn_layers.py"
	"cs231n/classifiers/rnn.py"
	"cs231n/net_visualization_pytorch.py"
)
NOTEBOOKS=(
	"RNN_Captioning.ipynb"
	"LSTM_Captioning.ipynb"
	"NetworkVisualization-PyTorch.ipynb"
)


FILES=( "${CODE[@]}" "${NOTEBOOKS[@]}" )
ZIP_FILENAME="a3.zip"

for FILE in "${FILES[@]}"
do
	if [ ! -f ${FILE} ]; then
		echo -e "${C_R}Required file ${FILE} not found, Exiting.${C_E}"
		exit 0
	fi
done

echo -e "### Zipping file ###"
rm -f ${ZIP_FILENAME}
zip -q "${ZIP_FILENAME}" -r ${NOTEBOOKS[@]} $(find . -name "*.py") $(find . -name "*.pyx") -x "makepdf.py"

echo -e "### Creating PDFs ###"
python makepdf.py --notebooks "${NOTEBOOKS[@]}"

echo -e "### Done! Please submit a3.zip and the pdfs to Gradescope. ###"
