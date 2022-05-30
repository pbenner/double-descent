#! /bin/bash

jupyter nbconvert double-descent-examples.ipynb --to markdown --output README.md

sed -i 's,README_files/,https://raw.githubusercontent.com/pbenner/modhmm/master/README_files/,g' README.md
