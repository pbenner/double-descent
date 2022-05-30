#! /bin/bash

pandoc double-descent-examples.ipynb -o README.md --to=gfm --extract-media=README_files --webtex

sed -i 's,README_files/,https://raw.githubusercontent.com/pbenner/double-descent/master/README_files/,g' README.md
sed -i 's,\\#,,'g README.md