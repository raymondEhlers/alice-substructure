#!/usr/bin/env bash

if [[ $# -eq 0 ]]; then
    echo "Must pass notebook name..."
    exit 1
fi

notebook=${1}

#jupyter nbconvert hardestKtSummary.ipynb --to html_toc \
#jupyter nbconvert hardestKtSummary.ipynb --to html --template ../.venv/lib/python3.8/site-packages/jupyter_contrib_nbextensions/templates/toc2.tpl \
#jupyter nbconvert hardestKtSummary.ipynb --to html --template "html_with_toc" \
#jupyter nbconvert hardestKtSummary.ipynb --to html --template "html_toc" \
#jupyter nbconvert hardestKtSummary.ipynb --to html --template "test_html" \
#jupyter nbconvert hardestKtSummary.ipynb --to html --template "classic" \
#jupyter nbconvert hardestKtSummary.ipynb --to html --template "my_html_with_toc" \
jupyter nbconvert ${notebook} --to html --template "my_html_with_toc" \
    --TagRemovePreprocessor.remove_cell_tags='{"remove_cell"}' \
    --TagRemovePreprocessor.remove_all_outputs_tags='{"remove_output"}' \
    --TagRemovePreprocessor.remove_input_tags='{"remove_input"}' \
     --no-input --no-prompt --debug

# TODO: Convert to template "lab", but need to figure out how to keep it from filling the entire width of the page.
