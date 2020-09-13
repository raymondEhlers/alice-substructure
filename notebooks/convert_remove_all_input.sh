#!/usr/bin/env bash

jupyter nbconvert hardestKtSummary.ipynb --to html \
    --TagRemovePreprocessor.remove_cell_tags='{"remove_cell"}' \
    --TagRemovePreprocessor.remove_all_outputs_tags='{"remove_output"}' \
    --TagRemovePreprocessor.remove_input_tags='{"remove_input"}' \
     --no-input --no-prompt --template "classic"

# TODO: Convert to template "lab", but need to figure out how to keep it from filling the entire width of the page.
