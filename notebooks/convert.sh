#!/usr/bin/env bash

jupyter nbconvert "${1}" \
    --TagRemovePreprocessor.remove_cell_tags='{"remove_cell"}' \
    --TagRemovePreprocessor.remove_all_outputs_tags='{"remove_output"}' \
    --TagRemovePreprocessor.remove_input_tags='{"remove_input"}'

