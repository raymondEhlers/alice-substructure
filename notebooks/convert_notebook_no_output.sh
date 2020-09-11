#!/usr/bin/env bash

# Only works for nbconvert 6
# See: https://stackoverflow.com/a/47774393/12907985
#jupyter nbconvert --clear-output --to notebook --output=hardestKtSummary.noOutput.ipynb hardestKtSummary.ipynb
# For nbconvert 5
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to notebook --output=hardestKtSummary.noOutput.ipynb hardestKtSummary.ipynb

