#!/usr/bin/env bash

if [[ $# -eq 0 ]]; then
    echo "Must pass notebook name..."
    exit 1
fi

notebook=${1}

jupyter nbconvert ${notebook} --to "python"

