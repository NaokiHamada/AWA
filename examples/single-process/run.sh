#!/bin/sh

###############################################################################
# This example shows how to run experiments in a single process.
###############################################################################

# Run an experiment with config.json
python -m awa  # Produce solutions.csv

# Show the results
cat solutions.csv
