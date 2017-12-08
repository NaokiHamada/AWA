#!/bin/sh

###############################################################################
# This example shows how to run time-consuming experiments with timeout.
###############################################################################

# Run an experiment with config.json
python -m awa  # Produce solutions.csv

# Show the results
cat solutions.csv
