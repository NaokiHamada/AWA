#!/bin/sh

###############################################################################
# This example shows how to configure optimizer settings.
###############################################################################

# Run an experiment with config.json
python -m awa  # Produce solutions.csv

# Show the results
cat solutions.csv
