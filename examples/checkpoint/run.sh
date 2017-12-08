#!/bin/sh

###############################################################################
# This example shows how to use checkpointing to resume experiments.
###############################################################################

# Run an experiment
python -m awa -c config_1.json  # Produce solutions.csv

# Continue the experiment from the end of the previous run
python -m awa -c config_2.json  # Append to solutions.csv

# Show the results
cat solutions.csv
