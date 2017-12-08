#!/bin/sh

###############################################################################
# This example shows how to ensure the reproducibility.
# 1. Run two experiments with a fixed random seed
# 2. They must produce the same solutions
###############################################################################

# Run the first experiment with a random seed
python -m awa -c config_1.json  # Produce results_1.csv

# Run the second experiment with the same seed
python -m awa -c config_2.json  # Produce results_2.csv

# Cut the timestamp columns off
cut -d, -f2,3 --complement results_1.csv > solutions_1.csv
cut -d, -f2,3 --complement results_2.csv > solutions_2.csv

# They should have the same solutions
diff solutions_1.csv solutions_2.csv  # No output means the same contents
