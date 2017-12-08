#!/bin/sh

###############################################################################
# This example shows how to run experiments in multiple nodes.
###############################################################################

# Register an SSH public key to all worker nodes
ssh 192.168.0.1 "echo $(cat ~/.ssh/id_rsa.pub) >> ~/.ssh/authorized_keys"
ssh 192.168.0.2 "echo $(cat ~/.ssh/id_rsa.pub) >> ~/.ssh/authorized_keys"

# Copy the objective function script to all worker nodes
scp sphere_2D.sh 192.168.0.1:
scp sphere_2D.sh 192.168.0.2:

# Run an experiment with config.json
python -m awa  # Produce solutions.csv

# Show the results
cat solutions.csv
