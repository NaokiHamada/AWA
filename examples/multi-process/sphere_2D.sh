#!/bin/sh

###############################################################################
# Calculate the Sphere problem with 2 variables, 2 objectives
# f(x, y) := (f_1(x, y), f_2(x, y))
# where f_1(x, y) := (x - 1)^2 + y^2
#       f_2(x, y) := x^2 + (y - 1)^2
###############################################################################

# Set variables
x=$1
y=$2

# Calculate objectives and print them
python -c "f1 = (($x) - 1)**2 + ($y)**2;\
           f2 = ($x)**2 + (($y) - 1)**2;\
           print('{} {}'.format(f1, f2))"
