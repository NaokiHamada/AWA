#!/bin/sh
env parameters=$@ qsub -V job.sh -o out -N abc
while [ `qstat | grep abc` ]
do
    sleep 10
done
tail -n1 out
