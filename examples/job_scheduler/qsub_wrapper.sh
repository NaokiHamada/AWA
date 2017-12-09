#!/bin/sh

set -e

# Make a unique job name by MD5 for avoiding name confliction
name=$(md5sum <(echo $*) | cut -d' ' -f1)

# File names for the standard output and the standard error
outfile=${name}.out
errfile=${name}.err

# Submit an evaluation job
qsub -N ${name} -o ${outfile} -e ${errfile} sphere_2D.sh ${parameters}

# Wait for the job completion
while [ $(qstat | grep ${name}) ]
do
    sleep 10
done

# Write an evaluation value
tail -n1 ${outfile}

# Clear temporary files
rm -f ${outfile} ${errfile}
