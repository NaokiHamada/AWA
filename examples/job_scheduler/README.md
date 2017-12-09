# Run AWA with Multiple Workers on a Job Scheduler
This example shows a configuration for running AWA with multiple workers that dispatch the evaluation of objective functions to an SGE job.

One of the main difficulties in cooperating with a job scheduler is that `qsub` command in SGE returns immediately after an evaluation job is queued, not waiting for the completion of the job.
It would not work to write a `qsub` command in `command` of `workers` in a configuration file.
You need to write a wrapper script `qsub_wrapper.sh` that performs the followings:
1. execute `qsub` command to queue an objective function evaluation
2. run an infinite loop to watch the progress of the job
3. if the job finishes, exit the loop and read a standard output log
4. print the last line of the log on the standard output

## Run a Demo
```
./run.sh
```

