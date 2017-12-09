# Use Timeout to Terminate Evaluations
This example shows a configuration to terminate time-consuming evaluations.

In some problems, the evaluation time of objective functions can significantly vary.
For instance, hyperparameter tuning for deep neural networks involves network training of different model size.

For such a task, one can use _timeout_ to terminate bottlenecking evaluations in order to accelerate search.
AWA does not provide a special support to do timeout, but it can be done by simply adding an operating system's timeout command to `command` of `workers` in a configuration file.
The evaluation value of timeouted solutions are set to `inf`.

The following sample code implements a timeout by using Linux's `timeout` command.
The execution of the objective function [time_consuming_sphere_2D.sh](time_consuming_sphere_2D.sh) takes 0-4 seconds.
Evaluations taking 2 or more seconds will be terminated by the `timeout` command.
```json
{
  "workers": [
    { "command": "timeout 2s ./time_consuming_sphere_2D.sh $x $y" }
  ]
}
```

## Run a Demo
```
./run.sh
```
