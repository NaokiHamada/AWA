# Suspend and Resume
This example shows a configuration to suspend and resume AWA.

AWA records evaluated solutions into a cache file.
This file provides a checkpointing that enables us to shutdown AWA anytime and restart from the state previously shutted down.

Let us run two iterations of AWA from scratch.
Its configuration file is as follows:
```json
{
  "cache": "solutions.csv",
  "optimizer": {
    "seed": 42,
    "max_iters": 2
  }
}
```
The cache file path is specified in `cache`.
Also notice that we use a fixed random seed for reproducibility.

Then, let us resume the run.
The configuration file becomes as follows:
```json
{
  "cache": "solutions.csv",
  "optimizer": {
    "seed": 42,
    "max_iters": 3
  }
}
```
Notice that we increase `max_iters` for continuation and keep the rest settings unchanged for recovering the previous state.

During the second run, the already-evaluated solutions in the first run are read from the cache file and new solutions are evaluated and appended to `solutions.csv`.

## Run a Demo
```
./run.sh
```

