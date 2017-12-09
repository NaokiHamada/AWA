# Configure AWA's parameters
This example shows a configuration to setup AWA's parameters for tuning search performance.

AWA's parameters can be configured in the `optimizer` section:
```json
{
  "optimizer": {
    "seed": 42,
    "max_iters": 3,
    "max_evals": {
      "(1,0)": 64,
      "(0,1)": 64,
      "(1,1)": 32,
      "(3,1)": 16,
      "(1,3)": 16
    },
    "scalarization": "weighted_sum",
    "optimization": "cmaes",
    "x0": [
      [0.7, 0.7],
      [0.3, 0.3]
    ],
    "w0": [
      [0.8, 0.2],
      [0.3, 0.7]
    ]
  }
}
```
- `seed`: A seed of the random number generator.
- `max_iters`: The number of iterations to stop AWA.
- `max_evals`: The number of evaluations to stop CMA-ES running in AWA.
- `scalarization`: The scalarization method.
- `optimization`: The optimization method.
- `x0`: Initial solutions.
- `w0`: Initial weights.

See [specifications](../../FORMATS.md#optimizer) for more details.

## Run a Demo
```
./run.sh
```

