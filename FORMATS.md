# File Format Specifications
This document describes AWA's configuration file and cache file with rigorous definitions and intuitive examples.

## Configuration file
The configuration of AWA specifies what optimization problem to be solved, which optimizer and its parameters to use, and how to compute objective functions.
The configuration file is written in the JSON format, and is passed as a runtime argument to AWA with `--config` or `-c` option.
The following command lines are valid examples that run AWA with configuration files:
```
$ python -m awa --config=a.json
$ python -m awa -c b.json
```
When this option is omitted, AWA tries to read `config.json` in the current directory.

### Top-level elements
The configuration file has several top-level elements that represent the root of specifications:

|Keyword   |Type|Required|Description (default)                                               |
|----------|----|--------|--------------------------------------------------------------------|
|objectives|int |no      |Number of objectives (`1`)                                          |
|cache     |str |no      |Cache file name (`"solutions.csv"`)                                 |
|parameters|list|yes     |Parameters to be optimized (see [#parameters](#parameters))         |
|workers   |list|yes     |Workers for computing objective functions (see [#workers](#workers))|
|optimizer |dict|no      |Optimizer (see [#optimizer](#optimizer))                            |

The following code is an example of the top-level elements.
The contents of `parameters`, `workers` and `optimizer` are described in the subsequent sections.
```json
{
  "objectives": 2,
  "cache": "solutions.csv",
  "parameters": [],
  "workers": [],
  "optimizer": {}
}
```

### parameters
`parameters` is a list of dictionaries, each of which contains the followings elements:

|Keyword|Type     |Required|Description (default)                           |
|-------|---------|--------|------------------------------------------------|
|name   |str      |no      |Parameter name (`"x1"`, `"x2"`, ...)            |
|type   |str      |no      |Parameter type, `"float"` or `"int"` (`"float"`)|
|min    |float/str|no      |Lower bound (`"-inf"`)                          |
|max    |float/str|no      |Upper bound (`"+inf"`)                          |

If you want to define parameters with all default settings, you can specify empty dictionaries.
The following example defines four parameters with default settings:
```json
"parameters": [
  {}, {}, {}, {}
]
```
which is equivalent to
```json
"parameters": [
  { "name": "x1", "type": "float", "min": "-inf", "max": "+inf" },
  { "name": "x2", "type": "float", "min": "-inf", "max": "+inf" },
  { "name": "x3", "type": "float", "min": "-inf", "max": "+inf" },
  { "name": "x4", "type": "float", "min": "-inf", "max": "+inf" }
]
```

### workers
`workers` is a list of dictionaries each having a single element:

|Keyword|Type     |Required|Description (default)              |
|-------|---------|--------|-----------------------------------|
|command|str      |yes     |Shell command to evaluate solutions|

An element in `workers` is responsible for evaluating objective functions.
In its `command`, it would receive a point (parameter values), compute the value of objective functions at the given point and send the value to the standard output.
A point of evaluation is passed via a shell variable `$parameters` in which parameter values are aligned in defined order in configuration element `parameters`, separated with a space.
The simplest case is shown below:
```json
"workers": [
  { "command": "./objective_functions $parameters" }
]
```
You can also use parameter names, which is useful for passing to named arguments:
```json
"workers": [
  { "command": "./objective_functions --x=$x1 --y=$x2" }
]
```

By specifying multiple elements in `workers`, AWA runs the evaluation of objective functions in parallel.
The following example runs four parallel evaluations.
```json
"workers": [
  { "command": "./objective_functions $parameters" },
  { "command": "./objective_functions $parameters" },
  { "command": "./objective_functions $parameters" },
  { "command": "./objective_functions $parameters" }
]
```

### optimizer
`optimizer` is a dictionary of configurations.

|Keyword      |Type   |Required|Description (default)                  |
|-------------|-------|--------|---------------------------------------|
|seed         |int    |no      |Random seed (random)                   |
|max_iters    |int/str|no      |Max iterations (representing iteration)|
|max_evals    |int/str|no      |Max evaluations (`"+inf"`)             |
|scalarization|str    |no      |Scalarization (`"weighted_sum"`)       |
|optimization |str    |no      |Optimization (`"cmaes"`)               |
|x0           |matrix |no      |Initial parameters (random)            |
|w0           |matrix |no      |Initial weights (Identity matrix)      |

```json
{
  "optimizer": {
    "seed": 42,
    "max_iters": 3,
    "max_evals": 64,
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

## Examples
We demonstrate complete examples of the configuration file.

### Minimum specification
```json
{
  "parameters": [
    {}
  ],
  "workers": [
    { "command": "echo $x1" }
  ]
}
```

### Full specification
```json
{
  "objectives": 2,
  "cache": "solutions.csv",
  "parameters": [
    { "name": "x1", "type": "float", "min": "-inf", "max": "+inf" },
    { "name": "x2", "type": "float", "min": "-inf", "max": "+inf" },
    { "name": "x3", "type": "float", "min": "-inf", "max": "+inf" }
  ],
  "workers": [
    { "command": "echo $x1 $2" }
  ],
  "optimizer": {
    "seed": 42,
    "max_evals": 64,
    "max_iters": 3,
    "scalarization": "weighted_sum",
    "optimization": "cmaes",
    "x0": [[1.0, 1.0, 1.0],
           [1.0, 1.0, 1.0]],
    "w0": [[1.0, 0.0],
           [0.0, 1.0]]
  }
}
```

## Cache file
The cache file records all evaluated points and their values, accompanied by IDs, timestamps and exit codes.
The default file name is `solutions.csv`.
The following table shows an example of the cache file.

|Solution ID|Evaluation Start   |Evaluation End     |Exit|x1    |x2    |...|xn    |f1    |f2    |...|fn    |
|-----------|-------------------|-------------------|----|------|------|---|------|------|------|---|------|
|1          |2017-10-10 17:22:13|2017-10-10 23:22:15|N   |1.2e-2|1.4e-3|...|0.7e-3|1.1e-3|2.5e-2|...|1.2e-2|
|2          |2017-10-10 17:22:11|2017-10-10 23:22:16|E   |0.6e-2|1.1e-3|...|1.5e-3|3.8e-3|1.7e-2|...|4.6e-2|
|...        |...                |...                |... |...   |...   |...|...   |...   |...   |...|...   |

- Each row is recorded when the evaluation has finished rather than has started
- Solution ID is a unique integer incrementing from one
- Exit is a character indicating the exit code
  - `N`: the evaluation exited normally
  - `O`: the point was out of bounds
  - `E`: an error occurred during evaluation

During optimization, this file is used as a cache: if a given point has been already evaluated and recorded in the file, then AWA's workers simply return the value in the record, avoiding duplicate objective function calls.

After optimization, this file can be seen as a result of optimization.
You can filter an optimal solution (or non-dominated solutions in multi-objective cases) from this file.
Post-optimal analysis may also be conducted on this file.

Furthermore, this file, together with a saved random seed, provides checkpointing.
By restarting AWA with the cache file and the random seed used in the previous AWA run, you can quickly go back to the state where the previous AWA finished and resume the execution.
