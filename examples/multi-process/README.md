# Run AWA with Multiple Worker Tasks
This example shows a configuration to parallelize objective function evaluations by running multiple worker tasks that invoke evaluation processes on a local host.

To parallelize evaluations, add multiple elements of `wokers` in `config.json` as follows:
```json
{
  "workers": [
    { "command": "./sphere_2D.sh $parameters" },
    { "command": "./sphere_2D.sh $parameters" },
    { "command": "./sphere_2D.sh $parameters" },
    { "command": "./sphere_2D.sh $parameters" }
  ]
}
```
which runs four workers asynchronously.

# Run a Demo
```
./run.sh
```

