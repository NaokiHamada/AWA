# Reproduce results
This example shows a configuration to ensure the reproducibility of results.

Since AWA involves random numbers, different runs with the same configuration would produce different results.
By adding `seed` into a configuration file, you can fix a random seed and ensure the reproducibility of results.
```json
"optimizer": {
  "seed": 42
}
```

## Requirements on objective functions
As well as fixing a random seed, you also need a _purely functional_ objective function.
That is, given the same input, the objective function should always return the same output.
If this requirement is not satisfied, AWA with a fixed seed may produce different results.
Pay special attention to pure functionality when your objective function uses parallel computation or I/O.

## Run a Demo
Let us fix a random seed and confirm that AWA performs a reproducible run.
Type the following command:
```
$ ./run.sh
```

This script runs AWA twice and compare the results.

More specifically, it runs AWA as follows:
```
# Run the first experiment with a random seed specified
$ python -m awa -c config_1.json  # Produce results_1.csv

# Run the second experiment with the same seed
$ python -m awa -c config_2.json  # Produce results_2.csv
```

Both results will be identical (except for timestamps):
```
# Cut the timestamp columns off
cut -d, -f2,3 --complement results_1.csv > solutions_1.csv
cut -d, -f2,3 --complement results_2.csv > solutions_2.csv

# They should have the same solutions
diff solutions_1.csv solutions_2.csv  # No output means the same contents
```

