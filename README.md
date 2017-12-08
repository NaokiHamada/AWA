# AWA
A Python implementation of Adaptive Weighted Aggregation.

__Notice:__
This is an alpha version.
There would be many bugs and undocumented features.
APIs and I/O formats will be changed in an upcoming release version without backward compatibility.

# Requirements
To use AWA, you need a Python environment installed:
- python>=3.5

and Python packages (not necessarily installed before installing AWA):
- cma>=1.1.7
- curio>=0.4
- numpy>=1.11
- scipy>=0.18

# Install
To install AWA, type:
```
pip install git+https://github.com/NaokiHamada/AWA.git@master
```
which will install AWA and required Python packages listed above.

# Run
To run AWA, move into a directory that contains a file `config.json`.
Then type:
```
python -m awa -c config.json
```
or simply:
```
python -m awa
```

After the run, a result file `solutions.csv` will be created.

# How to write `config.json`
See [examples](./examples/) and [file format specifications](./FORMATS.md).

# License
[MIT license](./LICENSE).

# References
- [Naoki Hamada, Yuichi Nagata, Shigenobu Kobayashi, Isao Ono, Adaptive Weighted Aggregation: A Multi-Start Framework Taking Account of the Coverage of Solutions for Continuous Multi-Objective Optimization, Transaction of the Japanese Society for Evolutionary Computation, Released September 03, 2012, Online ISSN 2185-7385](https://doi.org/10.11394/tjpnsec.3.31)
- [Naoki Hamada, Yuichi Nagata, Shigenobu Kobayashi, Isao Ono, On the Stopping Criterion of Adaptive Weighted Aggregation for Multiobjective Continuous Optimization, Transaction of the Japanese Society for Evolutionary Computation, Released March 02, 2013, Online ISSN 2185-7385](https://doi.org/10.11394/tjpnsec.4.13)
