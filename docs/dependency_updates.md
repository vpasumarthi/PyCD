---
summary: "Runtime dependencies, development installation, and optional tools"
read_when:
  - "updating Python dependencies or CI"
  - "installing the development environment"
---

# Install the development environment

Package metadata requires Python 3.7 or newer. The development requirements
include lint tooling that requires Python 3.10 or newer. CI validates Python
3.10 and 3.11; this does not establish runtime compatibility for every version
accepted by package metadata.

From the repository root, in a fresh Python 3.10+ environment:

```sh
python -m pip install -r requirements.txt
python -m pip check
python -m pytest -q PyCD/tests
```

`requirements.txt` installs PyCD in editable mode and obtains its runtime
requirements (NumPy, SciPy, Matplotlib, PyYAML and h5py) from `setup.py`.
It also installs test tooling and includes `requirements-lint.txt`, which owns
the pylint, astroid, isort and mccabe pins. The existing NumPy minimum and
reviewed certifi and Pygments pins are retained.

Runtime-only users can still install with `python -m pip install -e .`.
The development requirements are not a complete version lock: dependency
versions can differ by Python version and installation date.

# Keep optional tools separate

IPython, pprofile and line-profiler are not used by tracked application code,
examples or scripts and are not needed for the test suite. They are no longer
installed by the general requirements. Install them separately in an optional
tooling environment if needed; no profiling workflow is configured here.
Dependencies of the remaining packages are resolved from their package
metadata instead of the historical environment's transitive pins.

The old requirements had independent blockers beyond pprofile's unsupported
`use_2to3` build option: incompatible IPython/prompt-toolkit/traitlets pins,
and source-build failures for PyYAML 5.4, SciPy 1.0.1 and line-profiler 2.1.2
in the checked Python 3.11 macOS ARM environment. Removing only pprofile did
not resolve the original requirements.

# Validate dependency updates

GitHub Actions installs the entire development requirements file, checks
installed dependency consistency, loads the certifi CA bundle, checks the
pylint executable, and runs the test suite on Python 3.10 and 3.11.

To inspect remaining code lint findings:

```sh
python -m pylint --errors-only PyCD
```

The pylint version check in CI is an installation smoke test, not enforcement
of a clean lint report. Remaining source-code lint findings are separate from
dependency installation cleanup.
