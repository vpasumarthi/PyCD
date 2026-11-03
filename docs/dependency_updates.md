---
summary: "Python compatibility and validation for the certifi and pylint dependency updates"
read_when:
  - "updating Python dependencies or CI"
---

# Install and validate dependency updates

Python 3.6 is no longer supported. Package metadata requires Python 3.7 or
newer. Lint tooling requires Python 3.10 or newer; this does not establish
runtime compatibility for every Python version accepted by package metadata.

Install the package and lint tooling in a Python 3.10+ virtual environment:

```sh
python -m pip install -e . pytest -r requirements-lint.txt
python -m pip check
python -m pytest -q PyCD/tests
python -m pylint --errors-only PyCD
```

`requirements-lint.txt` owns the pylint, astroid, isort and mccabe pins.
The legacy `requirements.txt` includes that file and consequently requires
Python 3.10+ as a combined development environment.

The GitHub Actions workflow validates package installation, the exact certifi
pin from `requirements.txt`, lint dependency installation, the CA bundle,
and the test suite on Python 3.10 and 3.11. It does not enforce a clean pylint
report: pylint currently reports errors in existing code. Its version check
is an installation smoke test only.

The full legacy requirements are not validated by that workflow. A baseline
resolution attempt on November 3, 2026 failed while building
`pprofile==1.11.0` because its setup uses the unsupported `use_2to3` option.
That failure predates these updates and remains unresolved.
