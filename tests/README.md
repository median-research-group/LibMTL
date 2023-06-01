## Test

```shell
pip install coverage
coverage erase
coverage run -m --parallel-mode --source ./LibMTL pytest tests/test_nyu.py
coverage run -m --parallel-mode --source ./LibMTL pytest tests/test_office31.py
coverage run -m --parallel-mode --source ./LibMTL pytest tests/test_office_home.py
coverage run -m --parallel-mode --source ./LibMTL pytest tests/test_qm9.py
coverage run -m --parallel-mode --source ./LibMTL pytest tests/test_pawsx.py
coverage combine
coverage report
rm -rf tests/htmlcov
coverage html -d tests/htmlcov
pip install coverage-badge
rm tests/coverage.svg
coverage-badge -o tests/coverage.svg
```
