## Test

```shell
coverage erase
coverage run -m --parallel-mode pytest test_nyu.py
coverage run -m --parallel-mode pytest test_office31.py
coverage run -m --parallel-mode pytest test_office_home.py
coverage run -m --parallel-mode pytest test_qm9.py
coverage run -m --parallel-mode pytest test_pawsx.py
coverage combine
coverage report
coverage-badge -o coverage.svg
```


