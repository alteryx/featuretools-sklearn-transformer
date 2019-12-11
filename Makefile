.PHONY: clean
clean:
	find . -name '*.pyo' -delete
	find . -name '*.pyc' -delete
	find . -name __pycache__ -delete
	find . -name '*~' -delete

.PHONY: lint-tests
lint-tests:
	flake8 featuretools_sklearn_transformer && isort --check-only --recursive featuretools_sklearn_transformer

.PHONY: lint-fix
lint-fix:
	autopep8 --in-place --recursive --max-line-length=100 --select="E225,E303,E302,E203,E128,E231,E251,E271,E127,E126,E301,W291,W293,E226,E306,E221" featuretools_sklearn_transformer
	isort --recursive featuretools_sklearn_transformer

.PHONY: unit-tests
unit-tests:
	coverage erase
	pytest --cov=featuretools_sklearn_transformer --cache-clear --show-capture=stderr -vv

.PHONY: installdeps
installdeps:
	pip install --upgrade pip -q
	pip install -e . -q
	pip install -r test-requirements.txt -q

.PHONY: entry-point-test
entry-point-test:py
	cd ~ && python -c "from featuretools.wrappers.sklearn import DFSTransformer"
