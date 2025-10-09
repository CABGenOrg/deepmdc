SHELL := /bin/bash

.PHONY: setup
setup:
	@\
	python3 manage.py setup; \
	source .venv/bin/activate; \
	python3 manage.py install;

.PHONY: run
run:
	@\
	source ./.venv/bin/activate; \
	python3 cabgen_hopfield_main.py \
