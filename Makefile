.PHONY: setup weights test lint

setup:
	python -m scripts.setup_env

weights:
	python -m scripts.download_weights

test:
	python -m unittest discover -s tests -p "test_*.py" -v

lint:
	python -m compileall -q schema fixtures ai_core backend eval scripts tests
