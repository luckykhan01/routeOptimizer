PYTHON ?= python
PIP ?= pip

.PHONY: setup generate train serve demo test clean full_demo

setup:
	$(PIP) install --upgrade pip
	$(PIP) install -e .[dev]

generate:
	$(PYTHON) src/data/generate_data.py

train:
	$(PYTHON) src/models/train_eta.py

serve:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

demo:
	streamlit run src/ui/app.py --server.port 8501

test:
	pytest

clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache','mlruns','__pycache__','data/processed','models']]; [pathlib.Path(p).mkdir(parents=True, exist_ok=True) for p in ['data/processed','models']]"

full_demo:
	scripts\\run_demo.bat
