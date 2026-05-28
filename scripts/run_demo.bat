@echo off
setlocal

echo [1/4] Installing dependencies...
pip install -e .[dev]
if errorlevel 1 exit /b 1

echo [2/4] Generating synthetic data...
python src/data/generate_data.py
if errorlevel 1 exit /b 1

echo [3/4] Training ETA model...
python src/models/train_eta.py --config configs/train.yaml
if errorlevel 1 exit /b 1

echo [4/4] Running tests...
pytest -q
if errorlevel 1 exit /b 1

echo Demo pipeline completed successfully.
endlocal
