# Pipeline Integrity Monitor — Makefile
# On Windows Git Bash: use `mingw32-make <target>`
# On Mac/Linux:        use `make <target>`

PYTHON = python
VENV   = .venv
PIP    = $(VENV)/Scripts/pip
UV     = uvicorn

.PHONY: setup download prepare train evaluate api dashboard verify git clean

setup:
	python -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "\n✓ Setup complete. Activate with: source .venv/Scripts/activate"

download:
	$(PYTHON) scripts/download_data.py

download-synthetic:
	$(PYTHON) scripts/download_data.py --synthetic --n-per-class 300

prepare:
	$(PYTHON) scripts/prepare_dataset.py

train-resnet:
	$(PYTHON) scripts/train.py --model resnet18 --epochs 20

train-efficientnet:
	$(PYTHON) scripts/train.py --model efficientnet_b0 --epochs 20

train:
	$(PYTHON) scripts/train.py --model resnet18 --epochs 20

evaluate:
	$(PYTHON) scripts/evaluate.py --model models/best_model.pth

api:
	$(UV) api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run dashboard/app.py

verify:
	$(PYTHON) scripts/verify.py

verify-api:
	$(PYTHON) scripts/verify.py --api

git:
	git status
	@echo "---"
	git add .
	git commit -m "checkpoint: $(shell date +%Y-%m-%d)"
	git push
	git log --oneline -5

clean:
	rm -rf data/processed outputs models/checkpoints mlruns
	@echo "Cleaned processed data, outputs, and model checkpoints."
	@echo "Run 'make prepare && make train' to rebuild."