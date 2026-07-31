LOCAL_PYTHON := ../../.venv/bin/python
VERIFIED_PYTHON := /home/eii/project/openpi0.5-rtc-reward-learning/.venv/bin/python
PYTHON ?= $(shell if [ -x "$(LOCAL_PYTHON)" ]; then printf '%s' "$(LOCAL_PYTHON)"; else printf '%s' "$(VERIFIED_PYTHON)"; fi)

.PHONY: figures pdf verify clean

figures:
	$(PYTHON) scripts/generate_japanese_figures.py

pdf:
	./build.sh

verify:
	$(PYTHON) scripts/verify_japanese_report.py

clean:
	$(RM) *.aux *.bbl *.blg *.log *.lof *.lot *.out *.toc
