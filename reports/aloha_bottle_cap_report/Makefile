.PHONY: all figures compile verify clean

all: figures compile

figures:
	../../.venv/bin/python scripts/generate_annual_report_figures.py

compile:
	bash build.sh

verify:
	../../.venv/bin/python scripts/verify_report.py

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot
