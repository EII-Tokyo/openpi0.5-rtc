#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
build_log="artifacts/latex_build.log"
: > "$build_log"
xelatex -interaction=nonstopmode -halt-on-error aloha_bottle_cap_report_ja.tex 2>&1 | tee -a "$build_log"
bibtex aloha_bottle_cap_report_ja 2>&1 | tee -a "$build_log"
xelatex -interaction=nonstopmode -halt-on-error aloha_bottle_cap_report_ja.tex 2>&1 | tee -a "$build_log"
xelatex -interaction=nonstopmode -halt-on-error aloha_bottle_cap_report_ja.tex 2>&1 | tee -a "$build_log"
pdfinfo aloha_bottle_cap_report_ja.pdf > artifacts/pdfinfo.txt
