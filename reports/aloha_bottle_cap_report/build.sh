#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

xelatex -interaction=nonstopmode -halt-on-error aloha_bottle_cap_report.tex
bibtex aloha_bottle_cap_report
xelatex -interaction=nonstopmode -halt-on-error aloha_bottle_cap_report.tex
xelatex -interaction=nonstopmode -halt-on-error aloha_bottle_cap_report.tex

pdfinfo aloha_bottle_cap_report.pdf > artifacts/pdfinfo.txt
cp aloha_bottle_cap_report.log artifacts/latex_build.log
{
  grep -Ei 'undefined references|undefined citations|multiply defined|overfull|underfull|not found|error' aloha_bottle_cap_report.log || true
} > artifacts/latex_issues.txt

sed -n '1,20p' artifacts/pdfinfo.txt
