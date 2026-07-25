# VLDB 2027 submission package

## Final manuscript

- `submission_files/final/VLDB2027_Manuscript.pdf`
- `main.tex` and `sec_*.tex`: editable manuscript source
- `acmart.cls`, `pvldb.sty`, `ACM-Reference-Format.bst`: official
  PVLDB Volume 20 template files, kept unmodified

The compiled PDF has 10 pages. References begin on page 9 and continue on
page 10; all non-reference content fits within the 12-page Regular
Research Paper limit. The PDF is single blind and contains all author and
affiliation information.

## Supporting files

- `CMT_FILLING_GUIDE_ZH.md`: exact CMT field values and policy checks
- `literature_review.md`: official requirements and accepted-paper study
- `ARTIFACT_README.md`: reviewer-facing artifact instructions to publish
  as the public repository's root README
- `literature/`: seven downloaded PVLDB papers used for positioning
- `official_template_main.tex` and `OFFICIAL_TEMPLATE_README.md`: untouched
  official sample and instructions

## Compile

From this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

On this Windows network workspace, compile through the mapped `Y:` drive;
pdfTeX may fail when writing directly through a UNC path.

## Required author checks before upload

1. Confirm no substantially identical KBS, ESWA, Neurocomputing, or other
   version is currently under review.
2. Confirm this work has not been rejected from the PVLDB Research Track
   during the preceding 12 months.
3. Publish the artifact README and verify the GitHub URL without login.
4. Identify and verify the nominated author reviewer.
5. Have all authors enter their complete CMT conflicts and approve the
   final author order.

