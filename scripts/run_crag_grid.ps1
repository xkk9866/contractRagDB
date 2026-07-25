# CRAG plan grid: wait for the shared retrieval views, then generate, score
# and analyse. Hosted generation is API-bound so it does not contend with the
# local Ollama job on the GPUs.
param([int]$EvidencePid = 0)

$ErrorActionPreference = "Continue"
Set-Location "y:\xkk_workspace\workspace\contractRAGDB - VLDB"

if ($EvidencePid -gt 0) {
  Write-Host "[chain] waiting for evidence pid $EvidencePid"
  try { Wait-Process -Id $EvidencePid -ErrorAction Stop } catch { }
}
Write-Host "[chain] evidence done, starting generation"

python -X utf8 scripts\experiment_plangrid_all.py --track crag --stage generate `
  --models "qwen-flash,qwen-plus,qwen-max,deepseek-v4-flash,deepseek-v3.2,glm-4.7" `
  --workers 96 --n_train 300 --n_cal 500 --n_test 600 `
  >> experiments\cggrid_generate.log 2>&1

Write-Host "[chain] generation done, scoring with the judge"
python -X utf8 scripts\experiment_plangrid_all.py --track crag --stage score `
  --models "qwen-flash,qwen-plus,qwen-max,deepseek-v4-flash,deepseek-v3.2,glm-4.7" `
  --judge qwen-max --n_train 300 --n_cal 500 --n_test 600 `
  >> experiments\cggrid_score.log 2>&1

Write-Host "[chain] scoring done, analysing"
python -X utf8 scripts\experiment_plangrid_all.py --track crag --stage analyze `
  --models "qwen-flash,qwen-plus,qwen-max,deepseek-v4-flash,deepseek-v3.2,glm-4.7" `
  --alphas "0.35,0.45,0.55,0.62,0.70" --n_train 300 --n_cal 500 --n_test 600 `
  >> experiments\cggrid_analyze.log 2>&1

Write-Host "[chain] CRAG plan grid complete"
