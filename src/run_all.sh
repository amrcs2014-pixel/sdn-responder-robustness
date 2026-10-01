#!/bin/bash
# Paper B grids (deterministic, resumable). Needs data/, models/qwen1.5b and models/adapters/sft.pt
cd "$(dirname "$0")"
export PYTHONHASHSEED=0
W=${W:-14}
serve() { python llm_server.py $1 > ../results/llm_server.log 2>&1 & until grep -q "ready" ../results/llm_server.log; do sleep 5; done; }
stop() { pkill -f llm_server.py 2>/dev/null || powershell -Command "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -like '*llm_server*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"; sleep 8; }
NL=none,portdrop,template,vtemplate,vtemplate_D
for g in B_main B_none B_cost B_stale B_abl_t A_main; do python run_exp.py $g --only $NL,vt_prov,vt_thr,vt_cost,vt_hyst --workers 10; done
serve "--merge sft=../models/adapters/sft.pt"
python run_exp.py B_main --only llm,verimit,verimit_D --workers $W --maxseed 3
python run_exp.py B_cost --only verimit_D --workers $W
python run_exp.py B_abl --workers $W
python run_exp.py A_main --only verimit --workers $W
stop
echo ALLDONE
