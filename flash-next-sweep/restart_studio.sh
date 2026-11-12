#!/bin/bash
# restart_studio.sh PLAN: stop the sweep and Unsloth Studio, start Studio with online tokenization off, rerun the plan.
PLAN=$1
pkill -9 -f 'sbx/fns_job'
for p in $(pgrep -f 'python3 /content/sbx/serve.py'); do pkill -9 -P $p; kill -9 $p; done
pkill -9 -f 'from multiprocessing'
sleep 8
cd /content/fns/runs && for f in nt-*.json; do [ -f "$f" ] && mv "$f" "${f%.json}-mapfork.json"; done
cd /content
UNSLOTH_STUDIO_ONLINE_TOKENIZATION=0 UNSLOTH_DATASET_NUM_PROC=1 setsid nohup python3 /content/sbx/serve.py >> /content/sbx/serve.log 2>&1 < /dev/null &
sleep 45
until curl -sf localhost:8888/api/health >/dev/null; do sleep 5; done
for p in $(pgrep -f 'python3 /content/sbx/serve.py'); do echo "serve $p $(tr '\0' '\n' < /proc/$p/environ | grep -E "ONLINE|NUM_PROC")"; done
PLAN=$PLAN PYTHONUNBUFFERED=1 exec python3 /content/sbx/fns_job4.py >> /content/fns/job.log 2>&1
