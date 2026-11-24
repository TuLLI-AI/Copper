#!/bin/bash
# Runs the Bayesian multi-answer CoT experiment for every dataset, split,
# strategy and model, in parallel. Each run calls the model, saves the traces
# and writes one result with the default setup (answer prior c, step mean).
# Runs that are already complete are skipped.
#
# To launch llama and gpt-4o concurrently (recommended for 2× wall-clock
# speedup), run TWO nohup invocations in two shells — one with
# TOGETHER_API_KEY set, one with OPENAI_API_KEY set — so their throttle
# pools are independent.
#
# Required env (one or both):
#   TOGETHER_API_KEY     for Llama on Together
#   OPENAI_API_KEY       for gpt-4o on direct OpenAI    (recommended)
#   OPENROUTER_API_KEY   for gpt-4o via OpenRouter      (alternative)
#
# IMPORTANT: if both TOGETHER_API_KEY and OPENROUTER_API_KEY are set,
# gpt-4o calls will route via OpenRouter (which can drop logprobs).
# To force OpenAI direct: `unset OPENROUTER_API_KEY`.
#
# Usage (recommended — two parallel nohups):
#   cd <repo root>
#
#   # Shell 1: llama
#   unset OPENAI_API_KEY OPENROUTER_API_KEY
#   export TOGETHER_API_KEY=...
#   MODELS="meta-llama/Llama-3.3-70B-Instruct-Turbo" \
#     nohup bash launchers/shell/run_all_v3.sh > logs/all_v3_llama.log 2>&1 &
#   disown
#
#   # Shell 2: gpt-4o
#   unset TOGETHER_API_KEY OPENROUTER_API_KEY
#   export OPENAI_API_KEY=sk-...
#   MODELS="openai/gpt-4o" \
#     nohup bash launchers/shell/run_all_v3.sh > logs/all_v3_gpt4o.log 2>&1 &
#   disown
#
# Or single process (sequential — llama queue first, then gpt-4o):
#   export TOGETHER_API_KEY=... OPENAI_API_KEY=...; unset OPENROUTER_API_KEY
#   nohup bash launchers/shell/run_all_v3.sh > logs/all_v3.log 2>&1 &
#
# Knobs (env overrides):
#   DATA_ROOT     default bbh_data
#   OUT_ROOT      default results/all_v3
#   MODELS        default both — set to one model to scope a single launch
#   STRATEGIES    default "combined separate"
#   SPLITS        default "val test"
#   TASKS         default = autodiscover under DATA_ROOT
#   MAX_PARALLEL  default 16   (concurrent worker cap PER nohup process —
#                  with two parallel processes you get 32 total workers)
#   SEED          default 42
#   NUM_RETRIES   default 5

set -u

# --- Required env -----------------------------------------------------------
if [ -z "${TOGETHER_API_KEY:-}"   ] && \
   [ -z "${OPENAI_API_KEY:-}"     ] && \
   [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "ERROR: set TOGETHER_API_KEY (llama), OPENAI_API_KEY (gpt-4o), " \
       "or OPENROUTER_API_KEY before launch." >&2
  exit 1
fi
[ -n "${TOGETHER_API_KEY:-}"   ] && export TOGETHER_API_KEY
[ -n "${OPENAI_API_KEY:-}"     ] && export OPENAI_API_KEY
[ -n "${OPENROUTER_API_KEY:-}" ] && export OPENROUTER_API_KEY

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT" || { echo "ERROR: cannot cd to $REPO_ROOT"; exit 1; }
echo "Repo root: $REPO_ROOT"

DATA_ROOT="${DATA_ROOT:-bbh_data}"
OUT_ROOT="${OUT_ROOT:-results/all_v3}"
MODELS="${MODELS:-meta-llama/Llama-3.3-70B-Instruct-Turbo openai/gpt-4o}"
STRATEGIES="${STRATEGIES:-combined separate}"
SPLITS="${SPLITS:-val test}"
MAX_PARALLEL="${MAX_PARALLEL:-16}"
SEED="${SEED:-42}"
NUM_RETRIES="${NUM_RETRIES:-5}"

if [ ! -d "$DATA_ROOT" ]; then
  echo "ERROR: DATA_ROOT not found: $DATA_ROOT" >&2
  exit 1
fi
mkdir -p logs "$OUT_ROOT"

# --- Helpers ---------------------------------------------------------------
ds_len() {
  python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$1" 2>/dev/null
}

results_count() {
  python3 -c "
import json,sys
try:
    print(len(json.load(open(sys.argv[1]))['results']))
except Exception:
    print(0)
" "$1" 2>/dev/null
}

throttle() {
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$MAX_PARALLEL" ]; do
    sleep 2
  done
}

# --- Autodiscover tasks ----------------------------------------------------
if [ -z "${TASKS:-}" ]; then
  TASKS=""
  for d in "$DATA_ROOT"/*/; do
    [ -d "$d" ] || continue
    t="$(basename "$d")"
    has_any=0
    for split in $SPLITS; do
      [ -f "$d/multi_cot_${split}.json" ] && has_any=1
    done
    [ "$has_any" = "1" ] && TASKS="$TASKS $t"
  done
  TASKS="${TASKS# }"
fi
[ -z "$TASKS" ] && { echo "ERROR: no tasks found under $DATA_ROOT" >&2; exit 1; }

echo "MODELS      : $MODELS"
echo "DATA_ROOT   : $DATA_ROOT"
echo "OUT_ROOT    : $OUT_ROOT"
echo "STRATEGIES  : $STRATEGIES"
echo "SPLITS      : $SPLITS"
echo "MAX_PARALLEL: $MAX_PARALLEL"
echo "TASKS       : $TASKS"
echo

# --- Launch ----------------------------------------------------------------
echo "============================================================"
echo " Fresh base runs — combined + separate, both models"
echo "============================================================"

started=0
skipped=0
total=0

for MODEL in $MODELS; do
  MODEL_SLUG="${MODEL//\//_}"
  echo
  echo "--- model: $MODEL  (slug: $MODEL_SLUG) ---"
  for T in $TASKS; do
    for SPLIT in $SPLITS; do
      DATASET="$DATA_ROOT/$T/multi_cot_${SPLIT}.json"
      if [ ! -f "$DATASET" ]; then
        echo "[skip] $DATASET missing"
        continue
      fi
      N="$(ds_len "$DATASET")"
      if [ -z "$N" ] || [ "$N" -lt 1 ]; then
        echo "[skip] $DATASET empty"
        continue
      fi

      for STRAT in $STRATEGIES; do
        total=$((total + 1))
        OUT="$OUT_ROOT/${MODEL_SLUG}/${T}_${SPLIT}_${STRAT}"
        LOG="logs/all_v3_${MODEL_SLUG}_${T}_${SPLIT}_${STRAT}.log"

        if [ -f "$OUT/results.json" ]; then
          have="$(results_count "$OUT/results.json")"
          if [ -n "$have" ] && [ "$have" -ge "$N" ]; then
            echo "[skip] $MODEL_SLUG/$T/$SPLIT/$STRAT — already $have/$N rows"
            skipped=$((skipped + 1))
            continue
          fi
        fi

        throttle
        echo "[launch] $MODEL_SLUG/$T/$SPLIT/$STRAT  (n=$N)  -> $OUT"
        python experiments/runners/run_BBH_multi_cot.py \
          --model_name "$MODEL" \
          --dataset_path "$DATASET" \
          --output_dir "$OUT" \
          --cot_strategy "$STRAT" \
          --pool_mode logitpool \
          --answer_aggregation_fn c \
          --statement_aggregation_fn mean \
          --step_dependency_mode flat \
          --seed "$SEED" \
          --num_retries "$NUM_RETRIES" \
          > "$LOG" 2>&1 &
        started=$((started + 1))
      done
    done
  done
done

echo
echo "started=$started  skipped=$skipped  total=$total — waiting for workers..."
wait
echo "ALL RUNS DONE"

# --- Summary ---------------------------------------------------------------
echo
echo "============================================================"
echo " Summary"
echo "============================================================"
for MODEL in $MODELS; do
  MODEL_SLUG="${MODEL//\//_}"
  echo
  echo "--- $MODEL ---"
  for T in $TASKS; do
    for SPLIT in $SPLITS; do
      DATASET="$DATA_ROOT/$T/multi_cot_${SPLIT}.json"
      [ -f "$DATASET" ] || continue
      N="$(ds_len "$DATASET")"
      for STRAT in $STRATEGIES; do
        OUT="$OUT_ROOT/${MODEL_SLUG}/${T}_${SPLIT}_${STRAT}"
        have=0
        [ -f "$OUT/results.json" ] && have="$(results_count "$OUT/results.json")"
        printf '  %-42s %-5s %-9s %4s/%-4s\n' "$T" "$SPLIT" "$STRAT" "$have" "$N"
      done
    done
  done
done
