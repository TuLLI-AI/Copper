# Copper

Code for the Neurips 2026 paper *CoTs as Probabilistic Programs: A Programmatic View of
Thinking Step-by-Step in Language Models*.

Copper is a small discrete probabilistic programming language for
chain-of-thought (CoT). This repo turns a model's chain-of-thought into a probalistic
program and runs exact inference over the answer candidates.

```
cot_prog/            the library: DSL, parser, compiler, prompts, axes.py (parameter names)
console.py           interactive per-example UI (Streamlit)
experiments/
  runners/           experiments: run_BBH_multi_cot.py (Bayesian), run_vanilla_cot.py (Likelihood)
  baselines/         experiments: reasoning_baselines.py (Direct / CoT / Combined CoT / ToT)
  data_prep/         data: build bbh_data/<task>/{multi_cot,baseline}_{val,test}.json
  reports/           visualization: per-example HTML reports of the program
launchers/shell/     large-scale parallel runs: run_all_v3.sh (all tasks and models)
```

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # direct deps, pinned to tested versions
# or, for the exact environment this code was verified in:
# pip install -r requirements-lock.txt   # full transitive closure (Python 3.12.13)
```

API keys come from the environment (they are not loaded from `.env`):

| model | key | model name |
|---|---|---|
| gpt-4o | `OPENROUTER_API_KEY` (or `OPENAI_API_KEY`) | `openai/gpt-4o` |
| Llama-3.3-70B, Qwen-2.5-7B | `TOGETHER_API_KEY` | `meta-llama/Llama-3.3-70B-Instruct-Turbo`, `Qwen/Qwen2.5-7B-Instruct-Turbo` |

Llama and Qwen need Together: OpenRouter's providers for those models do not
return full per-token logprobs, which the step weights require.

## Data

BoolQ and StrategyQA are included under `bbh_data/boolq/` and
`bbh_data/strategyqa/`, so the examples below run as they are. The three BBH
tasks are not included. Build them:

```bash
python experiments/data_prep/prep_bbh.py --task date_understanding --out_dir bbh_data
python experiments/data_prep/prep_bbh.py --task geometric_shapes --out_dir bbh_data
python experiments/data_prep/prep_bbh.py --task salient_translation_error_detection --out_dir bbh_data
```

(`prep_boolq.py` and `prep_strategyqa.py` rebuild the two included datasets.)

## Parameters

| flag | values | default |
|---|---|---|
| `--cot_strategy` | `combined` (#calls=1), `separate` (#calls=N) | `combined` |
| `--answer_aggregation_fn` | `c` (verbal), `p` (logit), `cal`, `mean`, `max`, `min` | `c` |
| `--statement_aggregation_fn` | `mean`, `max`, `min`, `pool`, `p`, `c` | `mean` |
| `--pool_mode` | `logitpool` | `logitpool` |
| `--step_dependency_mode` | `flat` | `flat` |

## 1. Interactive UI

```bash
streamlit run console.py
```

## 2. Running a single experiment

**Likelihood program** (one standard CoT call per example):

```bash
python experiments/runners/run_vanilla_cot.py --model_name openai/gpt-4o \
  --dataset_path bbh_data/boolq/multi_cot_test.json \
  --output_dir results/vanilla/openai_gpt-4o/boolq_test
```

Its `summary.json` accuracy divides by the full dataset size, so a partial run
(`--num_samples`) reports a low number.

**Bayesian program: one base run** (calls the LLM, saves traces and a result for one configuration):

```bash
python experiments/runners/run_BBH_multi_cot.py \
  --model_name openai/gpt-4o \
  --dataset_path bbh_data/boolq/multi_cot_test.json \
  --output_dir results/all_v3/openai_gpt-4o/boolq_test_combined \
  --cot_strategy combined --answer_aggregation_fn c --statement_aggregation_fn mean
```

**Re-run on saved traces** (no LLM calls; change the two aggregation flags to try other settings):

```bash
python experiments/runners/run_BBH_multi_cot.py \
  --reuse_traces_from results/all_v3/openai_gpt-4o/boolq_test_combined \
  --output_dir results/all_v3/openai_gpt-4o/boolq_test_combined_rerun \
  --pool_mode logitpool --statement_aggregation_fn mean \
  --answer_aggregation_fn cal --step_dependency_mode flat
```

Every run writes `summary.json` with two accuracies, both in percent:
`accuracy` divides by the dataset size (`n_total`: the sampled examples, or in
reuse mode all source checkpoints), so an errored or missing example counts as
wrong; this is the same denominator the Likelihood runner uses, and the one
to report. `accuracy_on_predicted` divides by the examples that produced a
prediction (`n_predicted`) and is only a diagnostic: it is inflated whenever
examples error, and error counts differ between configurations.

**Prompting baselines** (Direct / CoT / Combined CoT / Tree-of-Thoughts):

```bash
python experiments/baselines/reasoning_baselines.py \
  --input bbh_data/boolq/baseline_test.json \
  --output_dir results/baseline/openai_gpt-4o/boolq_test_direct \
  --model openai/gpt-4o --baselines direct
```

## 3. Large-scale parallel runs

**All tasks and models.** The launcher starts one run per task, split and
model, in parallel (`MAX_PARALLEL`, default 16), with the default setup:

```bash
MODELS="openai/gpt-4o" STRATEGIES="combined" bash launchers/shell/run_all_v3.sh
```

It writes to `results/all_v3/<model>/<task>_<split>_<strategy>/` and skips runs
that are already complete. Each run's accuracy is in its `summary.json`.

## 4. Visualization of the program: HTML reports

Per-example pages showing the chain-of-thought, the Copper program built from
it, and the inference result:

```bash
python experiments/reports/json_to_html.py <run_dir>/ckpts/ -o <run_dir>/html/   # Bayesian runs
python experiments/reports/vanilla_to_html.py <run_dir>                          # Likelihood runs
python experiments/reports/baselines_to_html.py <baseline_output_dir>            # baselines
```

## Citation

If you find the paper useful, please cite:

```bibtex
@inproceedings{
richardson2026cots,
title={CoTs as Probabilistic Programs: A Programmatic View of Thinking Step-by-Step in Language Models},
author={Kyle Richardson and Yu Feng and Poorva Garg and Junyan Cheng and Guy Van den Broeck and Dan Roth},
booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
year={2026},
url={https://openreview.net/forum?id=YEVrT8IA3J}
}
```

## License

The code is released under the Apache License 2.0 (see `LICENSE`). The BoolQ and
StrategyQA files under `bbh_data/` keep the licenses of their original datasets.
