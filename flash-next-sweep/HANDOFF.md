# Handoff: Qwen3.8-Flash-Next Block Swap data, and the Block Swap off run

Data PR (fork, do not merge): https://github.com/LeoBorcherding/unsloth-zoo/pull/44
Branch `data/flash-next-sweep-results`, head `d7200901`. Everything below is as of 2026-11-13.

## 1. For the teammate: time Block Swap off on AWS

**Why.** Block Swap trains the 180B Flash-Next on one 96 GB GPU. The open question is efficiency:
is one GPU at 6 to 9 s/step cheaper per token than the same LoRA with Block Swap off across
several GPUs? The Block Swap side is measured. The off side can't run on one card (it OOMs while
loading, "tried to allocate 93.77 GiB"), and Colab has no multi-GPU VMs, so it needs AWS.

**Box.** It needs about 373 GiB of VRAM (Unsloth Studio's own estimate, `required_gb=372.632`) plus
~180 GiB of host RAM and ~250 GB of disk for the 173 GB checkpoint.

| instance | GPUs | fits? | notes |
|---|---|---|---|
| G7e with 8x RTX PRO 6000 Blackwell (if the account has it) | 8 x 96 GB | yes | **best**: same card as every Block Swap run, so step times compare directly |
| p5.48xlarge | 8 x H100 80 GB | yes | fine; compare by GPU-hours and dollars, not raw step time |
| anything with 4 x 96 GB or less | | no | 384 GB minus multi-GPU overhead is under 373 GiB |

About 1 hour covers it: ~15 to 25 min to download and load, then the steps.

**Code, same as the Block Swap runs.**
- unsloth: `LeoBorcherding/unsloth`, branch `studio/online-tok-host-ram` (e3e8fd3bec, upstream main 211f0ee22 + the fork fix)
- unsloth-zoo: `LeoBorcherding/unsloth-zoo` at `989ac3f4df75e78dd6c5b4038f796321f016586b` (branch `data/flash-next-stack`)

```bash
git clone https://github.com/LeoBorcherding/unsloth && cd unsloth
git checkout studio/online-tok-host-ram
./studio/setup.sh --local
# then put the zoo stack into the env Unsloth Studio trains with:
python -m pip install --no-deps --force-reinstall "unsloth-zoo @ git+https://github.com/LeoBorcherding/unsloth-zoo@989ac3f4df75e78dd6c5b4038f796321f016586b"
```

**Run it in Unsloth Studio** (Train page, or `POST /api/train/start`), with every visible GPU:

| setting | value |
|---|---|
| model | `unsloth/Qwen3.8-Flash-Next-FP8` |
| method | LoRA, 16-bit (load_in_4bit off) |
| LoRA r / alpha / dropout | 16 / 16 / 0 |
| target modules | q, k, v, o, gate, up, down (attention + MLP on) |
| dataset | `mlabonne/FineTome-100k`, train split, rows 0 to 400, packing on |
| context | 1024 |
| batch / grad accum | 1 / 1 (then a second run at batch 2 if there's time) |
| max steps | 20 (10 is the minimum; skip saving) |
| optimizer | paged_adamw_8bit, lr 2e-4, linear, warmup 1 |
| gradient checkpointing | unsloth |
| seed | 3407 |
| **Block Swap** | **off** (Count 0) |

For API runs, the exact JSON body is `payload()` in `flash-next-sweep/fns_job.py` with `offload_layers: 0`.

**Send back:**
1. Instance type, GPU model and count, and the hourly price.
2. Step times for steps 3 onward (the median is the number that matters), plus total run time and
   the time from start to step 1.
3. Peak VRAM per GPU (`nvidia-smi --query-gpu=memory.used --format=csv -l 5` in another terminal).
4. Did it train, or the error if not? Unsloth Studio spreads the model over all visible GPUs when it
   doesn't fit one; nobody has run that path for this model yet, so a failure there is itself a
   finding.
5. A screenshot of the Train page mid-run.

**What we'll compare** (Block Swap side, measured on a Colab G4, RTX PRO 6000 96 GB):

| setup | GPUs | s/step | tokens/s | GPU-hours per 1M tokens |
|---|---|---|---|---|
| Count 32 | 1 | 6.05 | 169 | 1.64 |
| Count 36, batch 2 | 1 | 9.07 | 226 | 1.23 |
| Block Swap off | 8 | ? | ? | ? |

For 8 GPUs to beat Count 36 batch 2 per GPU-hour, they'd need more than 8 x 226 = 1,808 tokens/s, i.e.
under ~0.57 s/step at ctx 1024 batch 1. My guess is they won't, because a model split across cards runs
them one after another (about 1 of 8 busy at a time). That guess is what this run tests.

## 2. State of the work (for whoever picks this up)

**Done and posted (fork only, no upstream posts):**
- PR #44 body: TL;DR (373 GiB vs 75 to 83 GiB, time cost, total run time), hero chart, Block Swap
  explainer, Count 32 GIF, results table, 10 charts, the fork bug, details folded at the bottom.
- GIF comment: https://github.com/LeoBorcherding/unsloth-zoo/pull/44#issuecomment-6860115104
- Charts are made by `flash-next-sweep/make_charts.py` (source of truth:
  `~/git_dev/unsloth-audit-2026-10/reviews/block-swap/colab/fns/charts/real/`; copy into the branch,
  commit as LeoBorcherding, push, bump the raw SHA in the body).

**Open:**
- **Placeholders for Block Swap off are in now** ("not run yet"): chart 10-efficiency, the hero's time panel, an off row in the body's table, and the TL;DR. To fill them, save the run as `runs/aws-off-ctx1024.json` (same shape as the other runs, `status`, `median_step_s`, `cfg` with `ctx`, `bs` and `gpus`). Chart 10 picks it up on its own; the hero's time row and the body table are manual edits.
- **Efficiency chart** waits on the run above. Once it lands, add a thick chart: GPU-hours per 1M
  tokens, dollars per 1M tokens, wall-clock for 1,000 steps, off vs Count 32 vs Count 36 batch 2.
  Then add an off row to the table and a line to the TL;DR.
- **Count 32 waits ~9.5 min before step 1** in all four Count 32 runs; Count 36 waits under 1 min.
  Not explained. Leading guess: pinning 12.4 GiB of host memory (Count 36 pins 2.5) under host-RAM
  pressure. Untested. It's what makes Count 36 the faster choice under ~180 steps.
- **Fork-bug fix branch** `studio/online-tok-host-ram` is pushed to LeoBorcherding/unsloth, with no PR
  anywhere. Its draft PR body wrongly blames online tokenization; the cause is the eager path's
  `dataset.map(num_proc=N)` forking the trainer. Fix the body before any upstream PR (needs Leo's OK).
- Two unconfirmed Unsloth Studio findings, no issues filed: `finetune_mlp_modules=False` is ignored for
  MoE (same 2.78B trainable params), and `load_in_4bit=True` silently trains 16-bit on Flash-Next.
- The Block Swap off and ctx 4096 GIFs, and the last VM's logs, were lost: Colab reclaimed
  `fl-g4-fnsgif2-45668` right after its last run. The off result in the charts comes from the run's
  final status event (`runs/gif-off-ctx1024.json` says so).

**Colab:** no VMs live; fleet back in night mode. The sweep used ~53 CU and the GIF VMs about 20 more.

**Upstream status:** every Flash-Next PR is merged (unsloth#15159, #15632, #15633; zoo#2070, #2268,
#1619, #2269). Nothing from this work has been posted upstream.
