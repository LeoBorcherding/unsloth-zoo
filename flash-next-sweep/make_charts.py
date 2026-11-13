"""Real Flash-Next sweep charts from ../../pulled/*/runs/*.json (Unsloth Studio training API on Colab G4)."""
import glob, json, os, sys
sys.path.insert(0, os.path.expanduser('~/.claude/skills/thick-charts'))
import thick_charts as tc
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

H = os.path.dirname(os.path.abspath(__file__))
R = {}
def own_samples(d):
    # The first polls of a run can still read the previous run's phase: keep from the last loading_model on.
    sm = d.get('samples') or []
    i = max((j for j, s in enumerate(sm) if s[1] == 'loading_model'), default = None)
    if i is None: return sm
    while i > 0 and sm[i - 1][1] == 'loading_model': i -= 1
    return sm[i:]


for f in glob.glob(os.path.join(H, '../../pulled/*/runs/*.json')):
    n = os.path.basename(f)[:-5].replace('-onlinetok', '')
    d = json.load(open(f)); d['own'] = own_samples(d); R[n] = d
GIB = 2**30
ZOO = '989ac3f4'
FOOT = (f'Colab G4: RTX PRO 6000 Blackwell 96 GB, 177 GiB host RAM, Linux. Unsloth Studio on unslothai/unsloth main 211f0ee22, zoo stack {ZOO} '
        '(main + zoo#1619 + zoo#2269 + save-size fix). Qwen3.8-Flash-Next-FP8, LoRA r16 all linear, 8-bit paged AdamW, FineTome-100k rows 0-400, '
        'packed, 10 steps, batch 1. Run through the Unsloth Studio training API, 2026-11-12. VRAM = torch peak allocated; host = peak host RAM in use.')
QFMT = lambda v: '?' if v == 0 else tc._fmt(v)
BG, TRACK, INK, MUTED, FAINT, GREEN, PINK, BASE = tc.BG, tc.TRACK, tc.INK, tc.MUTED, tc.FAINT, tc.GREEN, tc.PINK, tc.BASE


def ok(k):
    return k in R and R[k]['status'] == 'completed'


def vram(k):
    o = R[k].get('offload') or {}
    return o.get('vram_peak_bytes', 0) / GIB if o else R[k]['peak_gpu_mib'] / 1024


def toks(k):
    c = R[k]['cfg']; return c.get('ctx', 1024) * c.get('bs', 1) / R[k]['median_step_s']


# 0. Hero: Block Swap off vs on. Off = Unsloth Studio's own VRAM estimate (log: usable_gb=94.91 required_gb=372.632, GiB).
OFF_EST = 372.6
off = R.get('gif-off-ctx1024')
off_lab = 'Block Swap off (estimate)'
hero_on = [(k, l, kd) for k, l, kd in (('nt-c32-ctx1024', 'Block Swap, Count 32', 'new'), ('nt-c36-ctx1024', 'Block Swap, Count 36', 2)) if ok(k)]
tc.thick_chart(
    panels = [{'title': 'VRAM to train', 'unit': 'GiB', 'groups': [('Qwen3.8-Flash-Next, LoRA r16', [(off_lab, OFF_EST, 'rival')] +
                                                                    [(l, round(vram(k), 1), kd) for k, l, kd in hero_on])]},
              {'title': '96 GB cards needed', 'unit': 'GPUs', 'groups': [('RTX PRO 6000 Blackwell', [('Block Swap off (at least)', 4, 'rival'),
                                                                                                 ('Block Swap on', 1, 'new')])]},
              {'title': 'The price: training time', 'unit': 'hours / 1,000 steps', 'groups': [('ctx 1024, batch 1', [('Block Swap off, multi-GPU', 0.0, '#3a3a3a', 'not run yet')] + [(f"{l.replace('Block Swap, ', '')}, {R[k]['median_step_s']:.2f} s/step", round(R[k]['median_step_s'] * 1000 / 3600, 1), kd) for k, l, kd in hero_on])]}],
    title = 'A 180B MoE on one GPU instead of four',
    subtitle = 'Block Swap keeps 32 to 36 of 48 layers in host RAM and streams them in. The cost is step time: more layers swapped, slower steps.',
    meta = ['Qwen3.8-Flash-Next-FP8 (180B, 48 layers, 512 experts)', 'Unsloth Studio', 'Colab G4'],
    footnote = 'Off: VRAM Unsloth Studio estimates for this model with no Block Swap (training log, required_gb=372.632 against usable_gb=94.91 per card). No step time for off: it cannot run on one card, and Colab offers no 4-GPU VM. Run with Block Swap off on this G4, it fails while loading: CUDA out of memory, tried to allocate 93.77 GiB. ' + FOOT,
    out = os.path.join(H, '00-hero'), ncols = 3, width = 14.0, fmt = QFMT)

# 1. Count ladder
counts = [k for k in ('nt-c32-ctx1024', 'nt-c36-ctx1024', 'nt-c40-ctx1024') if ok(k)]
lab = {k: f"Count {R[k]['cfg']['offload']}" for k in counts}
kinds = ['base', 0, 2]
tc.thick_chart(
    panels = [
        {'title': 'Peak VRAM', 'unit': 'GiB', 'groups': [('RTX PRO 6000, 96 GB', [(lab[k], round(vram(k), 1), kinds[i]) for i, k in enumerate(counts)])]},
        {'title': 'Peak host RAM', 'unit': 'GiB of 177', 'groups': [('Colab G4 VM', [(lab[k], round(R[k]['peak_host_used_gib'], 1), kinds[i]) for i, k in enumerate(counts)])]},
        {'title': 'Step time', 'unit': 's / step', 'groups': [('ctx 1024, batch 1', [(lab[k], R[k]['median_step_s'], kinds[i]) for i, k in enumerate(counts)])]},
        {'title': 'Throughput', 'unit': 'tokens / s', 'groups': [('ctx 1024, batch 1', [(lab[k], round(toks(k)), kinds[i]) for i, k in enumerate(counts)])]},
    ],
    title = 'A 180B MoE trains on one 96 GB card in Unsloth Studio',
    subtitle = 'Every 4 more layers in host RAM trade VRAM for host RAM and speed.',
    meta = ['Qwen3.8-Flash-Next-FP8 (180B, 48 layers)', 'Block Swap Count = layers kept in host RAM', 'Unsloth Studio training API'],
    footnote = FOOT + ' Count 32 ran 6.05 s/step here vs 6.87 in the October bench script on the same card.', out = os.path.join(H, '01-count-ladder'))

# 2. The fork bug: host RAM vs time, aligned at the training phase
fig = plt.figure(figsize = (10, 6.4))
fig.text(0.04, 1 - 0.42 / 6.4, 'Why Unsloth Studio killed it: forking a 133 GiB trainer', fontsize = 21, weight = 'bold', va = 'center')
fig.text(0.04, 1 - 0.88 / 6.4, 'Each fork of the trainer added ~27 GiB of host RAM in seconds and the VM killed it; with no fork, it trains.', fontsize = 12.5, color = GREEN, va = 'center')
fig.text(0.04, 1 - 1.22 / 6.4, 'Count 32, ctx 1024  ·  host RAM in use on the VM  ·  t = 0 at the training phase', fontsize = 9.5, color = MUTED, va = 'center')
ax = fig.add_axes([0.08, 0.14, 0.88, 0.6]); ax.set_facecolor(BG)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(colors = MUTED, length = 0); ax.grid(color = TRACK, lw = 1.2); ax.set_axisbelow(True)
series = [('c32-ctx1024', 'dataset.map pool forks (defaults)', PINK), ('nt-c32-ctx1024-mapfork', 'dataset.map pool forks (rerun)', '#c9a24a'),
          ('nt-c32-ctx1024', 'num_proc 1 by env', '#7d9ad6'), ('fix-c32-ctx1024', 'fix branch, no env', GREEN)]
for k, name, c in series:
    if k not in R: continue
    sm = R[k]['own']
    t0 = next((s[0] for s in sm if s[1] == 'training'), None)
    if t0 is None: continue
    keep = [s for s in sm if -60 <= s[0] - t0 <= 120 and s[1] != 'error']
    keep = keep if R[k]['status'] == 'error' else [s for s in keep if s[0] - t0 <= 100]
    xs = [s[0] - t0 for s in keep]; ys = [s[3] for s in keep]
    ax.plot(xs, ys, lw = 5, color = c, solid_capstyle = 'round')
    killed = R[k]['status'] == 'error'
    if killed:
        ax.plot([xs[-1]], [ys[-1]], marker = 'X', ms = 16, color = c)
        dy = {'c32-ctx1024': 6, 'nt-c32-ctx1024-mapfork': -9}.get(k, 0)
        ax.text(xs[-1] + 4, ys[-1] + dy, f'killed at {ys[-1]:.0f} GiB: ' + name, color = c, fontsize = 10.5, weight = 'bold', va = 'center')
    else:
        ax.text(xs[-1] + 3, ys[-1] + (4 if 'fix' in k else -4), name + ', trains', color = c, fontsize = 10.5, weight = 'bold', va = 'center')
ax.axhline(177, color = MUTED, lw = 2, ls = (0, (4, 3))); ax.text(-58, 178.5, 'VM limit 177 GiB', color = MUTED, fontsize = 10, weight = 'bold')
ax.set_xlim(-60, 175); ax.set_ylim(120, 182); ax.set_xlabel('seconds from the training phase', color = MUTED); ax.set_ylabel('host RAM in use, GiB', color = MUTED)
fig.text(0.04, 0.025, FOOT[:190] + '…', fontsize = 7.5, color = FAINT)
fig.savefig(os.path.join(H, '02-fork-bug.png'), dpi = 200); fig.savefig(os.path.join(H, '02-fork-bug.svg')); plt.close(fig)

# 3. Every run, one row: trains / VRAM OOM / host kill / guard refused
rows = []
order = lambda k: (0 if R[k]['status'] == 'completed' else 1, str(R[k]['cfg']['offload']), k)
for k, d in sorted(R.items(), key = lambda kv: order(kv[0])):
    if k.startswith('gif-') and k != 'gif-off-ctx1024': continue  # GIF reruns repeat sweep configs
    c = d['cfg']; e = (d.get('error') or '')
    st = 'trains' if d['status'] == 'completed' else 'refused' if 'needs' in e and 'host RAM' in e else 'VRAM OOM' if 'CUDA out of memory' in e else 'host kill' if 'exitcode=-9' in e else d['status']
    fork = ' (fix branch)' if k.startswith('fix') else '' if k.startswith(('nt-', 'gif-')) and not k.endswith('mapfork') else ' (map fork)'
    desc = 'Block Swap off' if c['offload'] == 0 else f"Count {c['offload']}" + (f", r{c['r']}" if c.get('r') else '') + (', MLP toggle off (ignored)' if c.get('mlp') is False else '') + (f", ctx {c['ctx']}" if c.get('ctx', 1024) != 1024 else '') + (f", batch {c['bs']}" if c.get('bs') else '') + (', 4-bit asked (ran 16-bit)' if c.get('load_in_4bit') else '') + fork
    rows.append((desc, st, d))
H3 = 1.8 + 0.42 * len(rows) + 0.5
fig = plt.figure(figsize = (10, H3))
fig.text(0.04, 1 - 0.42 / H3, 'Every Flash-Next run in this sweep', fontsize = 21, weight = 'bold', va = 'center')
fig.text(0.04, 1 - 0.88 / H3, 'Tokenizing in process: Count 32 to 36 train. Count 40 and Auto are refused for host RAM; rank 64 never fits VRAM.', fontsize = 12.5, color = GREEN, va = 'center')
ax = fig.add_axes([0.04, 0.4 / H3, 0.92, 1 - 1.5 / H3]); ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(len(rows), -1.0)
col = {'trains': GREEN, 'refused': BASE, 'VRAM OOM': PINK, 'host kill': '#c9a24a'}
ax.text(0, -0.4, 'run', color = MUTED, fontsize = 9.5, weight = 'bold'); ax.text(4.6, -0.4, 'result', color = MUTED, fontsize = 9.5, weight = 'bold')
ax.text(6.4, -0.4, 'VRAM GiB', color = MUTED, fontsize = 9.5, weight = 'bold'); ax.text(7.7, -0.4, 'host GiB', color = MUTED, fontsize = 9.5, weight = 'bold'); ax.text(8.9, -0.4, 's / step', color = MUTED, fontsize = 9.5, weight = 'bold')
for i, (desc, st, d) in enumerate(rows):
    y = i + 0.5
    ax.text(0, y, desc, va = 'center', fontsize = 11, weight = 'bold')
    ax.add_patch(FancyBboxPatch((4.6, y - 0.32), 1.5, 0.64, boxstyle = 'round,pad=0,rounding_size=0.08', color = col.get(st, BASE), lw = 0))
    ax.text(5.35, y, st, ha = 'center', va = 'center', fontsize = 10.5, weight = 'bold', color = BG if st == 'trains' else INK)
    o = d.get('offload') or {}
    v = o.get('vram_peak_bytes', 0) / GIB if o else d['peak_gpu_mib'] / 1024
    ax.text(6.4, y, f'{v:.1f}', va = 'center', fontsize = 11, color = MUTED); ax.text(7.7, y, f"{d['peak_host_used_gib']:.0f}" if d.get('peak_host_used_gib') else '-', va = 'center', fontsize = 11, color = MUTED)
    ax.text(8.9, y, f"{d['median_step_s']:.2f}" if d['median_step_s'] else '-', va = 'center', fontsize = 11, color = MUTED)
fig.text(0.04, 0.012, 'VRAM: torch peak allocated where the run got far enough to report it, else nvidia-smi peak. Host: peak host RAM in use on the VM. ' + FOOT[:120] + '…', fontsize = 7.5, color = FAINT)
fig.savefig(os.path.join(H, '03-every-run.png'), dpi = 200); fig.savefig(os.path.join(H, '03-every-run.svg')); plt.close(fig)

# 4. Loss curves of the runs that trained
fig = plt.figure(figsize = (10, 5.8))
fig.text(0.04, 1 - 0.42 / 5.8, 'Same loss, step for step, at every Count', fontsize = 21, weight = 'bold', va = 'center')
fig.text(0.04, 1 - 0.88 / 5.8, 'Moving layers to host RAM changes speed, not the math: the curves overlap.', fontsize = 12.5, color = GREEN, va = 'center')
ax = fig.add_axes([0.08, 0.15, 0.86, 0.6]); ax.set_facecolor(BG)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(colors = MUTED, length = 0); ax.grid(color = TRACK, lw = 1.2)
pal = [BASE, tc.STEPS[1], GREEN, PINK, '#c9a24a', '#7d9ad6', '#b58ad6', '#d6d06b']
done = [(k, d) for k, d in R.items() if d['status'] == 'completed' and not k.startswith(('fix', 'gif-')) and d['cfg'].get('ctx', 1024) == 1024 and not d['cfg'].get('bs')]
for (k, d), c in zip(sorted(done), pal):
    m = d['metrics']; ax.plot(m['step_history'], m['loss_history'], lw = 4.5, color = c, solid_capstyle = 'round', marker = 'o', ms = 6)
    m['_label'] = [r for r in rows if r[2] is d][0][0]
for (k, d), c in zip(sorted(done), pal):
    pass
ax.legend([d['metrics']['_label'] for k, d in sorted(done)], frameon = False, labelcolor = INK, fontsize = 10.5, loc = 'upper right')
fig.text(0.04, 0.03, 'Same data and batches (ctx 1024, batch 1, seed 3407). Per-batch loss: it moves with the batch, not a trend. Same 2.78B trainable parameters in every run.', fontsize = 8, color = FAINT)
ax.set_xlabel('step', color = MUTED); ax.set_ylabel('loss', color = MUTED)
fig.savefig(os.path.join(H, '04-loss.png'), dpi = 200); fig.savefig(os.path.join(H, '04-loss.svg')); plt.close(fig)

# 5. Where the time goes: load, wait for step 1, the steps, save (Count 32 vs Count 36), and the guard refusal vs a kill
def split(k):
    sm = R[k]['own']; tr = [x for x in sm if x[1] == 'training']
    t0, ts = sm[0][0], tr[0][0]; first = next(x[0] for x in tr if (x[5] or 0) >= 1); last = tr[-1][0]
    return [('load model + set up LoRA', round((ts - t0) / 60, 1), 'base'), ('wait for step 1', round((first - ts) / 60, 1), 'rival'),
            (f"{R[k]['cfg'].get('steps', 10)} training steps", round((last - first) / 60, 1), 'new'), ('total', round(R[k]['elapsed_s'] / 60, 1), 2)]
tk = [(k, l) for k, l in (('nt-c32-ctx1024', 'Count 32 (6.05 s/step)'), ('nt-c36-ctx1024', 'Count 36 (9.07 s/step)')) if ok(k)]
if tk:
    tc.thick_chart(
        panels = [{'title': 'One 10-step run, start to finish', 'unit': 'minutes', 'groups': [(l, split(k)) for k, l in tk]},
                  {'title': 'Too big for host RAM', 'unit': 'minutes', 'groups': [('How long until you find out', [
                      ('fork kill (before the fix)', round(R['c32-ctx1024']['elapsed_s'] / 60, 1), 'rival', 'exit -9'),
                      ('guard refusal (zoo#1619)', max(0.1, round(R['nt-auto-ctx1024']['elapsed_s'] / 60, 1)) if 'nt-auto-ctx1024' in R else 0.1, 'new', 'says what fits')])]}],
        title = 'Where the time goes in a run', subtitle = 'Loading takes 13 min. Count 32 then waits ~9.5 min before step 1; Count 36 starts in under a minute.',
        meta = ['Unsloth Studio', 'Colab G4', 'checkpoint already on disk'],
        footnote = 'The ~9.5 min wait before step 1 at Count 32 repeated in all four Count 32 runs (9.3 to 9.8 min); Count 36 waited 0.6 to 0.9 min. Cause not confirmed. ' + FOOT,
        out = os.path.join(H, '05-time'), ncols = 2)

# 6. Speed: what a batch or a longer context costs at Count 36
sp = [(k, l, kind) for k, l, kind in (('nt-c36-ctx1024', 'ctx 1024, batch 1', 'base'), ('nt-c36-bs2', 'ctx 1024, batch 2', 'new'), ('nt-c36-ctx4096', 'ctx 4096, batch 1', 1)) if ok(k)]
if sp:
    tc.thick_chart(
        panels = [{'title': 'Throughput', 'unit': 'tokens / s', 'groups': [('Count 36', [(l, round(toks(k)), kind, f"{R[k]['median_step_s']:.1f} s/step") for k, l, kind in sp])]},
                  {'title': 'Peak VRAM', 'unit': 'GiB', 'groups': [('Count 36', [(l, round(vram(k), 1), kind) for k, l, kind in sp])]}],
        title = 'Batch 2 doubles throughput for free', subtitle = 'The step waits on layer copies, so a second sequence rides along at the same step time.',
        meta = ['Qwen3.8-Flash-Next-FP8', 'Block Swap Count 36', 'Unsloth Studio'], footnote = FOOT.replace('batch 1. ', ''),
        out = os.path.join(H, '06-speed'), ncols = 2)

# 7. Copy vs compute per layer (Unsloth Studio Offloaded layers panel; the trainer zeroes timing once a run ends)
PANEL = [('Count 32', 89.9, 93.4), ('Count 36', 96.4, 103.9), ('Count 36, batch 2', 97.4, 106.6), ('Count 36, ctx 4096', 95.7, 514.7)]
tc.thick_chart(
    panels = [{'title': 'Copy in vs compute, per swapped layer', 'unit': 'ms', 'groups': [(n, [('copy in from host RAM', c, 'base'),
              ('compute on the GPU', g, 'new', f'{g / c:.1f}x the copy')]) for n, c, g in PANEL]}],
    title = 'Why swapping barely costs speed',
    subtitle = 'The next layer copies in while this one computes, so compute hides the copy; at ctx 4096 compute is 5x it.',
    meta = ['Qwen3.8-Flash-Next-FP8', 'prefetch depth 2', 'Unsloth Studio Offloaded layers panel'],
    footnote = 'Per-layer averages read off the Offloaded layers panel on the Unsloth Studio training page mid-run. ' + FOOT,
    out = os.path.join(H, '07-copy-vs-compute'), ncols = 1)

# 8. Where the 180B lives (Block Swap state the trainer reports)
live = [(k, l) for k, l in (('nt-c32-ctx1024', 'Count 32'), ('nt-c36-ctx1024', 'Count 36')) if ok(k) and R[k].get('offload')]
if live:
    o = {k: R[k]['offload'] for k, _ in live}
    g = lambda k, f: round(o[k][f] / GIB, 1)
    tc.thick_chart(
        panels = [{'title': 'On the GPU', 'unit': 'GiB of 95', 'groups': [(l, [('peak VRAM', g(k, 'vram_peak_bytes'), 'new'),
                   ('swap slots (part of peak)', g(k, 'pool_bytes'), 1)]) for k, l in live]},
                  {'title': 'In host RAM', 'unit': 'GiB', 'groups': [(l, [('swapped layers', g(k, 'host_bytes'), 'base'),
                   ('of which pinned (fast copy)', g(k, 'pinned_bytes'), 'rival', 'the rest is pageable')]) for k, l in live]}],
        title = 'Where 180B parameters live while training',
        subtitle = 'About half the 161 GiB model sits in host RAM, and only a sliver of that is pinned for fast copies.',
        meta = ['Qwen3.8-Flash-Next-FP8', 'Block Swap state reported by the trainer'], footnote = FOOT,
        out = os.path.join(H, '08-where-it-lives'), ncols = 2)

# 9. What trains on one G4: LoRA rank x Count, Block Swap off first
cols = ['Block Swap off', 'Count 28', 'Count 32', 'Count 36', 'Count 40', 'Auto (43)']
def cell(rank, col):
    if col == 'Block Swap off':
        d = R.get('gif-off-ctx1024')
        if rank == 64: return ('not run', TRACK)
        if not d: return ('pending', TRACK)
        return ('trains', GREEN) if d['status'] == 'completed' else ('VRAM OOM', PINK) if 'memory' in (d.get('error') or '').lower() else ('fails', PINK)
    if col in ('Count 40', 'Auto (43)'): return ('refused', BASE) if rank == 16 else ('refused*', BASE)
    if rank == 64: return ('VRAM OOM', PINK)
    return {'Count 28': ('untested**', TRACK), 'Count 32': ('trains', GREEN), 'Count 36': ('trains', GREEN)}[col]
HG = 4.4
fig = plt.figure(figsize = (10, HG)); fig.patch.set_facecolor(BG)
fig.text(0.04, 1 - 0.42 / HG, 'What trains on one 96 GB card', fontsize = 21, weight = 'bold', va = 'center', color = INK)
fig.text(0.04, 1 - 0.88 / HG, 'LoRA rank 16 trains at Count 32 to 36. Rank 64 never fits. The host-RAM guard stops at 39.', fontsize = 12, color = GREEN, va = 'center')
ax = fig.add_axes([0.03, 0.17, 0.94, 0.58]); ax.axis('off'); ax.set_xlim(-1.7, len(cols)); ax.set_ylim(2.05, -0.5)
for j, c in enumerate(cols):
    ax.text(j + 0.5, -0.25, c, ha = 'center', fontsize = 10, color = MUTED, weight = 'bold')
for i, rank in enumerate((16, 64)):
    ax.text(-0.12, i + 0.5, f'LoRA rank {rank}', ha = 'right', va = 'center', fontsize = 11.5, weight = 'bold', color = INK)
    for j, c in enumerate(cols):
        lab, colr = cell(rank, c)
        ax.add_patch(FancyBboxPatch((j + 0.06, i + 0.1), 0.88, 0.8, boxstyle = 'round,pad=0,rounding_size=0.12', fc = colr, lw = 0))
        ax.text(j + 0.5, i + 0.5, lab, ha = 'center', va = 'center', fontsize = 10.5, weight = 'bold', color = BG if colr == GREEN else INK)
fig.text(0.04, 0.085, '* not run at rank 64: the guard counts layer bytes, which rank does not change.  ** only ran with the dataset.map fork, which killed it for host RAM.', fontsize = 7.5, color = FAINT, wrap = True)
fig.text(0.04, 0.035, 'Colab G4, RTX PRO 6000 96 GB, 177 GiB host RAM. Unsloth Studio, zoo stack ' + ZOO + ', Qwen3.8-Flash-Next-FP8, ctx 1024, batch 1.', fontsize = 7.5, color = FAINT)
fig.savefig(os.path.join(H, '09-what-fits.png'), dpi = 200, facecolor = BG); fig.savefig(os.path.join(H, '09-what-fits.svg'), facecolor = BG); plt.close(fig)

# 10. Efficiency: GPU-hours and wall-clock per 1M tokens. Block Swap off is a placeholder until the multi-GPU run lands
# (drop its run JSON in as runs/aws-off-ctx1024.json with cfg.gpus set).
PEND = '#3a3a3a'
effs = [(k, l, kd) for k, l, kd in (('nt-c32-ctx1024', 'Count 32, 1 GPU', 'new'), ('nt-c36-ctx1024', 'Count 36, 1 GPU', 1), ('nt-c36-bs2', 'Count 36 batch 2, 1 GPU', 2)) if ok(k)]
aws = R.get('aws-off-ctx1024')
def eff_rows(per_gpu):
    rows = [(l, round(1e6 / toks(k) / 3600, 2), kd) for k, l, kd in effs]
    if aws and aws['status'] == 'completed':
        g = aws['cfg'].get('gpus', 8); h = 1e6 / toks('aws-off-ctx1024') / 3600
        rows.insert(0, (f'Block Swap off, {g} GPUs', round(h * (g if per_gpu else 1), 2), 'rival'))
    else:
        rows.insert(0, ('Block Swap off, multi-GPU', 0.0, PEND, 'not run yet, see HANDOFF.md'))
    return rows
tc.thick_chart(
    panels = [{'title': 'GPU-hours per 1M tokens', 'unit': 'lower is better', 'groups': [('ctx 1024', eff_rows(True))]},
              {'title': 'Wall-clock hours per 1M tokens', 'unit': 'lower is better', 'groups': [('ctx 1024', eff_rows(False))]}],
    title = 'Is one GPU with Block Swap cheaper per token?',
    subtitle = ('Block Swap off on 8 GPUs is waiting on a run. To beat batch 2 per GPU-hour it needs over 1,808 tokens/s.' if not aws else
                'Block Swap off measured on a multi-GPU box vs Block Swap on one card.'),
    meta = ['Qwen3.8-Flash-Next-FP8', 'LoRA r16', 'Unsloth Studio'],
    footnote = ('Block Swap off: placeholder until someone runs it on a box with ~373 GiB of VRAM (8 x 96 GB or 8 x H100); recipe in flash-next-sweep/HANDOFF.md. '
                'GPU-hours = GPUs x wall-clock. ' + FOOT),
    out = os.path.join(H, '10-efficiency'), ncols = 2, fmt = QFMT)

# 11. Best settings per context length: measured rows, then untested candidates
P = '#3a3a3a'
CTXG = [('ctx 1024', [('nt-c36-bs2', 'Count 36, batch 2', 'new'), ('nt-c32-ctx1024', 'Count 32, batch 1', 2), ('nt-c36-ctx1024', 'Count 36, batch 1', 'base')],
                     [('Count 32, batch 2', 'should fit, ~83 GiB'), ('Count 36, batch 4', 'should fit, ~77 GiB'), ('Count 28, batch 1', 'may not fit VRAM')]),
        ('ctx 4096', [('nt-c36-ctx4096', 'Count 36, batch 1', 'new')],
                     [('Count 32, batch 1', 'should fit, ~85 GiB'), ('Count 36, batch 2', 'should fit, ~80 GiB')]),
        ('either context', [], [('Block Swap off, ~8 GPUs', 'untested, see HANDOFF.md')])]
def best_groups(val):
    return [(g, [(l, val(k), kd) for k, l, kd in meas if ok(k)] + [(l if l.startswith('Block') else l + ' (untested)', 0.0, P, n) for l, n in todo]) for g, meas, todo in CTXG]
tc.thick_chart(
    panels = [{'title': 'Throughput', 'unit': 'tokens / s, higher is better', 'groups': best_groups(lambda k: round(toks(k)))},
              {'title': 'Peak VRAM', 'unit': 'GiB of 95', 'groups': best_groups(lambda k: round(vram(k), 1))}],
    title = 'The best settings on one 96 GB GPU',
    subtitle = 'Pick context for your data. At 1024, Count 36 batch 2 is fastest; 4096 trains at nearly the same tokens/s.',
    meta = ['Qwen3.8-Flash-Next-FP8', 'LoRA r16', 'Unsloth Studio', 'Colab G4'],
    footnote = 'Lower Count is faster but needs more VRAM; Count 40+ is refused for host RAM. Untested VRAM figures are estimates from the measured runs. ' + FOOT.replace('batch 1. ', ''),
    out = os.path.join(H, '11-best-settings'), ncols = 2, fmt = QFMT)

print('charts:', sorted(f for f in os.listdir(H) if f.endswith('.png')))
