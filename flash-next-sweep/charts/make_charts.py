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
BG, TRACK, INK, MUTED, FAINT, GREEN, PINK, BASE = tc.BG, tc.TRACK, tc.INK, tc.MUTED, tc.FAINT, tc.GREEN, tc.PINK, tc.BASE


def ok(k):
    return k in R and R[k]['status'] == 'completed'


def vram(k):
    o = R[k].get('offload') or {}
    return o.get('vram_peak_bytes', 0) / GIB if o else R[k]['peak_gpu_mib'] / 1024


def toks(k):
    c = R[k]['cfg']; return c.get('ctx', 1024) * c.get('bs', 1) / R[k]['median_step_s']


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
    footnote = FOOT, out = os.path.join(H, '01-count-ladder'))

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
    c = d['cfg']; e = (d.get('error') or '')
    st = 'trains' if d['status'] == 'completed' else 'refused' if 'needs' in e and 'host RAM' in e else 'VRAM OOM' if 'CUDA out of memory' in e else 'host kill' if 'exitcode=-9' in e else d['status']
    fork = '' if k.startswith('nt-') and not k.endswith('mapfork') else ' (map fork)'
    desc = f"Count {c['offload']}" + (f", r{c['r']}" if c.get('r') else '') + (', MLP toggle off (ignored)' if c.get('mlp') is False else '') + (f", ctx {c['ctx']}" if c.get('ctx', 1024) != 1024 else '') + (f", batch {c['bs']}" if c.get('bs') else '') + (', 4-bit asked (ran 16-bit)' if c.get('load_in_4bit') else '') + fork
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
    ax.text(6.4, y, f'{v:.1f}', va = 'center', fontsize = 11, color = MUTED); ax.text(7.7, y, f"{d['peak_host_used_gib']:.0f}", va = 'center', fontsize = 11, color = MUTED)
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
done = [(k, d) for k, d in R.items() if d['status'] == 'completed' and not k.startswith('fix') and d['cfg'].get('ctx', 1024) == 1024 and not d['cfg'].get('bs')]
for (k, d), c in zip(sorted(done), pal):
    m = d['metrics']; ax.plot(m['step_history'], m['loss_history'], lw = 4.5, color = c, solid_capstyle = 'round', marker = 'o', ms = 6)
    m['_label'] = [r for r in rows if r[2] is d][0][0]
for (k, d), c in zip(sorted(done), pal):
    pass
ax.legend([d['metrics']['_label'] for k, d in sorted(done)], frameon = False, labelcolor = INK, fontsize = 10.5, loc = 'upper right')
fig.text(0.04, 0.03, 'Same data and batches (ctx 1024, batch 1, seed 3407). Per-batch loss: it moves with the batch, not a trend. Same 2.78B trainable parameters in every run.', fontsize = 8, color = FAINT)
ax.set_xlabel('step', color = MUTED); ax.set_ylabel('loss', color = MUTED)
fig.savefig(os.path.join(H, '04-loss.png'), dpi = 200); fig.savefig(os.path.join(H, '04-loss.svg')); plt.close(fig)

# 5. Where the time goes in one run, and the guard's refusal vs a kill
k = 'nt-c32-ctx1024'
if ok(k):
    sm = R[k]['own']; spans = []
    for p in ('loading_model', 'configuring', 'training', 'finalizing'):
        ts = [x[0] for x in sm if x[1] == p]
        if ts: spans.append((p, ts[-1] - ts[0] + 3))
    tc.thick_chart(
        panels = [{'title': 'Where one run spends its time', 'unit': 'minutes', 'groups': [('Count 32, checkpoint already on disk', [
            ({'loading_model': 'load 173 GB + install swap', 'configuring': 'configure LoRA', 'training': 'train 10 steps', 'finalizing': 'save adapter'}[p], round(s / 60, 1), i if i < 3 else 'new') for i, (p, s) in enumerate(spans)])]},
                  {'title': 'Too big for host RAM', 'unit': 'minutes', 'groups': [('How long until you find out', [
                      ('Count 32 with a fork: loads, then killed', round(R['c32-ctx1024']['elapsed_s'] / 60, 1), 'rival', 'exit -9'),
                      ('Auto: refused by the guard', max(0.1, round(R['nt-auto-ctx1024']['elapsed_s'] / 60, 1)) if 'nt-auto-ctx1024' in R else 0.1, 'new', 'says how many layers fit')])]}],
        title = 'Waiting on a 180B model', subtitle = 'Loading is most of a run; zoo#1619 refuses an impossible plan before the load, with a reason.',
        meta = ['Unsloth Studio', 'Colab G4'], footnote = FOOT + ' The refusal names the layer count that fits; the kill only says exitcode -9.',
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

print('charts:', sorted(f for f in os.listdir(H) if f.endswith('.png')))
