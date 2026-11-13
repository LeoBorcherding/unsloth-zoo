"""Charts for unslothai/unsloth#17701 and unslothai/unsloth-zoo#2499, from runs/*.json."""
import json, os, sys
sys.path.insert(0, os.path.expanduser('~/.claude/skills/thick-charts'))
import thick_charts as tc
import matplotlib.pyplot as plt

R = lambda n: json.load(open(f'runs/{n}.json'))
main, fix, dflt = R('40041-nt-c32-ctx1024-mapfork'), R('35699-fix-c32-ctx1024'), R('40041-c32-ctx1024')
LIMIT = main['host_total_gib']

def train_start(d):
    s = d['samples']
    return next(x[0] for i, x in enumerate(s) if x[1] == 'training' and i and s[i - 1][1] != 'training')

# 17701: host RAM over the whole run
fig, ax = plt.subplots(figsize=(10, 5.4))
ax.set_facecolor(tc.BG)
for d, col, lab in [(main, tc.PINK, 'main: OOM-killed before step 1'), (fix, tc.GREEN, 'this PR: trains all 25 steps')]:
    pts = [(x[0] / 60, x[3]) for x in d['samples'] if x[1] != 'error']
    ax.plot(*zip(*pts), color=col, lw=3.2, label=lab, solid_capstyle='round')
    if d is main:
        k = max(range(len(pts)), key=lambda i: pts[i][1])
        ax.scatter([pts[k][0]], [pts[k][1]], s=140, color=col, marker='X', zorder=5)
        ax.annotate(f'tokenizer pool forks the trainer:\n{pts[k-2][1]:.0f} -> {pts[k][1]:.0f} GiB in 9 s, then killed',
                    (pts[k][0], pts[k][1]), xytext=(1.0, 150), color=col, fontsize=10.5,
                    arrowprops=dict(arrowstyle='-', color=col))
ax.axhline(LIMIT, color=tc.FAINT, ls='--', lw=1.2)
ax.text(0.3, LIMIT + 1.5, f'host RAM on the VM: {LIMIT:.0f} GiB', color=tc.MUTED, fontsize=10)
ax.set_xlim(0, 28); ax.set_ylim(0, 188)
ax.set_xlabel('minutes since the run started', color=tc.MUTED); ax.set_ylabel('host RAM used (GiB)', color=tc.MUTED)
ax.tick_params(colors=tc.MUTED)
for sp in ax.spines.values(): sp.set_visible(False)
ax.grid(axis='y', color=tc.TRACK, lw=0.8)
ax.legend(loc='lower right', frameon=False, labelcolor=tc.INK, fontsize=11)
fig.text(0.06, 0.955, 'Forking the trainer to tokenize runs the host out of RAM', color=tc.INK, fontsize=16, weight='bold')
fig.text(0.06, 0.905, 'Block Swap keeps 133 GiB of layers in host RAM; the dataset.map fork duplicates the pages it touches.', color=tc.GREEN, fontsize=11.5)
fig.text(0.06, 0.055, 'Qwen3.8-Flash-Next-FP8, LoRA r16, Block Swap Count 32, ctx 1024, Colab G4 (RTX PRO 6000 96 GB, 177 GiB host).', color=tc.FAINT, fontsize=8.5)
fig.text(0.06, 0.02, 'main 211f0ee22 vs this PR e3e8fd3be, Nov 12 2026. Host RAM sampled every 3 s. The two runs were on different G4 VMs, so load speed differs.', color=tc.FAINT, fontsize=8.5)
fig.subplots_adjust(left=0.08, right=0.97, top=0.85, bottom=0.2)
fig.savefig('charts/pr-17701-host-ram.png', dpi=160); fig.savefig('charts/pr-17701-host-ram.svg')

steps = lambda d: d.get('steps_seen') or 0
tc.thick_chart(
    panels=[
        {'title': 'Peak host RAM', 'unit': f'GiB (VM has {LIMIT:.0f})', 'groups': [('Count 32, ctx 1024', [
            ('main, defaults', dflt['peak_host_used_gib'], 'rival', 'killed'),
            ('main, online tok. off', main['peak_host_used_gib'], 'rival', 'killed'),
            ('this PR, defaults', fix['peak_host_used_gib'], 'new', 'trained')])]},
        {'title': 'Training steps finished', 'unit': 'steps', 'groups': [('Count 32, ctx 1024', [
            ('main, defaults', steps(dflt), 'rival'),
            ('main, online tok. off', steps(main), 'rival'),
            ('this PR, defaults', steps(fix), 'new')])]},
    ],
    title='Block Swap runs in Unsloth Studio: killed before step 1 on main',
    subtitle='Tokenizing in process keeps host RAM 15 GiB lower, and the run trains.',
    meta=['Qwen3.8-Flash-Next-FP8 (180B MoE)', 'LoRA r16', 'Colab G4, RTX PRO 6000 96 GB'],
    footnote='Nov 12 2026. main 211f0ee22, this PR e3e8fd3be. Both main runs exit with code -9 (OOM killer). '
             'The PR run had no env overrides; it ran 25 steps at 6.05 s/step.',
    out='charts/pr-17701-outcome', ncols=2)

# 2499: disk-space estimate for one merge
tc.thick_chart(
    panels=[
        {'title': 'Disk needed for the merge, as estimated', 'unit': 'GiB', 'groups': [('Clef-flash, 21 layers in Block Swap', [
            ('main', 329, 'rival', 'save refused'),
            ('this PR', 4.6, 'new', 'saved'),
            ('free on the disk', 177.9, 'base', '191 GB')])]},
        {'title': 'One 4 KiB view into a 64 MiB arena', 'unit': 'KiB counted', 'groups': [('unit test', [
            ('main', 65536, 'rival'),
            ('this PR', 4, 'new')])]},
    ],
    title='The save-size check counted the whole Block Swap arena per view',
    subtitle='Counting each tensor\'s own bytes takes the estimate from 329 GiB to 4.6 GiB.',
    meta=['Block Swap packs a layer\'s weights into one host arena', 'every norm and bias is a view into it'],
    footnote='Merge numbers are from the original repro (Nov 10 2026). The unit test is '
             'tests/test_save_size_shared_storage.py; main = huggingface_hub.get_torch_storage_size.',
    out='charts/pr-2499-save-size', ncols=2)
