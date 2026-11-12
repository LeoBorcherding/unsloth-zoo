"""Flash-Next Block Swap sweep driven through Unsloth Studio's own training API, with UI screenshots.
Runs detached on the VM:  PLAN=a python3 fns_job.py   -> /content/fns/{runs,shots,logs}, progress.jsonl"""
import json, os, re, subprocess, sys, threading, time, urllib.request, urllib.error

ROOT = "/content/fns"
RUNS, SHOTS = f"{ROOT}/runs", f"{ROOT}/shots"
MODEL = "unsloth/Qwen3.8-Flash-Next-FP8"
ZOO = os.environ.get("ZOO_REF", "git+https://github.com/LeoBorcherding/unsloth-zoo@989ac3f4df75e78dd6c5b4038f796321f016586b")
os.makedirs(RUNS, exist_ok = True); os.makedirs(SHOTS, exist_ok = True)
BASE = "http://127.0.0.1:%d" % json.load(open("/content/sbx/status.json")).get("port", 8888)


def log(ev):
    ev["t"] = time.time()
    open(f"{ROOT}/progress.jsonl", "a").write(json.dumps(ev) + "\n")
    print(json.dumps(ev), flush = True)


def call(method, path, body = None, timeout = 120):
    req = urllib.request.Request(BASE + path, method = method)
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer " + TOKEN[0])
    try:
        with urllib.request.urlopen(req, json.dumps(body).encode() if body is not None else None, timeout = timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:3000]
    except Exception as e:
        return 0, repr(e)[:500]


TOKEN = [""]


def login():
    u, p = open("/content/sbx/login").read().split()[:2]
    req = urllib.request.Request(BASE + "/api/auth/login", method = "POST", data = json.dumps({"username": u, "password": p}).encode(),
                                 headers = {"Content-Type": "application/json"})
    TOKEN[0] = json.loads(urllib.request.urlopen(req, timeout = 60).read())["access_token"]


def studio_python():
    out = subprocess.run("ps -eo args", shell = True, capture_output = True, text = True).stdout
    for line in out.splitlines():
        if "run_server" in line or "serve.py" in line or "uvicorn" in line:
            exe = line.split()[0]
            if "python" in exe and os.path.exists(exe):
                return exe
    for c in (os.path.expanduser("~/.unsloth/studio/unsloth_studio/bin/python"), sys.executable):
        if os.path.exists(c):
            return c


def prep():
    py = studio_python()
    log({"event": "studio-python", "py": py})
    if not os.path.exists(f"{ROOT}/zoo_done"):
        r = subprocess.run([py, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall", "unsloth-zoo @ " + ZOO], capture_output = True, text = True)
        log({"event": "zoo", "rc": r.returncode, "out": (r.stdout + r.stderr)[-800:]})
        # The training worker may run under a different interpreter than the server: cover python3 too.
        if py != "/usr/bin/python3":
            subprocess.run(["python3", "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall", "unsloth-zoo @ " + ZOO], capture_output = True)
        open(f"{ROOT}/zoo_done", "w").write(ZOO)
    v = subprocess.run([py, "-c", "import unsloth_zoo,os;print(unsloth_zoo.__version__, os.path.dirname(unsloth_zoo.__file__))"], capture_output = True, text = True)
    log({"event": "zoo-version", "out": (v.stdout + v.stderr)[-300:]})
    t = time.time()
    subprocess.run([py, "-c", f"from huggingface_hub import snapshot_download as s; s({MODEL!r}, allow_patterns=['*.safetensors','*.json','*.jinja','*.txt','tokenizer*','*.py'])"])
    log({"event": "downloaded", "s": round(time.time() - t)})
    # Studio's opt-in latest-transformers sidecar (Flash-Next needs transformers >= 5.20).
    backend = "/content/unsloth/studio/backend"
    code = ("import sys; sys.path.insert(0, %r)\nimport json, urllib.request\nimport utils.transformers_latest as tl\n"
            "_f = tl._fetch_remote_model_types\n"
            "tl._fetch_remote_model_types = lambda ref: _f(ref) or (_f(ref[:-2]) if ref.endswith('.0') else None)\n"
            "v = json.load(urllib.request.urlopen('https://pypi.org/pypi/transformers/json', timeout=30))['info']['version']\n"
            "print(tl.install_latest_transformers(v))" % backend)
    for i in range(5):
        r = subprocess.run([py, "-c", code], capture_output = True, text = True)
        log({"event": "latest-tf", "try": i, "out": (r.stdout + r.stderr)[-500:]})
        if "'success': True" in r.stdout:
            break
        time.sleep(30 * (i + 1))


def gpu_used_mib():
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output = True, text = True).stdout.split()[0])
    except Exception:
        return None


def top_procs():
    import psutil
    out = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            out.append((p.memory_info().rss, p.info["pid"], " ".join(p.info["cmdline"] or [])[:80]))
        except Exception:
            pass
    shm = next((int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("Shmem:")), 0)
    return {"shmem_gib": round(shm / 2**20, 1), "top": [[round(r / 2**30, 1), pid, c] for r, pid, c in sorted(out, reverse = True)[:4]]}


def host():
    import psutil
    vm = psutil.virtual_memory()
    rss = 0
    for p in psutil.process_iter(["cmdline"]):
        try:
            if "python" in " ".join(p.info["cmdline"] or [])[:200]:
                rss = max(rss, p.memory_info().rss)
        except Exception:
            pass
    return round((vm.total - vm.available) / 2**30, 1), round(rss / 2**30, 1), round(vm.total / 2**30, 1)


SHOT_JS = r'''
import asyncio, sys, json
from playwright.async_api import async_playwright
base, out, user, pw, path = sys.argv[1:6]
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=1.5)
        await pg.goto(base + "/login"); await pg.wait_for_timeout(3000)
        ins = await pg.query_selector_all("input")
        if len(ins) >= 2:
            await ins[0].fill(user); await ins[1].fill(pw); await pg.keyboard.press("Enter")
            await pg.wait_for_timeout(5000)
        await pg.goto(base + path); await pg.wait_for_timeout(9000)
        await pg.screenshot(path=out, full_page=True)
        await b.close()
asyncio.run(main())
'''


def shot(name, path = "/studio"):
    try:
        if not os.path.exists(f"{ROOT}/pw_done"):
            subprocess.run("python3 -m pip install -q playwright && python3 -m playwright install --with-deps chromium", shell = True, capture_output = True, timeout = 900)
            open(f"{ROOT}/pw_done", "w").write("1")
        open(f"{ROOT}/shot.py", "w").write(SHOT_JS)
        u, p = open("/content/sbx/login").read().split()[:2]
        r = subprocess.run(["python3", f"{ROOT}/shot.py", BASE, f"{SHOTS}/{name}.png", u, p, path], capture_output = True, text = True, timeout = 180)
        log({"event": "shot", "name": name, "ok": os.path.exists(f"{SHOTS}/{name}.png"), "err": r.stderr[-300:]})
    except Exception as e:
        log({"event": "shot", "name": name, "ok": False, "err": repr(e)[:300]})


def payload(c):
    body = {
        "model_name": MODEL, "training_type": "LoRA/QLoRA", "load_in_4bit": c.get("load_in_4bit", False),
        "max_seq_length": c.get("ctx", 1024), "hf_dataset": "mlabonne/FineTome-100k", "format_type": "auto",
        "train_split": "train", "dataset_slice_start": 0, "dataset_slice_end": 400,
        "num_epochs": 1, "learning_rate": "2e-4", "batch_size": c.get("bs", 1), "gradient_accumulation_steps": 1,
        "warmup_steps": 1, "max_steps": c.get("steps", 10), "save_steps": 0, "packing": c.get("packing", True),
        "optim": c.get("optim", "paged_adamw_8bit"), "lr_scheduler_type": "linear", "use_lora": True,
        "lora_r": c.get("r", 16), "lora_alpha": c.get("r", 16), "lora_dropout": 0.0,
        "target_modules": c.get("targets", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        "gradient_checkpointing": "unsloth", "offload_layers": c.get("offload", 32), "prefetch_depth": c.get("prefetch", 2),
        "random_seed": 3407, "trust_remote_code": False, "train_on_completions": False,
        "finetune_attention_modules": c.get("attn", True), "finetune_mlp_modules": c.get("mlp", True),
    }
    if c.get("budget") is not None:
        body["offload_vram_gb"] = c["budget"]
    return body


def run(c):
    out = f"{RUNS}/{c['id']}.json"
    if os.path.exists(out):
        return
    call("POST", "/api/train/stop", {}); call("POST", "/api/train/reset", {}); time.sleep(5)
    body = payload(c)
    log({"event": "start", "id": c["id"], "cfg": c})
    t0 = time.time()
    s, r = call("POST", "/api/train/start", body, timeout = 300)
    res = {"id": c["id"], "cfg": c, "start_http": s, "start_resp": r if isinstance(r, (dict, list)) else str(r)[:2000]}
    peak_gpu, peak_used, peak_rss, samples, last, phases, shots, offload = 0, 0, 0, [], None, {}, 0, None
    step_t = {}
    while s in (200, 201, 202) and time.time() - t0 < c.get("cutoff", 50 * 60):
        time.sleep(3)
        _, st = call("GET", "/api/train/status")
        if not isinstance(st, dict):
            continue
        ph = st.get("phase"); phases.setdefault(ph, round(time.time() - t0))
        g = gpu_used_mib(); used, rss, total = host()
        peak_gpu, peak_used, peak_rss = max(peak_gpu, g or 0), max(peak_used, used), max(peak_rss, rss)
        _, m = call("GET", "/api/train/metrics")
        step = m.get("current_step") if isinstance(m, dict) else None
        if step is not None and step not in step_t:
            step_t[step] = time.time() - t0
        _, o = call("GET", "/api/train/offload")
        if isinstance(o, dict) and o.get("active"):
            offload = o
        if len(samples) < 4000:
            samples.append([round(time.time() - t0, 1), ph, g, used, rss, step] + ([top_procs()] if len(samples) % 3 == 0 or used > 140 else []))
        if ph == "training" and step and step >= 3 and shots == 0:
            shots = 1
            threading.Thread(target = shot, args = (c["id"] + "-training",), daemon = True).start()
        if ph in ("completed", "error", "stopped") or (ph == "idle" and time.time() - t0 > 60):
            last = st
            break
        last = st
    if shots == 0 and c.get("shot_on_end", True):
        shot(c["id"] + "-end")
    steps = sorted(step_t.items())
    dts = [b[1] - a[1] for a, b in zip(steps, steps[1:])]
    dts_sorted = sorted(dts[1:]) if len(dts) > 2 else sorted(dts)
    res.update(status = (last or {}).get("phase"), error = (last or {}).get("error"), message = (last or {}).get("message"),
               warnings = (last or {}).get("warnings"), phases_at_s = phases, elapsed_s = round(time.time() - t0),
               peak_gpu_mib = peak_gpu, peak_host_used_gib = peak_used, peak_proc_rss_gib = peak_rss, host_total_gib = host()[2],
               median_step_s = round(dts_sorted[len(dts_sorted) // 2], 2) if dts_sorted else None, steps_seen = len(steps),
               offload = offload, samples = samples)
    _, m = call("GET", "/api/train/metrics")
    res["metrics"] = m if isinstance(m, dict) else str(m)[:500]
    json.dump(res, open(out, "w"), indent = 1)
    log({"event": "done", "id": c["id"], "status": res["status"], "peak_gpu_mib": peak_gpu, "rss": peak_rss,
         "median_step_s": res["median_step_s"], "error": (res["error"] or "")[:300]})
    call("POST", "/api/train/stop", {}); time.sleep(20)


P = {"steps": 10}
PLANS = {
    # Forks off (online tokenization 0, UNSLOTH_DATASET_NUM_PROC=1). Count 32 sits at ~95 GB VRAM, so the heavier variants go up a Count.
    "a": [dict(id = "nt-c32-ctx1024", offload = 32, ctx = 1024), dict(id = "nt-auto-ctx1024", offload = "auto", ctx = 1024, steps = 3),
          dict(id = "nt-c36-ctx1024", offload = 36, ctx = 1024), dict(id = "nt-c40-ctx1024", offload = 40, ctx = 1024),
          dict(id = "nt-c36-ctx4096", offload = 36, ctx = 4096)],
    "b": [dict(id = "nt-c32-attn-only", offload = 32, mlp = False), dict(id = "nt-c36-r64", offload = 36, r = 64),
          dict(id = "nt-c36-bs2", offload = 36, bs = 2), dict(id = "nt-c32-4bit", offload = 32, load_in_4bit = True, steps = 5)],
}

if __name__ == "__main__":
    login()
    log({"event": "plan-start", "plan": os.environ["PLAN"]})
    prep(); login()
    shot("00-studio-home", "/")
    for c in PLANS[os.environ["PLAN"]]:
        c = {**P, **c}
        try:
            login(); run(c)
        except Exception as e:
            log({"event": "case-crash", "id": c["id"], "err": repr(e)[:500]})
    log({"event": "plan-done"})
