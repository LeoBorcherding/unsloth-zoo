"""On the VM: once training reaches step 2, record the Unsloth Studio training page for 40 s and cut a GIF of the
Offloaded layers panel (plus the full page) into /content/fns/shots."""
import asyncio, json, os, subprocess, time, urllib.request
from playwright.async_api import async_playwright
BASE = "http://127.0.0.1:%d" % json.load(open("/content/sbx/status.json")).get("port", 8888)
OUT = "/content/fns/shots"; os.makedirs(OUT, exist_ok = True)


def login():
    u, p = open("/content/sbx/login").read().split()[:2]
    r = urllib.request.Request(BASE + "/api/auth/login", data = json.dumps({"username": u, "password": p}).encode(),
                               headers = {"Content-Type": "application/json"}, method = "POST")
    return json.loads(urllib.request.urlopen(r, timeout = 60).read())


def get(path, tok):
    return json.loads(urllib.request.urlopen(urllib.request.Request(BASE + path, headers = {"Authorization": "Bearer " + tok}), timeout = 60).read())


def log(**e):
    e["t"] = time.time(); open("/content/fns/progress.jsonl", "a").write(json.dumps({"event": "gif", **e}) + "\n")


async def main():
    while True:
        t = login()
        try:
            st = get("/api/train/status", t["access_token"]); m = get("/api/train/metrics", t["access_token"])
            if st.get("phase") == "training" and (m.get("current_step") or 0) >= 2:
                break
        except Exception:
            pass
        await asyncio.sleep(5)
    async with async_playwright() as pw:
        b = await pw.chromium.launch()
        ctx = await b.new_context(viewport = {"width": 1600, "height": 1000}, record_video_dir = "/tmp/vid", record_video_size = {"width": 1600, "height": 1000})
        pg = await ctx.new_page()
        await pg.goto(BASE + "/login")
        await pg.evaluate("([a, r]) => { localStorage.setItem('unsloth_auth_token', a); localStorage.setItem('unsloth_auth_refresh_token', r || ''); }", [t["access_token"], t.get("refresh_token")])
        await pg.goto(BASE + "/studio"); await pg.wait_for_timeout(6000)
        panel = pg.get_by_text("Offloaded layers").first
        await panel.scroll_into_view_if_needed()
        await pg.evaluate("window.scrollBy(0, -140)"); await pg.wait_for_timeout(1500)
        box = await pg.evaluate("""() => { const el = [...document.querySelectorAll('*')].find(e => e.childElementCount === 0 && e.textContent.trim() === 'Offloaded layers');
            let c = el; for (let i = 0; i < 8 && c; i++) { const r = c.getBoundingClientRect(); if (r.width > 900 && r.height > 200) return [r.x, r.y, r.width, r.height]; c = c.parentElement; } return null; }""")
        t0 = time.time(); await pg.wait_for_timeout(40000)
        await pg.screenshot(path = f"{OUT}/fix-training-full.png")
        vpath = await pg.video.path(); await ctx.close(); await b.close()
    skip = 9  # page load + scroll at the start of the clip
    full = f"{OUT}/studio-training.gif"; crop = f"{OUT}/offloaded-layers.gif"
    pal = "split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer"
    subprocess.run(["ffmpeg", "-y", "-ss", str(skip), "-i", vpath, "-vf", f"fps=10,scale=1200:-1:flags=lanczos,{pal}", full], capture_output = True)
    if box:
        x, y, w, h = [int(v) for v in box]
        subprocess.run(["ffmpeg", "-y", "-ss", str(skip), "-i", vpath, "-vf", f"crop={w}:{h}:{x}:{y},fps=12,scale=1100:-1:flags=lanczos,{pal}", crop], capture_output = True)
    log(ok = os.path.exists(crop), box = box, full = os.path.getsize(full) if os.path.exists(full) else 0, crop = os.path.getsize(crop) if os.path.exists(crop) else 0)
asyncio.run(main())
