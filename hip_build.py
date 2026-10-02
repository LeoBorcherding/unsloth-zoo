"""Run unsloth_zoo's real install_llama_cpp(gpu_support=True) on a Linux box with the ROCm SDK
but no GPU. Only torch's device query is stubbed (FAKE_GFX); git clone, cmake, the compile
and the link are real. Prints what the build produced and exits 0 either way.

usage: FAKE_GFX=gfx1151[,gfx1030] python hip_build.py <zoo checkout> <llama.cpp dir>
"""
import glob, importlib.util, json, os, re, subprocess, sys, traceback, types

root, folder = sys.argv[1], sys.argv[2]
gfxs = os.environ["FAKE_GFX"].split(",")

spec = importlib.util.spec_from_file_location("zoo_llama_cpp", os.path.join(root, "unsloth_zoo", "llama_cpp.py"))
lc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = lc
spec.loader.exec_module(lc)

lc.torch = types.SimpleNamespace(
    version = types.SimpleNamespace(hip = os.environ.get("FAKE_HIP", "7.0"), cuda = None),
    cuda = types.SimpleNamespace(
        is_available = lambda: True,
        device_count = lambda: len(gfxs),
        get_device_properties = lambda i: types.SimpleNamespace(gcnArchName = gfxs[i]),
        get_device_capability = lambda i: (11, 5),
    ),
)

result = {"fake_gfx": gfxs, "error": None}
try:
    lc.install_llama_cpp(llama_cpp_folder = folder, gpu_support = True, print_output = True)
    result["install"] = "ok"
except BaseException as e:
    result["install"] = "raised"
    result["error"] = f"{type(e).__name__}: {str(e)[:400]}"
    traceback.print_exc()

cache = os.path.join(folder, "build", "CMakeCache.txt")
if os.path.exists(cache):
    keep = ("GGML_HIP:", "GGML_CUDA:", "GPU_TARGETS:", "AMDGPU_TARGETS:", "CMAKE_POSITION_INDEPENDENT_CODE:",
            "CMAKE_HIP_COMPILER:", "BUILD_SHARED_LIBS:", "CMAKE_BUILD_TYPE:")
    result["cmake_cache"] = [l.strip() for l in open(cache) if l.startswith(keep)]
try:
    result["llama_cpp_commit"] = subprocess.run(["git", "-C", folder, "rev-parse", "--short", "HEAD"],
                                                capture_output = True, text = True).stdout.strip()
except Exception:
    pass

bins = {}
for name in ("llama-quantize", "llama-cli", "llama-server", "llama-gguf-split", "llama-mtmd-cli"):
    hits = [p for p in glob.glob(os.path.join(folder, "**", name), recursive = True) if os.path.isfile(p)]
    if not hits:
        continue
    p = hits[0]
    ldd = subprocess.run(["ldd", p], capture_output = True, text = True).stdout
    objs = subprocess.run(["/opt/rocm/bin/roc-obj-ls", p], capture_output = True, text = True).stdout
    bins[name] = {
        "path": os.path.relpath(p, folder),
        "links_amdhip64": bool(re.search(r"libamdhip64", ldd)),
        "links_hipblas": bool(re.search(r"libhipblas", ldd)),
        "code_objects": sorted(set(re.findall(r"gfx\d+[a-z]?", objs))),
    }
result["binaries"] = bins
print("HIPBUILD_RESULT " + json.dumps(result))
