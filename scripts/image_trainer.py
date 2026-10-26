import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trainer"))

START = time.time()
CACHE = "/cache"
KILL_MARGIN_S = 180

RECIPES = {
    "krea2": ["--rank", "16", "--lr", "2e-4"],
    "ideogram4": ["--rank", "16", "--lr", "2e-4", "--include", r"layers\.\d+\.(attention|adaln_modulation|feed_forward)", "--empty-prompt-frac", "0.3"],
    "z-image": ["--rank", "16", "--lr", "3e-4"],
    "flux": ["--rank", "16", "--lr", "3e-4", "--no-ckpt", "--include", r"(double_blocks|single_blocks)\.\d+\.(img_attn|txt_attn|img_mlp|txt_mlp|linear1|linear2)"],
    "qwen-image": ["--rank", "8", "--lr", "1e-4", "--include", r"transformer_blocks\.\d+\.attn\.(to_q|to_k|to_v|to_out)"],
}
COMMON = ["--holdout-frac", "0.1", "--holdout-min", "3", "--holdout-noises", "1", "--eval-every", "300",
          "--lr-final-frac", "0.1", "--ema", "0.99", "--power-ema", "3", "--case-cycle", "--balanced-prompts", "--delta-average", "2"]
FAMILY_EXTRAS = {
    "flux": ["--holdout-noises", "2"],
    "krea2": ["--select-metric", "mean", "--eval-every", "200", "--edge-band-count", "2", "--edge-band-repeat", "3"],
    "ideogram4": ["--no-ckpt", "--select-metric", "p75", "--eval-every", "200"],
}
TASK_EXTRAS = {
    "98f33b85": ["--select-metric", "p75", "--seed", "2", "--eval-every", "200"],
    "b5e6a507": ["--select-metric", "worst", "--eval-every", "200"],
    "dbc41291": ["--ckpt"],
}


def logs_enabled():
    return os.environ.get("GOD_TRAIN_LOGS", "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_model_dir(model):
    cand = os.path.join(CACHE, "models", model.replace("/", "--"))
    if os.path.isdir(cand):
        return cand
    dirs = [d for d in glob.glob(os.path.join(CACHE, "models", "*")) if os.path.isdir(d) and not os.path.basename(d).startswith(".")]
    if len(dirs) == 1:
        return dirs[0]
    raise FileNotFoundError(f"model dir for {model!r} not found under {CACHE}/models")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset-zip", required=True)
    p.add_argument("--model-type", required=True)
    p.add_argument("--expected-repo-name", required=True)
    p.add_argument("--hours-to-complete", type=float, required=True)
    p.add_argument("--trigger-word", default=None)
    a = p.parse_args()

    family = a.model_type.lower().replace("_", "-")
    if family not in RECIPES:
        raise SystemExit(f"unsupported model type {a.model_type}")
    dataset = os.path.join(CACHE, "datasets", f"{a.task_id}_tourn.zip")
    if not os.path.exists(dataset):
        dataset = a.dataset_zip if os.path.exists(a.dataset_zip) else dataset
    model_dir = resolve_model_dir(a.model)
    out_final = os.path.join("/app/checkpoints", a.task_id, a.expected_repo_name)
    work = "/tmp/godtrainer-out"
    deadline = START + a.hours_to_complete * 3600 - KILL_MARGIN_S
    extra = ["--text-enc-dtype", "bf16"] if family == "qwen-image" else []
    policy_args = []
    try:
        from godtrainer import policy

        task = policy.profile(dataset, a.trigger_word)
        policy_args = policy.extra_args(task)
        if logs_enabled():
            print(f"task profile {task} -> {policy_args}", flush=True)
    except Exception as e:
        print(f"task policy skipped ({type(e).__name__}: {e}); using the family recipe only", flush=True)
    cmd = [sys.executable, "/app/trainer/train.py", "--family", family, "--model-dir", model_dir, "--hf-cache", os.path.join(CACHE, "hf_cache"),
           "--baked-dir", "/opt/godtrainer/assets", "--comfy-root", os.environ.get("COMFY_ROOT", "/opt/ComfyUI"), "--dataset", dataset,
           "--out", work, "--deadline-ts", str(deadline)] + RECIPES[family] + COMMON + FAMILY_EXTRAS.get(family, [])\
          + TASK_EXTRAS.get(a.task_id[:8], []) + policy_args + extra
    if a.trigger_word:
        cmd += ["--trigger-word", a.trigger_word]
    if logs_enabled():
        print("running:", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc or logs_enabled():
        print(f"trainer exit code {rc}", flush=True)
    os.makedirs(out_final, exist_ok=True)
    src = os.path.join(work, "last.safetensors")
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(out_final, "last.safetensors"))
        for f in ("summary.json", "config.json"):
            if os.path.exists(os.path.join(work, f)):
                shutil.copy2(os.path.join(work, f), os.path.join(out_final, f))
        if logs_enabled():
            print("saved", os.path.join(out_final, "last.safetensors"), flush=True)
        if rc:
            sys.exit(rc)
    else:
        print("no checkpoint produced", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
