import argparse
import gc
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True, choices=["krea2", "ideogram4", "qwen-image", "z-image", "flux"])
    p.add_argument("--model-dir", required=True)
    p.add_argument("--diffusion-checkpoint", default=None,
                   help="optional exact Comfy diffusion checkpoint override used for evaluator-parity experiments")
    p.add_argument("--hf-cache", default="/cache/hf_cache")
    p.add_argument("--baked-dir", default="/opt/godtrainer/assets")
    p.add_argument("--comfy-root", default=os.environ.get("COMFY_ROOT", "/opt/ComfyUI"))
    p.add_argument("--dataset", required=True, help="zip or directory of images + .txt captions")
    p.add_argument("--trigger-word", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--deadline-ts", type=float, default=None)
    p.add_argument("--minutes", type=float, default=None)
    p.add_argument("--save-reserve", type=float, default=90.0, help="seconds kept free before the deadline")

    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=None, help="default = rank (scale 1.0)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-final-frac", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lora-plus-ratio", type=float, default=1.0, help="learning-rate multiplier for LoRA up matrices")
    p.add_argument("--ema", type=float, default=0.99)
    p.add_argument("--include", default=None, help="regex on module names (relative to diffusion_model)")
    p.add_argument("--exclude", default=None)
    p.add_argument("--direct-include", default=None, help="regex for linear layers trained/exported as full direct deltas")
    p.add_argument("--empty-prompt-frac", type=float, default=0.5)
    p.add_argument("--paired-prompts", action="store_true", help="backprop captioned and empty losses on the same noise")
    p.add_argument("--balanced-prompts", action="store_true", help="alternate captioned and empty prompts instead of random empty_prompt_frac")
    p.add_argument("--case-cycle", action="store_true", help="cycle every image x sigma-band pair before repeating")
    p.add_argument("--te-lora", action="store_true", help="also train a LoRA on the text encoder (same rank/alpha; exported as text_encoders.* keys)")
    p.add_argument("--te-include", default=r"layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)")
    p.add_argument("--te-exclude", default=None)
    p.add_argument("--te-lr-mult", type=float, default=0.3,
                   help="text-encoder LoRA learning-rate multiplier")
    p.add_argument("--retrain-all", action="store_true", help="small-data phase 2: retrain on all images for s*·n_all/n_train steps and ship blind")
    p.add_argument("--retrain-scale", type=float, default=1.0, help="multiplier on the phase-2 step count")
    p.add_argument("--retrain-max-frac", type=float, default=0.7, help="skip phase 2 if s* is beyond this fraction of phase-1 steps (still improving)")
    p.add_argument("--retrain-phase1-frac", type=float, default=0.55, help="fraction of post-baseline budget reserved for phase-1 selection")
    p.add_argument("--retrain-max-images", type=int, default=12, help="disable retrain-all above this dataset size")
    p.add_argument("--band-power", type=float, default=0.0, help="loss weight sigma^-P per band (0 = evaluator-identical)")
    p.add_argument("--edge-band-count", type=int, default=0,
                   help="number of lowest and highest sigma bands to oversample; 0 disables")
    p.add_argument("--edge-band-repeat", type=int, default=1,
                   help="training exposure multiplier for selected edge bands")
    p.add_argument("--late-frac", type=float, default=0.0, help="final fraction of the budget that samples only --late-bands lowest-sigma bands")
    p.add_argument("--late-bands", type=int, default=16)
    p.add_argument("--bands", type=int, default=16)
    p.add_argument("--holdout-frac", type=float, default=0.1)
    p.add_argument("--holdout-min", type=int, default=3)
    p.add_argument("--holdout-noises", type=int, default=1)
    p.add_argument("--holdout-noise-batch", type=int, default=2, help="noise cases per holdout forward; evaluator uses 2")
    p.add_argument("--holdout-in-train", action="store_true")
    p.add_argument("--holdout-folds", type=int, default=0, help="deterministic K-fold split; 0 uses frac/min")
    p.add_argument("--holdout-fold", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=150)
    p.add_argument("--select-metric", default="mean", choices=["mean", "worst", "p75"],
                   help="holdout aggregate for checkpoint picking; official score is always the mean")
    p.add_argument("--soup", type=int, default=1, help="number of sequentially trained members averaged at the end")
    p.add_argument("--delta-average", type=int, default=1, help="exactly average the latest N evaluated effective deltas")
    p.add_argument("--hflip", action="store_true", help="add mirrored copies of the training images")
    p.add_argument("--crop-aug", type=int, default=0, help="zoom-crop copies per non-holdout image")
    p.add_argument("--crop-min-scale", type=float, default=0.8)
    p.add_argument("--band-min", type=int, default=0, help="lowest sigma-band index sampled in training")
    p.add_argument("--band-max", type=int, default=16, help="exclusive highest sigma-band index sampled in training")
    p.add_argument("--multires-noise-iterations", type=int, default=0, help="pyramid-noise levels; 0 keeps evaluator IID noise")
    p.add_argument("--multires-noise-discount", type=float, default=0.3)
    p.add_argument("--optimal-noise-candidates", type=int, default=1, help="draw N noises and choose the latent-nearest; 1 is IID")
    p.add_argument("--no-ckpt", dest="ckpt", action="store_false")
    p.add_argument("--ckpt", dest="ckpt", action="store_true",
                   help="force gradient checkpointing when a task-specific recipe needs lower memory")
    p.set_defaults(ckpt=True)
    p.add_argument("--ckpt-stride", type=int, default=1,
                   help="checkpoint every Nth transformer block; 1 checkpoints all blocks")
    p.add_argument("--text-enc-dtype", default=None, choices=[None, "fp16", "bf16", "fp32"])
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    cfg = parse()
    if not 0 <= cfg.band_min < cfg.band_max <= cfg.bands:
        raise SystemExit("require 0 <= --band-min < --band-max <= --bands")
    if cfg.crop_aug and not 0.5 <= cfg.crop_min_scale < 0.95:
        raise SystemExit("--crop-min-scale must be in [0.5, 0.95)")
    if cfg.multires_noise_iterations < 0 or not 0 < cfg.multires_noise_discount <= 1:
        raise SystemExit("require nonnegative multires iterations and discount in (0, 1]")
    if cfg.optimal_noise_candidates < 1:
        raise SystemExit("--optimal-noise-candidates must be at least 1")
    if cfg.ckpt_stride < 1:
        raise SystemExit("--ckpt-stride must be at least 1")
    if cfg.te_lr_mult <= 0:
        raise SystemExit("--te-lr-mult must be positive")
    if cfg.holdout_noises < 1 or cfg.holdout_noise_batch < 1:
        raise SystemExit("holdout noise counts must be at least 1")
    if cfg.edge_band_count < 0 or cfg.edge_band_count * 2 > cfg.bands:
        raise SystemExit("--edge-band-count must be between 0 and half of --bands")
    if cfg.edge_band_repeat < 1:
        raise SystemExit("--edge-band-repeat must be at least 1")
    if cfg.delta_average < 1:
        raise SystemExit("--delta-average must be at least 1")
    if cfg.alpha is None:
        cfg.alpha = float(cfg.rank)
    if cfg.deadline_ts is None:
        if cfg.minutes is None:
            raise SystemExit("--deadline-ts or --minutes required")
        cfg.deadline_ts = time.time() + cfg.minutes * 60
    if cfg.text_enc_dtype is None:
        cfg.text_enc_dtype = "bf16" if cfg.family == "qwen-image" else "fp16"
    os.makedirs(cfg.out, exist_ok=True)
    json.dump(vars(cfg), open(os.path.join(cfg.out, "config.json"), "w"), indent=1)

    import torch
    from godtrainer import comfy_boot, data, engine
    from godtrainer.assets import Assets

    mm = comfy_boot.boot(cfg.comfy_root, text_enc_dtype=cfg.text_enc_dtype)
    if cfg.family == "flux":
        from godtrainer import flux_patch

        flux_patch.apply()
    if cfg.te_lora:
        from godtrainer import text_encoder_patch

        text_encoder_patch.apply()
    engine.log(f"family={cfg.family} deadline in {(cfg.deadline_ts - time.time()) / 60:.1f} min; device {mm.get_torch_device()}")

    items = data.load_dataset(cfg.dataset, cfg.trigger_word)
    engine.log(f"dataset: {len(items)} images")
    if cfg.retrain_all and len(items) > cfg.retrain_max_images:
        engine.log(f"retrain-all: disabled for {len(items)} images (limit {cfg.retrain_max_images}); using the full standard budget")
        cfg.retrain_all = False
    rng = random.Random(1234)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    if cfg.holdout_folds > 1:
        if not 0 <= cfg.holdout_fold < cfg.holdout_folds:
            raise SystemExit("--holdout-fold must be in [0, --holdout-folds)")
        hold = {v for pos, v in enumerate(idx) if pos % cfg.holdout_folds == cfg.holdout_fold}
    else:
        n_hold = min(len(items) - 1, max(cfg.holdout_min, round(cfg.holdout_frac * len(items))))
        hold = set(idx[:n_hold])
    for i, it in enumerate(items):
        it["holdout"] = i in hold
        it["aug"] = False
    if cfg.hflip:
        from PIL import ImageOps

        extra = []
        for it in items:
            if it["holdout"]:
                continue
            img = ImageOps.mirror(it["image"])
            extra.append({"name": it["name"] + "#flip", "image": img, "caption": it["caption"], "sha256": it["sha256"] + "f", "holdout": False, "aug": True})
        items += extra
        engine.log(f"hflip: +{len(extra)} images")
    if cfg.crop_aug:
        extra = []
        for it in items:
            if it["holdout"] or it.get("aug"):
                continue
            for j in range(cfg.crop_aug):
                scale = rng.uniform(cfg.crop_min_scale, 0.95)
                img = data.zoom_crop(it["image"], scale, rng.random(), rng.random())
                extra.append({"name": f"{it['name']}#crop{j}", "image": img, "caption": it["caption"],
                              "sha256": f"{it['sha256']}c{j}", "holdout": False, "aug": True})
        items += extra
        engine.log(f"zoom-crop: +{len(extra)} images, scale {cfg.crop_min_scale:.2f}-0.95")

    assets = Assets(cfg.family, cfg.model_dir, cfg.hf_cache, cfg.baked_dir, cfg.diffusion_checkpoint)

    t = time.time()
    kind, payload = assets.vae()
    enc = engine.LatentEncoder(kind, payload)
    for it in items:
        it["latent"] = enc.encode(it["image"])
        it["image"] = None
    enc.free()
    engine.log(f"latents encoded in {time.time() - t:.0f}s; shape {tuple(items[0]['latent'].shape)}")

    t = time.time()
    clip = engine.load_clip(assets.text_encoder_sds(), cfg.family)
    conds = {}
    for prompt in sorted({it["caption"] for it in items} | {""}):
        conds[prompt] = engine.encode_prompt(clip, prompt, cfg.family)
    if not cfg.te_lora:
        del clip
        clip = None
        mm.unload_all_models()
        gc.collect()
        torch.cuda.empty_cache()
    engine.log(f"{len(conds)} prompts encoded in {time.time() - t:.0f}s")

    t = time.time()
    diffusion = assets.diffusion_sd()
    if isinstance(diffusion, tuple):
        diffusion_sd, diffusion_metadata = diffusion
    else:
        diffusion_sd, diffusion_metadata = diffusion, None
    model = engine.load_diffusion_model(diffusion_sd, cfg.family, diffusion_metadata)
    assets.release()
    gc.collect()
    for it in items:
        it["scaled"] = model.model.process_latent_in(it["latent"].clone()).float().cpu()
        it["latent"] = None
    engine.log(f"diffusion model loaded in {time.time() - t:.0f}s: {type(model.model).__name__} params={sum(p.numel() for p in model.model.diffusion_model.parameters()) / 1e9:.2f}B")

    trainer = engine.Trainer(cfg, model, items, conds, cfg.out, clip=clip)
    guider = engine.make_guider(model, conds[""])
    lat0 = items[0]["scaled"]
    sigmas = torch.tensor([0.5, 0.0])
    guider.sample(torch.zeros_like(lat0), lat0, engine.TrainSampler(trainer), sigmas, disable_pbar=True, seed=42)
    engine.log("done")


if __name__ == "__main__":
    main()
