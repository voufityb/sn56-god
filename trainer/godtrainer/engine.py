import copy
import gc
import hashlib
import json
import math
import os
import random
import re
import time

import numpy as np
import torch
from safetensors.torch import save_file

from . import assets as A

LOG = print


def logs_enabled():
    return os.environ.get("GOD_TRAIN_LOGS", "").strip().lower() in {"1", "true", "yes", "on"}


def log(*a):
    if not logs_enabled():
        return
    LOG(time.strftime("%H:%M:%S"), *a, flush=True)


def log_error(*a):
    LOG(time.strftime("%H:%M:%S"), *a, flush=True)



def load_diffusion_model(sd, family, metadata=None):
    import comfy.sd
    import comfy.model_sampling

    fp8 = any(v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) for v in sd.values())
    opts = {} if fp8 else {"dtype": torch.bfloat16}
    model = comfy.sd.load_diffusion_model_state_dict(
        sd, model_options=opts, metadata=metadata, disable_dynamic=True
    )
    if model is None:
        raise RuntimeError("Comfy could not detect the diffusion model")


    for p in model.model.parameters():
        p.requires_grad_(False)
    if family == "qwen-image":
        m = model.clone()

        class MS(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
            pass

        ms = MS(model.model.model_config)
        ms.set_parameters(shift=3.0, multiplier=1.0)
        orig = m.get_model_object("model_sampling")
        if hasattr(orig, "noise_scale"):
            ms.set_noise_scale(orig.noise_scale)
        m.add_object_patch("model_sampling", ms)
        model = m
    sampling = model.model.model_sampling
    if not isinstance(sampling, comfy.model_sampling.CONST) or getattr(sampling, "noise_scale", 1.0) != 1.0:
        raise ValueError("Model is not unit-noise rectified flow")
    return model


CLIP_TYPES = {"krea2": "KREA2", "ideogram4": "IDEOGRAM4", "qwen-image": "QWEN_IMAGE", "z-image": "LUMINA2", "flux": "FLUX"}
GUIDANCE = {"flux": 3.5}


def load_clip(sds, family):
    import comfy.sd

    return comfy.sd.load_text_encoder_state_dicts(sds, embedding_directory=None, clip_type=getattr(comfy.sd.CLIPType, CLIP_TYPES[family]),
                                                  model_options={}, disable_dynamic=True)


def encode_prompt(clip, text, family):
    cond = clip.encode_from_tokens_scheduled(clip.tokenize(text))
    g = GUIDANCE.get(family)
    if g is not None:
        cond = [[c[0], {**c[1], "guidance": g}] for c in cond]
    return cond


class LatentEncoder:
    def __init__(self, kind, payload):
        self.kind = kind
        if kind == "comfy":
            import comfy.sd

            self.vae = comfy.sd.VAE(sd=payload, dtype=torch.float32)
            self.vae.first_stage_model.eval().requires_grad_(False)
        else:
            from diffusers import AutoencoderKL

            self.vae = AutoencoderKL.from_pretrained(payload, torch_dtype=torch.float32).eval().requires_grad_(False).to("cuda")

    @torch.inference_mode()
    def encode(self, image):
        if self.kind == "comfy":
            pixels = torch.from_numpy(np.array(image.convert("RGB"), dtype=np.float32) / 255.0).unsqueeze(0)
            lat = self.vae.encode(pixels).float().cpu()
        else:
            arr = np.array(image.convert("RGB"), dtype=np.float32, copy=True) / 127.5 - 1.0
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to("cuda")
            lat = self.vae.encode(t).latent_dist.mode().float().cpu()
        if not torch.isfinite(lat).all():
            raise ValueError("non-finite latent")
        return lat

    def free(self):
        import comfy.model_management as mm

        self.vae = None
        mm.unload_all_models()
        gc.collect()
        torch.cuda.empty_cache()



class LoraWrapper:


    def __init__(self, up=None, down=None, scale=1.0, direct=None):
        self.up, self.down, self.scale, self.direct, self.enabled = up, down, scale, direct, True

    def __call__(self, w):
        if not self.enabled:
            return w
        delta = self.direct.float() if self.direct is not None else (self.up.float() @ self.down.float()) * self.scale
        return (w.float() + delta.reshape(w.shape)).to(w.dtype)

    def to(self, device):
        return self


def select_targets(diffusion_model, include=None, exclude=None):
    inc = re.compile(include) if include else None
    exc = re.compile(exclude) if exclude else None
    out = []
    for name, m in diffusion_model.named_modules():
        w = getattr(m, "weight", None)



        if not (isinstance(m, torch.nn.Linear) or m.__class__.__name__ == "Linear"):
            continue
        if w is None or w.dim() != 2:
            continue
        if inc and not inc.search(name):
            continue
        if exc and exc.search(name):
            continue
        out.append((name, m))
    return out


TE_TAG = "te:"


def export_key(name):

    if name.startswith(TE_TAG):
        return "text_encoders." + name[len(TE_TAG):]
    return "diffusion_model." + name


class Lora:
    def __init__(self, targets, rank, alpha, device, seed, te_targets=(), direct_targets=()):
        self.rank, self.alpha, self.device = rank, alpha, device
        self.scale = alpha / rank
        direct_names = {name for name, _ in direct_targets}
        self.shapes = [(name, tuple(m.weight.shape), name in direct_names) for name, m in targets]
        self.shapes += [(TE_TAG + name, tuple(m.weight.shape), False) for name, m in te_targets]
        self.names = [n for n, _, _ in self.shapes]
        self.direct_names = {n for n, _, direct in self.shapes if direct}
        self.wrappers = {name: LoraWrapper(scale=self.scale) for name in self.names}
        self.reinit(seed)

    def reinit(self, seed):

        g = torch.Generator(device="cpu").manual_seed(seed)
        self.params, self.up_params, self.down_params, self.direct_params = [], [], [], []
        self.model_up_params, self.model_down_params = [], []
        self.te_up_params, self.te_down_params = [], []
        for name, (out_f, in_f), direct in self.shapes:
            w = self.wrappers[name]
            if direct:
                delta = torch.nn.Parameter(torch.zeros(out_f, in_f, device=self.device))
                self.params.append(delta)
                self.direct_params.append(delta)
                w.up, w.down, w.direct, w.scale, w.enabled = None, None, delta, 1.0, True
            else:
                down = torch.nn.Parameter((torch.randn(self.rank, in_f, generator=g) / math.sqrt(in_f)).to(self.device))
                up = torch.nn.Parameter(torch.zeros(out_f, self.rank, device=self.device))
                self.params += [up, down]
                self.up_params.append(up)
                self.down_params.append(down)
                if name.startswith(TE_TAG):
                    self.te_up_params.append(up)
                    self.te_down_params.append(down)
                else:
                    self.model_up_params.append(up)
                    self.model_down_params.append(down)
                w.up, w.down, w.direct, w.scale, w.enabled = up, down, None, self.scale, True

    def attach(self, patcher, clip_patcher=None):
        for name, w in self.wrappers.items():
            if name.startswith(TE_TAG):
                clip_patcher.add_weight_wrapper(f"{name[len(TE_TAG):]}.weight", w)
            else:
                patcher.add_weight_wrapper(f"diffusion_model.{name}.weight", w)

    def state(self):
        return {
            n: ((self.wrappers[n].direct.detach().clone(),) if n in self.direct_names else
                (self.wrappers[n].up.detach().clone(), self.wrappers[n].down.detach().clone()))
            for n in self.names
        }

    def load_state(self, st):
        with torch.no_grad():
            for n, tensors in st.items():
                if n in self.direct_names:
                    self.wrappers[n].direct.copy_(tensors[0])
                else:
                    self.wrappers[n].up.copy_(tensors[0])
                    self.wrappers[n].down.copy_(tensors[1])

    def set_enabled(self, flag):
        for w in self.wrappers.values():
            w.enabled = flag

    def set_tensors(self, st, scale=None):

        for n, tensors in st.items():
            w = self.wrappers[n]
            if n in self.direct_names:
                w.direct = tensors[0]
            else:
                w.up, w.down = tensors
            if scale is not None and n not in self.direct_names:
                w.scale = scale

    def restore(self):
        pos = 0
        for n in self.names:
            w = self.wrappers[n]
            if n in self.direct_names:
                w.direct = self.params[pos]
                pos += 1
            else:
                w.up, w.down, w.scale = self.params[pos], self.params[pos + 1], self.scale
                pos += 2

    def state_from_flat(self, tensors):
        state, pos = {}, 0
        for n in self.names:
            count = 1 if n in self.direct_names else 2
            state[n] = tuple(tensors[pos:pos + count])
            pos += count
        return state


def lora_state_dict(states, scale, member_weights=None):

    n = len(states)
    member_weights = member_weights or [1.0 / n] * n
    sd = {}
    for name in states[0]:
        k = export_key(name)
        if len(states[0][name]) == 1:
            sd[f"{k}.diff"] = sum(s[name][0].float().cpu() * w for s, w in zip(states, member_weights)).contiguous()
        else:
            ups = [s[name][0].float().cpu() * w for s, w in zip(states, member_weights)]
            downs = [s[name][1].float().cpu() for s in states]
            up, down = torch.cat(ups, 1), torch.cat(downs, 0)
            sd[f"{k}.lora_up.weight"] = up.contiguous()
            sd[f"{k}.lora_down.weight"] = down.contiguous()
            sd[f"{k}.alpha"] = torch.tensor(float(scale * down.shape[0]))
    return sd


def soup_tensors(states, device):
    n = len(states)
    return {
        name: ((sum(s[name][0] / n for s in states).to(device),) if len(states[0][name]) == 1 else
               (torch.cat([s[name][0] / n for s in states], 1).to(device),
                torch.cat([s[name][1] for s in states], 0).to(device)))
        for name in states[0]
    }



def enable_checkpointing(diffusion_model, min_len=4, stride=1):
    import torch.utils.checkpoint as cp

    wrapped = 0
    for name, m in diffusion_model.named_modules():
        if isinstance(m, torch.nn.ModuleList) and len(m) >= min_len:
            for index, blk in enumerate(m):
                if index % stride:
                    continue
                if isinstance(blk, torch.nn.ModuleList) or getattr(blk, "_god_ckpt", False):
                    continue
                orig = blk.forward

                def fwd(*a, _orig=orig, **kw):
                    if torch.is_grad_enabled():
                        return cp.checkpoint(_orig, *a, use_reentrant=False, **kw)
                    return _orig(*a, **kw)

                blk.forward = fwd
                blk._god_ckpt = True
                wrapped += 1
    return wrapped



class TrainSampler:


    def __init__(self, trainer):
        self.trainer = trainer

    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        self.trainer.run(model_wrap, extra_args)
        return latent_image


def case_seed(image_hash, band, k, master=42):
    return int.from_bytes(hashlib.sha256(f"{master}:{image_hash}:{band}:{k}".encode()).digest()[:8], "little") % (2**63 - 1)


class Trainer:
    def __init__(self, cfg, model, items, conds, out_dir, clip=None):


        self.cfg, self.model, self.items, self.conds, self.out_dir = cfg, model, items, conds, out_dir
        self.clip = clip
        self.bands = cfg.bands
        self.device = model.load_device
        self.train_items = [i for i in items if (not i["holdout"]) or cfg.holdout_in_train]
        self.holdout_items = [i for i in items if i["holdout"] and not i.get("aug")]
        self.summary = {"family": cfg.family, "steps": 0, "evals": [], "members": []}
        self.best = None
        self.base_score = None
        self.saved_score = None
        self.step_time = None
        self.holdout_time = None
        self.rng = random.Random(cfg.seed)
        self.diff = model.model.diffusion_model
        targets = select_targets(self.diff, cfg.include, cfg.exclude)
        direct_targets = select_targets(self.diff, cfg.direct_include, cfg.exclude) if cfg.direct_include else []
        known = {name for name, _ in targets}
        targets += [(name, module) for name, module in direct_targets if name not in known]
        if not targets:
            raise RuntimeError("no LoRA targets matched")
        if cfg.direct_include and not direct_targets:
            raise RuntimeError("--direct-include matched no linear layers")
        self.targets = targets
        log(f"lora targets: {len(targets)} linear layers; first={targets[0][0]} last={targets[-1][0]}")
        if direct_targets:
            direct_params = sum(module.weight.numel() for _, module in direct_targets)
            log(f"direct-delta targets: {len(direct_targets)} linear layers, {direct_params / 1e6:.1f}M params; "
                f"first={direct_targets[0][0]} last={direct_targets[-1][0]}")
        te_targets = []
        if clip is not None:
            for q in clip.cond_stage_model.parameters():
                q.requires_grad_(False)
            te_targets = select_targets(clip.cond_stage_model, cfg.te_include, cfg.te_exclude)
            if not te_targets:
                raise RuntimeError("no text-encoder LoRA targets matched")
            log(f"text-encoder lora targets: {len(te_targets)} linear layers; first={te_targets[0][0]} last={te_targets[-1][0]}")
        self.lora = Lora(targets, cfg.rank, cfg.alpha, self.device, cfg.seed, te_targets, direct_targets)
        self.lora.attach(model, clip.patcher if clip is not None else None)
        if cfg.ckpt:
            log(f"checkpointed blocks (stride={cfg.ckpt_stride}):", enable_checkpointing(self.diff, stride=cfg.ckpt_stride))


    def remaining(self):
        return self.cfg.deadline_ts - time.time()

    def processed(self, guider, item, prompt):
        import comfy.samplers
        import comfy.sampler_helpers

        key = (item["sha256"], prompt)
        if key not in self._pc:
            lat = item["scaled"].to(self.device)
            raw = self.conds[prompt] if self.clip is None else encode_prompt(self.clip, prompt, self.cfg.family)
            conds = {"positive": comfy.sampler_helpers.convert_cond(raw)}
            out = comfy.samplers.process_conds(guider.inner_model, torch.zeros_like(lat), conds, self.device, lat, None, 42)
            if self.clip is not None:
                return out
            self._pc[key] = out
        return self._pc[key]

    def forward(self, guider, extra, item, prompt, noisy, sigma):
        guider.conds = self.processed(guider, item, prompt)
        t = torch.full((noisy.shape[0],), sigma, device=self.device, dtype=torch.float32)
        return guider(noisy, t, **extra)

    def band_weight(self, band):

        P = self.cfg.band_power
        if not P:
            return 1.0
        sig = (np.arange(self.bands) + 0.5) / self.bands
        w = sig ** (-P)
        return float(w[band] / w.mean())

    def training_bands(self, band_hi):






        bands = list(range(self.cfg.band_min, band_hi))
        if self.cfg.edge_band_count and self.cfg.edge_band_repeat > 1:
            edge = set(range(self.cfg.edge_band_count))
            edge.update(range(self.bands - self.cfg.edge_band_count, self.bands))
            bands = [
                band
                for band in bands
                for _ in range(self.cfg.edge_band_repeat if band in edge else 1)
            ]
        return bands

    def loss(self, guider, extra, item, prompt, sigma, noise):
        x = item["scaled"].to(self.device)
        noisy = (1 - sigma) * x + sigma * noise
        denoised = self.forward(guider, extra, item, prompt, noisy, sigma)
        pred = (noisy.float() - denoised.float()) / sigma
        return (pred - (noise - x).float()).square().mean()

    def _aggregate_holdout(self, per_image):






        mean = float(np.mean(per_image))
        metric = getattr(self.cfg, "select_metric", "mean") or "mean"
        if metric == "mean" or len(per_image) == 1:
            return mean
        if metric == "worst":
            return float(np.max(per_image))
        if metric == "p75":
            return float(np.quantile(per_image, 0.75))
        raise ValueError(f"unknown select_metric {metric!r}")

    @torch.no_grad()
    def holdout_score(self, guider, extra):
        t0 = time.time()
        per_mode = {"text": [], "no_text": []}
        band_acc = np.zeros(self.bands)
        for item in self.holdout_items:
            x = item["scaled"].to(self.device)
            for mode in per_mode:
                prompt = item["caption"] if mode == "text" else ""
                losses = []
                for b in range(self.bands):
                    s = (b + 0.5) / self.bands
                    band_losses = []
                    for start in range(0, self.cfg.holdout_noises, self.cfg.holdout_noise_batch):
                        stop = min(self.cfg.holdout_noises, start + self.cfg.holdout_noise_batch)
                        noise = torch.cat([
                            torch.randn(x.shape, generator=torch.Generator().manual_seed(case_seed(item["sha256"], b, k)),
                                        dtype=torch.float32)
                            for k in range(start, stop)
                        ]).to(self.device)
                        noisy = (1 - s) * x + s * noise
                        den = self.forward(guider, extra, item, prompt, noisy, s)
                        pred = (noisy.float() - den.float()) / s
                        band_losses += (pred - (noise - x).float()).square().flatten(1).mean(1).tolist()
                    losses += band_losses
                    band_acc[b] += float(np.mean(band_losses))
                per_mode[mode].append(float(np.mean(losses)))

        per_image = [0.5 * t + 0.5 * n for t, n in zip(per_mode["text"], per_mode["no_text"])]
        mean_score = float(np.mean(per_image))
        score = self._aggregate_holdout(per_image)
        self.last_holdout_mean = mean_score
        self.last_holdout_per_image = per_image
        self.holdout_time = time.time() - t0
        bands = band_acc / (2 * len(self.holdout_items))
        metric = getattr(self.cfg, "select_metric", "mean") or "mean"
        extra_log = "" if metric == "mean" else f" select={metric}:{score:.5f}"
        if getattr(self, "base_bands", None) is None:
            self.base_bands = bands
            log("base per-band loss:", " ".join(f"{v:.4f}" for v in bands),
                f"| text {np.mean(per_mode['text']):.5f} no_text {np.mean(per_mode['no_text']):.5f}"
                f" | per-img {[f'{v:.4f}' for v in per_image]}{extra_log}")
        else:
            log("per-band rel%:", " ".join(f"{(v / w - 1) * 100:+.1f}" for v, w in zip(bands, self.base_bands)),
                f"| text {np.mean(per_mode['text']):.5f} no_text {np.mean(per_mode['no_text']):.5f}"
                f" | per-img {[f'{v:.4f}' for v in per_image]}{extra_log}")
        return score

    def evaluate_candidates(self, guider, extra, cands, step):

        out = []
        for tag, st, scale in cands:
            if st is not None:
                self.lora.set_tensors(st, scale)
            sc = self.holdout_score(guider, extra)
            self.lora.restore()
            out.append((sc, tag))
            rel = (sc / self.base_score - 1) * 100 if self.base_score else 0.0
            log(f"[eval step {step}] {tag}: {sc:.6f} ({rel:+.2f}% vs base) holdout_time={self.holdout_time:.0f}s")
            self.summary["evals"].append({"step": step, "tag": tag, "score": sc, "t": time.time()})
        return out

    def save(self, states, scale, score, tag):
        os.makedirs(self.out_dir, exist_ok=True)
        sd = lora_state_dict(states, scale)
        tmp = os.path.join(self.out_dir, "last.safetensors.tmp")
        save_file(sd, tmp, metadata={"godtrainer": tag})
        os.replace(tmp, os.path.join(self.out_dir, "last.safetensors"))
        self.saved_score = score
        self.summary.update(saved={"tag": tag, "score": score, "base": self.base_score, "rel_pct": (score / self.base_score - 1) * 100 if self.base_score else None})
        json.dump(self.summary, open(os.path.join(self.out_dir, "summary.json"), "w"), indent=1)
        log(f"saved {tag} score={score:.6f}")


    def run(self, guider, extra):
        cfg = self.cfg
        self._pc = {}
        t_start = time.time()
        for it in self.items:
            it["scaled"] = it["scaled"].to(self.device)

        self.base_score = self.holdout_score(guider, extra)
        log(f"base holdout score {self.base_score:.6f} (holdout {len(self.holdout_items)} imgs, {self.holdout_time:.0f}s); train imgs {len(self.train_items)}")
        self.summary["base_score"] = self.base_score
        ident = [self.lora.state()]
        self.save(ident, self.lora.scale, self.base_score, "identity")
        self.best = (self.base_score, "identity", ident, self.lora.scale)

        members = []
        n_members = max(1, cfg.soup)
        for mi in range(n_members):
            if mi > 0:
                self.lora.reinit(cfg.seed + 1000 * mi)
            soup_reserve = ((self.holdout_time or 60) * 1.3 + 15) if n_members > 1 else 0
            available = self.remaining() - cfg.save_reserve - soup_reserve


            single_retrain = cfg.retrain_all and n_members == 1
            if single_retrain and not cfg.retrain_on_plateau:
                available *= cfg.retrain_phase1_frac
            budget_end = time.time() + available / (n_members - mi)
            res = self.train_member(guider, extra, mi, budget_end,
                                    plateau_evals=cfg.retrain_on_plateau if single_retrain else 0)
            if res is None:
                break
            members.append(res)
            self.summary["members"].append({"member": mi, "score": res[0], "tag": res[1]})
            if res[0] < self.best[0]:
                self.best = (res[0], f"m{mi}:{res[1]}", [res[2]], self.lora.scale)
                self.save(self.best[2], self.best[3], self.best[0], self.best[1])
        if cfg.retrain_all and len(members) == 1 and self.holdout_items:
            self.retrain_all(guider, extra, members[0])
        if len(members) > 1 and self.remaining() > cfg.save_reserve * 0.5 + (self.holdout_time or 60) * 1.1:
            states = [m[2] for m in members]
            soup = soup_tensors(states, self.device)
            sc = self.evaluate_candidates(guider, extra, [("soup", soup, self.lora.scale)], self.summary["steps"])[0][0]
            if sc < self.best[0]:
                self.best = (sc, "soup", states, self.lora.scale)
                self.save(states, self.lora.scale, sc, "soup")
        self.summary["final"] = {"best": self.best[1], "score": self.best[0], "base": self.base_score, "rel_pct": (self.best[0] / self.base_score - 1) * 100,
                                 "wall_s": time.time() - t_start}
        json.dump(self.summary, open(os.path.join(self.out_dir, "summary.json"), "w"), indent=1)
        log("final:", json.dumps(self.summary["final"]))

    def retrain_steps(self, s_star):
        n_train = len(self.train_items)
        n_all = n_train + len(self.holdout_items)
        return int(math.ceil(s_star * n_all / n_train * self.cfg.retrain_scale))

    def _plateau_handover(self, best_step, step, plateau_evals):
        # Stop phase 1 early only when retrain_all will accept its best step and phase 2 fits in the time left;
        # otherwise keep training, so a run that never plateaus spends its budget exactly like a plain run.
        cfg = self.cfg
        if best_step is None or step - best_step < plateau_evals * cfg.eval_every:
            return False
        if best_step > cfg.retrain_max_frac * self.summary["steps"]:
            return False
        need = self.retrain_steps(best_step) * (self.step_time or 3.0) * 1.1
        return need < self.remaining() - cfg.save_reserve - 30

    def retrain_all(self, guider, extra, phase1):


        cfg = self.cfg
        score1, tag1 = phase1[0], phase1[1]
        m = re.match(r"(raw|ema|pema)@(\d+)$", tag1)
        if not m or score1 >= self.base_score:
            log("retrain-all: phase 1 produced no usable best step; skipping")
            return
        select, s_star = m.group(1), int(m.group(2))
        n_all = len(self.train_items) + len(self.holdout_items)
        steps2 = self.retrain_steps(s_star)
        est = steps2 * (self.step_time or 3.0)
        avail = self.remaining() - cfg.save_reserve - 30
        if est * 1.1 > avail:
            log(f"retrain-all: needs {est:.0f}s for {steps2} steps, only {avail:.0f}s left; keeping phase-1 model")
            self.summary["retrain_all"] = {"skipped": True, "steps": steps2, "est_s": est, "avail_s": avail}
            return
        if s_star > cfg.retrain_max_frac * self.summary["steps"]:
            log(f"retrain-all: s*={s_star} is late in the run (best still improving); keeping phase-1 model")
            self.summary["retrain_all"] = {"skipped": True, "reason": "late_best", "steps": steps2}
            return
        log(f"retrain-all: phase 2 on {n_all} images for {steps2} steps (s*={s_star}, {select}); est {est:.0f}s of {avail:.0f}s")
        self.train_items = list(self.items)
        self.lora.reinit(cfg.seed)
        res = self.train_member(guider, extra, 1, time.time() + avail, fixed_steps=steps2, select=select)
        if res is None or res[2] is None:
            return
        self.best = (score1, f"retrain-all:{res[1]}", [res[2]], self.lora.scale)
        self.save(self.best[2], self.best[3], score1, self.best[1])
        self.summary["retrain_all"] = {"skipped": False, "steps": steps2, "s_star": s_star, "select": select, "phase1_score": score1}

    def train_member(self, guider, extra, mi, budget_end, fixed_steps=None, select=None, plateau_evals=0):


        cfg = self.cfg
        lora = self.lora


        groups = []
        if lora.model_up_params:
            groups.append({"params": lora.model_up_params, "lr_mult": cfg.lora_plus_ratio})
        stable = lora.model_down_params + lora.direct_params
        if stable:
            groups.append({"params": stable, "lr_mult": 1.0})
        if lora.te_up_params:
            groups.append({"params": lora.te_up_params, "lr_mult": cfg.lora_plus_ratio * cfg.te_lr_mult})
        if lora.te_down_params:
            groups.append({"params": lora.te_down_params, "lr_mult": cfg.te_lr_mult})
        opt = torch.optim.AdamW(
            groups,
            lr=cfg.lr, betas=(0.9, 0.99),
            weight_decay=cfg.weight_decay, eps=1e-8,
        )
        ema = [p.detach().clone() for p in lora.params] if cfg.ema > 0 else None
        pema = [p.detach().clone() for p in lora.params] if cfg.power_ema > 0 else None
        best = (self.base_score, "base", lora.state())
        history = []
        t0 = time.time()
        step, order, band_perm, cases, bi = 0, [], [], [], 0
        last_eval_step = 0
        best_step, plateaued = None, False
        gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 7919 * mi)
        while True:
            now = time.time()
            est_step = self.step_time or 3.0
            if fixed_steps is not None:
                if step >= fixed_steps or now + est_step > budget_end:
                    break
                eval_due = False
                p = min(1.0, step / max(1, fixed_steps))
            else:
                eval_due = (step - last_eval_step) >= cfg.eval_every
                eval_count = 3.3 if cfg.delta_average > 1 else 2.2
                need = est_step + (self.holdout_time or 60) * eval_count + 5
                if now + need > budget_end:
                    break

                p = min(1.0, (now - t0) / max(1.0, budget_end - t0))
            warm = min(1.0, (step + 1) / max(1, cfg.warmup_steps))
            lr = cfg.lr * warm * (cfg.lr_final_frac + (1 - cfg.lr_final_frac) * 0.5 * (1 + math.cos(math.pi * p)))
            for g in opt.param_groups:
                g["lr"] = lr * g["lr_mult"]

            if cfg.case_cycle:
                band_hi = cfg.band_max
                if p >= 1 - cfg.late_frac:
                    band_hi = min(band_hi, cfg.late_bands)
                if not cases:
                    cases = [(i, b) for i in range(len(self.train_items)) for b in self.training_bands(band_hi)]
                    self.rng.shuffle(cases)
                if not cases:
                    raise RuntimeError(f"empty training band range [{cfg.band_min}, {band_hi})")
                item_i, band = cases.pop()
                item = self.train_items[item_i]
            else:
                if not order:
                    order = list(range(len(self.train_items)))
                    self.rng.shuffle(order)
                item = self.train_items[order.pop()]
                if bi >= len(band_perm):
                    band_hi = cfg.band_max
                    if p >= 1 - cfg.late_frac:
                        band_hi = min(band_hi, cfg.late_bands)
                    band_perm = self.training_bands(band_hi)
                    if not band_perm:
                        raise RuntimeError(f"empty training band range [{cfg.band_min}, {band_hi})")
                    self.rng.shuffle(band_perm)
                    bi = 0
                band = band_perm[bi]
                bi += 1
            sigma = (band + 0.5) / self.bands
            noise = torch.randn(item["scaled"].shape, generator=gen, dtype=torch.float32).to(self.device)
            if cfg.optimal_noise_candidates > 1:
                candidates = [noise]
                candidates.extend(
                    torch.randn(item["scaled"].shape, generator=gen, dtype=torch.float32).to(self.device)
                    for _ in range(cfg.optimal_noise_candidates - 1)
                )
                noise = min(
                    candidates,
                    key=lambda candidate: torch.nn.functional.mse_loss(item["scaled"].float(), candidate).item(),
                )
            if cfg.multires_noise_iterations:
                import torch.nn.functional as F

                h, w = noise.shape[-2:]
                leading = noise.shape[:-2]
                for i in range(cfg.multires_noise_iterations):
                    ratio = self.rng.random() * 2 + 2
                    lh, lw = max(1, int(h / ratio**i)), max(1, int(w / ratio**i))
                    low = torch.randn((*leading, lh, lw), generator=gen, dtype=torch.float32).to(self.device)
                    low = F.interpolate(low.reshape(-1, 1, lh, lw), size=(h, w), mode="bilinear",
                                        align_corners=False).reshape(*leading, h, w)
                    noise = noise + low * cfg.multires_noise_discount**i
                    if lh == 1 or lw == 1:
                        break
                noise = noise / noise.std()
            ts = time.time()
            if cfg.paired_prompts:
                prompts = [item["caption"], ""]
            elif getattr(cfg, "balanced_prompts", False):
                prompts = [item["caption"] if self.summary["steps"] % 2 == 0 else ""]
            else:
                prompts = ["" if self.rng.random() < cfg.empty_prompt_frac else item["caption"]]
            losses = []
            oom = False
            try:
                for prompt in prompts:
                    with torch.enable_grad():
                        mode_loss = self.loss(guider, extra, item, prompt, sigma, noise)
                        (mode_loss * (self.band_weight(band) / len(prompts))).backward()
                    losses.append(mode_loss.detach())
            except torch.cuda.OutOfMemoryError:
                if cfg.ckpt and cfg.ckpt_stride == 1:
                    raise
                oom = True
            if oom:
                # A step without full gradient checkpointing ran out of memory (a no-ckpt recipe on a bigger model or
                # on unusually long captions such as Ideogram UI-design JSON prompts): drop the step, checkpoint more
                # blocks (every second one first, then all of them) and keep training.
                mode_loss = losses = None
                opt.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                stride = 1 if cfg.ckpt else 2
                cfg.ckpt, cfg.ckpt_stride = True, stride
                self.step_time = None
                log_error(f"CUDA out of memory at m{mi} step {step}; checkpointing every {stride} block(s), newly wrapped:",
                          enable_checkpointing(self.diff, stride=stride))
                continue
            loss = torch.stack(losses).mean()
            torch.nn.utils.clip_grad_norm_(lora.params, cfg.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if ema is not None:
                with torch.no_grad():
                    for e, q in zip(ema, lora.params):
                        e.mul_(cfg.ema).add_(q.detach(), alpha=1 - cfg.ema)
            if pema is not None:
                # Power-function EMA (Karras et al. 2024): weights iterate t by t^gamma, so the averaging window is a
                # fixed fraction of the member's run rather than a fixed number of steps, with no pull toward the init.
                decay = (1 - 1 / (step + 1)) ** (cfg.power_ema + 1)
                with torch.no_grad():
                    for e, q in zip(pema, lora.params):
                        e.mul_(decay).add_(q.detach(), alpha=1 - decay)
            dt = time.time() - ts
            self.step_time = dt if self.step_time is None else 0.9 * self.step_time + 0.1 * dt
            step += 1
            self.summary["steps"] += 1
            if step % 25 == 0:
                log(f"m{mi} step {step} loss {loss.item():.4f} band {band} lr {lr:.2e} step_t {self.step_time:.2f}s left {budget_end - time.time():.0f}s")
            if fixed_steps is None and (eval_due or step == cfg.eval_every):
                last_eval_step = step
                prev = best
                best = self._select(guider, extra, lora, ema, best, step, pema=pema)
                history.append(lora.state())
                # Only the newest delta_average states are ever averaged; older GPU copies just add up per eval.
                del history[:-max(1, cfg.delta_average)]
                if best is not prev:
                    best_step = step
                if plateau_evals and self._plateau_handover(best_step, step, plateau_evals):
                    plateaued = True
                    self.summary["plateau"] = {"step": step, "best_step": best_step}
                    log(f"m{mi} holdout flat since step {best_step}; handing over to retrain-all at step {step}")
                    break
        if fixed_steps is not None:
            averaged = {"ema": ema, "pema": pema}.get(select)
            if averaged is not None:
                state = {n: tuple(q.detach().clone() for q in tensors)
                         for n, tensors in lora.state_from_flat(averaged).items()}
            else:
                state = lora.state()
            log(f"member {mi} (fixed) done: steps={step}/{fixed_steps} shipped {select}@{step} (blind)")
            return (float("nan"), f"{select}@{step}", state)

        # After a plateau handover the states past best_step are on the overfitting side; skip scoring them so
        # retrain_all receives a raw@/ema@ best it can reproduce.
        if not plateaued and step > last_eval_step and self.remaining() > cfg.save_reserve + (self.holdout_time or 60) * 2.1:
            best = self._select(guider, extra, lora, ema, best, step, pema=pema)
            history.append(lora.state())
        if (not plateaued and cfg.delta_average > 1 and len(history) >= cfg.delta_average
                and time.time() + (self.holdout_time or 60) * 1.1 < budget_end):
            count = cfg.delta_average
            averaged = soup_tensors(history[-count:], self.device)
            tag = f"deltaavg{count}@{step}"
            sc = self.evaluate_candidates(guider, extra, [(tag, averaged, lora.scale)], step)[0][0]
            if sc < best[0]:
                state = {n: tuple(q.detach().clone() for q in tensors) for n, tensors in averaged.items()}
                best = (sc, tag, state)
                if sc < self.best[0]:
                    self.best = (sc, tag, [state], lora.scale)
                    self.save([state], lora.scale, sc, tag)
        log(f"member {mi} done: steps={step} best={best[1]} {best[0]:.6f} ({(best[0] / self.base_score - 1) * 100:+.2f}% vs base)")
        return best

    def _select(self, guider, extra, lora, ema, best, step, pema=None):
        # The raw weights lost to their EMA at every eval of the A/B runs, so the wider power EMA takes their slot
        # and the eval count per checkpoint stays the same.
        cands = [("pema", lora.state_from_flat(pema), lora.scale)] if pema is not None else [("raw", None, None)]
        if ema is not None:
            ema_state = lora.state_from_flat(ema)
            cands.append(("ema", ema_state, lora.scale))
        scores = self.evaluate_candidates(guider, extra, cands, step)
        for (sc, tag), (_, st, _) in zip(scores, cands):
            if sc < best[0]:
                state = lora.state() if st is None else {n: (u.detach().clone(), d.detach().clone()) for n, (u, d) in st.items()}
                best = (sc, f"{tag}@{step}", state)
                if sc < self.best[0]:
                    self.best = (sc, best[1], [state], lora.scale)
                    self.save([state], lora.scale, sc, best[1])
        return best


def make_guider(model, cond):
    import comfy.samplers

    g = comfy.samplers.CFGGuider(model)
    g.set_conds(cond, [])
    g.set_cfg(1.0)
    return g
