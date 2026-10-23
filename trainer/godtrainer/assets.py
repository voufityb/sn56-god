import glob
import json
import os

import torch
from safetensors.torch import load_file

FAMILIES = ("krea2", "ideogram4", "qwen-image", "z-image", "flux")


def _root_safetensors(model_dir, min_bytes=1 << 30):
    files = [f for f in glob.glob(os.path.join(model_dir, "*.safetensors")) if os.path.getsize(f) > min_bytes]
    return sorted(files, key=os.path.getsize, reverse=True)


def _shards(dir_path):
    idx = os.path.join(dir_path, "model.safetensors.index.json")
    idx2 = os.path.join(dir_path, "diffusion_pytorch_model.safetensors.index.json")
    for i in (idx, idx2):
        if os.path.exists(i):
            wm = json.load(open(i))["weight_map"]
            return [os.path.join(dir_path, f) for f in sorted(set(wm.values()))]
    return sorted(glob.glob(os.path.join(dir_path, "*.safetensors")))


def load_sharded(files):
    sd = {}
    for f in files:
        sd.update(load_file(f, device="cpu"))
    return sd


def hf_snapshot_dir(hf_cache, repo):
    flat = os.path.join(hf_cache, repo.replace("/", "--"))
    if os.path.isdir(flat) and _shards(flat):
        return flat
    snaps = glob.glob(os.path.join(hf_cache, "models--" + repo.replace("/", "--"), "snapshots", "*"))
    if snaps:
        return sorted(snaps)[-1]
    raise FileNotFoundError(f"text encoder {repo} not found under {hf_cache}")


def dequant_per_row_fp8(sd, requant_fp8=True):


    out = {}
    for k, v in sd.items():
        if k.endswith("weight_scale") or k.endswith("scale_weight"):
            continue
        s = None
        if k.endswith(".weight"):
            s = sd.get(k + "_scale")
            if s is None:
                s = sd.get(k.replace(".weight", ".scale_weight"))
        if s is not None and v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            w = v.float() * s.float().reshape(-1, *([1] * (v.dim() - 1)))
            if requant_fp8:
                w = w.to(torch.float8_e4m3fn).float()
            out[k] = w.to(torch.bfloat16)
        elif v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            out[k] = v.float().to(torch.bfloat16)
        else:
            out[k] = v
    return out


def to_bf16(sd):
    return {k: (v.to(torch.bfloat16) if v.is_floating_point() and v.dtype != torch.bfloat16 else v) for k, v in sd.items()}


def sub_dict(sd, prefix):
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


class Assets:


    def __init__(self, family, model_dir, hf_cache=None, baked_dir=None, diffusion_checkpoint=None):
        assert family in FAMILIES, family
        self.family, self.model_dir, self.hf_cache, self.baked_dir = family, model_dir, hf_cache, baked_dir
        self.diffusion_checkpoint = diffusion_checkpoint
        self._ckpt = None

    def _full_checkpoint(self):
        if self._ckpt is None:
            self._ckpt = load_sharded(_root_safetensors(self.model_dir)[:1])
        return self._ckpt

    def diffusion_sd(self):
        if self.diffusion_checkpoint:




            import comfy.utils

            return comfy.utils.load_torch_file(
                self.diffusion_checkpoint, device=torch.device("cpu"), return_metadata=True
            )
        f = self.family
        if f == "ideogram4":
            return dequant_per_row_fp8(load_sharded(_shards(os.path.join(self.model_dir, "transformer"))), requant_fp8=True)
        if f == "flux":
            return to_bf16(sub_dict(self._full_checkpoint(), "model.diffusion_model."))
        roots = _root_safetensors(self.model_dir)
        if f == "krea2":



            pref = [r for r in roots if "raw" in os.path.basename(r).lower()] or roots
            if pref:
                return to_bf16(load_sharded(pref[:1]))
            raise FileNotFoundError(f"no krea2 weights under {self.model_dir}")



        sd = load_sharded(roots[:1])
        if any(v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) for v in sd.values()):
            return sd
        return to_bf16(sd)

    def text_encoder_sds(self):
        f = self.family
        if f == "krea2":
            exact = os.path.join(self.baked_dir, "qwen3vl_4b_fp8_scaled.safetensors")
            if os.path.exists(exact):
                return [load_file(exact, device="cpu")]
            return [load_sharded(_shards(hf_snapshot_dir(self.hf_cache, "Qwen/Qwen3-VL-4B-Instruct")))]
        if f == "ideogram4":
            exact = os.path.join(self.baked_dir, "qwen3vl_8b_fp8_scaled.safetensors")
            if os.path.exists(exact):
                return [load_file(exact, device="cpu")]
            return [load_sharded(_shards(hf_snapshot_dir(self.hf_cache, "Qwen/Qwen3-VL-8B-Instruct")))]
        if f in ("qwen-image", "z-image"):
            return [load_sharded(_shards(os.path.join(self.model_dir, "text_encoder")))]
        ck = self._full_checkpoint()
        return [sub_dict(ck, "text_encoders.clip_l.transformer."), sub_dict(ck, "text_encoders.t5xxl.transformer.")]

    def vae(self):

        f = self.family
        if f == "z-image":
            return "diffusers", os.path.join(self.model_dir, "vae")
        if f == "flux":
            return "comfy", sub_dict(self._full_checkpoint(), "vae.")
        name = "flux2-vae.safetensors" if f == "ideogram4" else "qwen_image_vae.safetensors"
        return "comfy", load_file(os.path.join(self.baked_dir, name), device="cpu")

    def release(self):
        self._ckpt = None
