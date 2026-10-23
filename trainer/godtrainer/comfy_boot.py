import os
import sys

_BOOTED = {}


def boot(comfy_root, text_enc_dtype="fp16", static_vram=True):
    if _BOOTED:
        return _BOOTED["mm"]
    comfy_root = os.path.abspath(comfy_root)
    if comfy_root not in sys.path:
        sys.path.insert(0, comfy_root)
    import comfy.cli_args as cli

    a = cli.args
    a.disable_dynamic_vram = static_vram
    a.highvram = static_vram
    a.disable_smart_memory = True
    a.bf16_text_enc = text_enc_dtype == "bf16"
    a.fp16_text_enc = text_enc_dtype == "fp16"
    a.fp32_text_enc = text_enc_dtype == "fp32"
    a.fp8_e4m3fn_text_enc = False
    a.fp8_e5m2_text_enc = False
    import comfy.model_management as mm

    mm.in_training = True
    _BOOTED["mm"] = mm
    return mm
