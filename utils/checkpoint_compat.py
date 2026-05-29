# SPDX-License-Identifier: MIT
from __future__ import annotations

import os
import zipfile


HISTORICAL_PRE_FILM = "historical_pre_film"
LEGACY_BND_POST_FILM = "legacy_bnd_post_film"
AUTO_M2_DECODER_STYLE = "auto"
AUTO_SDE_CONTROL_STYLE = "auto"
RAW_HISTORICAL_SDE_CONTROL = "raw_historical"
ENCODED_SDE_CONTROL = "encoded"


def detect_cogflow_m2_decoder_style_from_checkpoint(ckpt_path: str | None) -> str | None:
    if not ckpt_path or not os.path.exists(ckpt_path):
        return None

    try:
        if zipfile.is_zipfile(ckpt_path):
            with zipfile.ZipFile(ckpt_path, "r") as zf:
                pkl_name = next((name for name in zf.namelist() if name.endswith("/data.pkl")), None)
                if pkl_name is None:
                    return None
                raw = zf.read(pkl_name)
        else:
            with open(ckpt_path, "rb") as f:
                raw = f.read()
    except OSError:
        return None

    if b"temporal_readout_mlp" in raw or b"motion_decoder.self_attn_K.0.mix.0.weight" in raw:
        return LEGACY_BND_POST_FILM
    if b"motion_decoder.self_attn_K.0.self_attn.in_proj_weight" in raw:
        return HISTORICAL_PRE_FILM
    return None


def resolve_cogflow_m2_decoder_style(
    *,
    explicit_style: str | None,
    ckpt_path: str | None,
) -> tuple[str, str]:
    style = explicit_style or AUTO_M2_DECODER_STYLE
    if style != AUTO_M2_DECODER_STYLE:
        return style, "explicit"

    detected = detect_cogflow_m2_decoder_style_from_checkpoint(ckpt_path)
    if detected is not None:
        return detected, "checkpoint"

    return HISTORICAL_PRE_FILM, "default"


def configure_cogflow_m2_decoder_style(
    cfg,
    *,
    explicit_style: str | None = None,
    ckpt_path: str | None = None,
) -> tuple[str, str]:
    model_cfg = getattr(cfg, "MODEL", None)
    if model_cfg is None:
        return HISTORICAL_PRE_FILM, "default"

    if explicit_style is None:
        explicit_style = model_cfg.get("M2_DECODER_STYLE", AUTO_M2_DECODER_STYLE)
    if ckpt_path is None:
        ckpt_path = cfg.get("ckpt_path", None)

    style, source = resolve_cogflow_m2_decoder_style(
        explicit_style=explicit_style,
        ckpt_path=ckpt_path,
    )
    model_cfg.M2_DECODER_STYLE = style
    cfg.m2_decoder_style = style
    cfg.m2_decoder_style_source = source
    return style, source


def resolve_cogflow_sde_control_style(
    *,
    explicit_style: str | None,
    decoder_style: str | None,
) -> tuple[str, str]:
    style = explicit_style or AUTO_SDE_CONTROL_STYLE
    if style != AUTO_SDE_CONTROL_STYLE:
        return style, "explicit"

    if decoder_style == LEGACY_BND_POST_FILM:
        return ENCODED_SDE_CONTROL, "decoder_style"
    return RAW_HISTORICAL_SDE_CONTROL, "decoder_style"


def configure_cogflow_sde_control_style(
    cfg,
    *,
    explicit_style: str | None = None,
) -> tuple[str, str]:
    model_cfg = getattr(cfg, "MODEL", None)
    if model_cfg is None:
        return RAW_HISTORICAL_SDE_CONTROL, "default"

    if explicit_style is None:
        explicit_style = model_cfg.get("SDE_CONTROL_STYLE", AUTO_SDE_CONTROL_STYLE)

    decoder_style = model_cfg.get(
        "M2_DECODER_STYLE",
        getattr(cfg, "m2_decoder_style", HISTORICAL_PRE_FILM),
    )
    style, source = resolve_cogflow_sde_control_style(
        explicit_style=explicit_style,
        decoder_style=decoder_style,
    )
    model_cfg.SDE_CONTROL_STYLE = style
    cfg.sde_control_style = style
    cfg.sde_control_style_source = source
    return style, source
