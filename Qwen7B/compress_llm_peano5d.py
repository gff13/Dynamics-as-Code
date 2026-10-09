#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Peano-5D LLM compression — MaxCover policy + lm_head tie alias (hcstyle_tie reporting).

MaxCover (same as PCG / Hilbert / Snake / Zorder maxcover):
  1) Default compress model.embed_tokens.weight (--skip-embedding to skip).
  2) Compress all bias tensors (non-linear path).
  3) Do not skip numel < 10; tiny tensors are also encoded.
  4) When lm_head.weight is tied to embed_tokens: skip lm_head in loop; alias after embed.

Backend: spacefill_peano5D.py (unchanged; loaded via importlib).
  order=4 → n=3^4=81 per dim, S=n^5=81^5 samples, dim=5.
  --embedmem loads spacefill_peano5D_embedmem.py (peak-memory only; same encode API).
  VirtualPeanoBase5D templates are passed through aux without .to(device).
Resume supported.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

EMBED_KEY = "model.embed_tokens.weight"
HEAD_KEY = "lm_head.weight"

CHECKPOINT_NAME = "compress_checkpoint.pt"
CHECKPOINT_META_NAME = "compress_checkpoint_meta.json"

PEANO_ORDER = 4
PEANO_N = 3 ** PEANO_ORDER  # 81
PEANO_S = PEANO_N ** 5
PEANO_DIM = 5


def get_dir_size(path: Path) -> int:
    total = 0
    if path.is_file():
        return path.stat().st_size
    for root, _, files in os.walk(path):
        for fname in files:
            fp = Path(root) / fname
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total


def tensor_storage_bytes(t: torch.Tensor) -> int:
    return t.untyped_storage().nbytes()


def tensor_dict_storage_bytes(d: Dict[str, Any]) -> int:
    n = 0
    for v in d.values():
        if isinstance(v, torch.Tensor):
            n += tensor_storage_bytes(v)
    return n


def tensor_dict_storage_bytes_dedupe(d: Dict[str, Any]) -> int:
    seen: Set[int] = set()
    total = 0
    for v in d.values():
        if not isinstance(v, torch.Tensor):
            continue
        st = v.untyped_storage()
        sid = id(st)
        if sid in seen:
            continue
        seen.add(sid)
        total += st.nbytes()
    return total


def safetensors_size_bytes(model_path: Path) -> Optional[int]:
    p = model_path / "model.safetensors"
    if p.is_file():
        return p.stat().st_size
    idx = model_path / "model.safetensors.index.json"
    if idx.is_file():
        total = 0
        for shard in model_path.glob("model-*.safetensors"):
            total += shard.stat().st_size
        return total if total else None
    return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_peano5d_dir(hc_root: Path, prefer_embedmem: bool = False) -> Path:
    names = (
        ("spacefill_peano5D_embedmem.py",)
        if prefer_embedmem
        else ("spacefill_peano5D.py",)
    )
    candidates = [
        Path("/home/ET/fgao/HyperCompression/resnet18/Peano"),
        Path("/home/ET/fgao/Dynamics_as_Code/Resnet18/Peano"),
        hc_root / "resnet18" / "Peano",
        hc_root / "Resnet18" / "Peano",
    ]
    for d in candidates:
        for name in names:
            if d.is_dir() and (d / name).is_file():
                return d
    if prefer_embedmem:
        return resolve_peano5d_dir(hc_root, prefer_embedmem=False)
    raise FileNotFoundError(
        "spacefill_peano5D.py not found under: " + str([str(c) for c in candidates])
    )


def load_peano5d_module(hc_root: Path, embedmem: bool = False):
    peano_dir = resolve_peano5d_dir(hc_root, prefer_embedmem=embedmem)
    fname = "spacefill_peano5D_embedmem.py" if embedmem else "spacefill_peano5D.py"
    path = peano_dir / fname
    if not path.is_file():
        raise FileNotFoundError(f"Missing Peano5D backend: {path}")
    if str(peano_dir) not in sys.path:
        sys.path.insert(0, str(peano_dir))
    mod_name = "spacefill_peano5d_maxcover_llm_embedmem" if embedmem else "spacefill_peano5d_maxcover_llm"
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, peano_dir, path.name


def _base_samples_for_device(base_samples: Any, device: str) -> Any:
    """VirtualPeanoBase5D must not receive .to(device); real tensors may."""
    if getattr(base_samples, "virtual", False):
        return base_samples
    if isinstance(base_samples, torch.Tensor):
        return base_samples.to(device)
    return base_samples


def lm_head_tied_embed(state_dict: Dict[str, torch.Tensor]) -> bool:
    embed = state_dict.get(EMBED_KEY)
    head = state_dict.get(HEAD_KEY)
    if embed is None or head is None:
        return False
    try:
        return embed.data_ptr() == head.data_ptr()
    except RuntimeError:
        return False


def alias_lm_head_to_embed(encoded_dict: Dict[str, Any], back_dict: Dict[str, torch.Tensor]) -> None:
    if HEAD_KEY in back_dict:
        return
    if EMBED_KEY not in back_dict:
        raise RuntimeError(f"Cannot alias {HEAD_KEY}: {EMBED_KEY} missing from back_dict")
    prefix_e = EMBED_KEY + "."
    prefix_h = HEAD_KEY + "."
    for k in list(encoded_dict.keys()):
        if not isinstance(k, str) or not k.startswith(prefix_e):
            continue
        suffix = k[len(prefix_e) :]
        nk = prefix_h + suffix
        if nk not in encoded_dict:
            encoded_dict[nk] = encoded_dict[k]
    back_dict[HEAD_KEY] = back_dict[EMBED_KEY]


def build_ready2encode_maxcover(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    param_names: set,
    skip_embedding: bool,
    skip_extra: List[str],
) -> Tuple[List[List[Any]], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    linear_weights = set()
    for mod_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            key = f"{mod_name}.weight" if mod_name else "weight"
            if key in state_dict:
                linear_weights.add(key)

    ready2encode: List[List[Any]] = []
    encoded_direct: Dict[str, torch.Tensor] = {}
    back_direct: Dict[str, torch.Tensor] = {}

    skip_prefixes = tuple(skip_extra)
    tied = lm_head_tied_embed(state_dict)

    for name, tensor in state_dict.items():
        if name not in param_names:
            if name == HEAD_KEY and tied and not skip_embedding:
                continue
            encoded_direct[name + ".origin_param"] = tensor
            encoded_direct[name + ".load_type"] = torch.tensor([0], dtype=torch.uint8)
            back_direct[name] = tensor.contiguous()
            continue

        if skip_embedding and "embed_tokens" in name:
            encoded_direct[name + ".origin_param"] = tensor
            encoded_direct[name + ".load_type"] = torch.tensor([0], dtype=torch.uint8)
            back_direct[name] = tensor.contiguous()
            continue

        if any(name.startswith(p) for p in skip_prefixes):
            encoded_direct[name + ".origin_param"] = tensor
            encoded_direct[name + ".load_type"] = torch.tensor([0], dtype=torch.uint8)
            back_direct[name] = tensor.contiguous()
            continue

        if name in linear_weights and name.endswith(".weight"):
            ready2encode.append([name, tensor, "linear"])
            continue

        ready2encode.append([name, tensor, "non-linear"])

    return ready2encode, encoded_direct, back_direct


def merge_encode_item_into_payload(
    encoded_dict: Dict[str, Any],
    back_dict: Dict[str, torch.Tensor],
    item: Dict[str, Any],
) -> None:
    tensor_name = item["tensor_name"]
    load_type = item["load_type"]
    if isinstance(load_type, torch.Tensor):
        load_type = int(load_type.item())
    else:
        load_type = int(load_type)

    if load_type == 0:
        lt = item["load_type"].to("cpu") if isinstance(item["load_type"], torch.Tensor) else torch.tensor(
            [0], dtype=torch.uint8
        )
        origin = item["origin_param"].to("cpu").contiguous()
        encoded_dict[tensor_name + ".load_type"] = lt
        encoded_dict[tensor_name + ".origin_param"] = origin
        back_dict[tensor_name] = origin
    elif load_type in (1, 2):
        back_dict[tensor_name] = item["back_tensor"].contiguous().to("cpu")
        lt = item["load_type"].to("cpu") if isinstance(item["load_type"], torch.Tensor) else torch.tensor(
            [load_type], dtype=torch.uint8
        )
        encoded_dict[tensor_name + ".load_type"] = lt
        encoded_dict[tensor_name + ".encoded_index"] = item["encoded_index"].to("cpu")
        encoded_dict[tensor_name + ".if_padding"] = item["if_padding"].to("cpu")
        encoded_dict[tensor_name + ".center_node"] = item["center_node"].to("cpu")
        encoded_dict[tensor_name + ".farthest_node"] = item["farthest_node"].to("cpu")
        encoded_dict[tensor_name + ".U"] = item["U"].to("cpu")
        encoded_dict[tensor_name + ".K"] = item["K"].to("cpu")
        encoded_dict[tensor_name + ".original_shape"] = item["original_shape"].to("cpu")
        encoded_dict[tensor_name + ".padding_bits"] = item["padding_bits"].to("cpu")
        encoded_dict[tensor_name + ".uint_i"] = item["uint_i"].to("cpu")
    else:
        raise ValueError(f"Unknown load_type={load_type} for {tensor_name}")


def estimate_tensor_payload_bytes(encoded_dict: Dict[str, Any], tensor_name: str) -> int:
    prefix = tensor_name + "."
    total = 0
    for k, v in encoded_dict.items():
        if isinstance(k, str) and k.startswith(prefix) and isinstance(v, torch.Tensor):
            total += tensor_storage_bytes(v)
    return total


def build_peano_globals_and_aux(
    *,
    rect_l: float,
    order: int,
    class_max: int,
    loss_max: float,
    loss_hope: float,
    stop_mae: float,
    get_Try_peano_torch_base,
    template_device: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Any]]:
    """Replicate compress_params template setup from spacefill_peano5D.py."""
    n = 3 ** order
    s = n ** 5
    print(f"\n生成 Peano-5D 基础采样点模板: order={order}, n={n}, S=n^5={s:,}, dim=5 …")
    print(f"  rect_l={rect_l}, class_max={class_max}, stop_mae={stop_mae}")
    t0 = time.perf_counter()
    base_samples = get_Try_peano_torch_base(order, rect_l, template_device)
    elapsed = time.perf_counter() - t0
    if getattr(base_samples, "virtual", False):
        print(
            f"  base_samples: VirtualPeanoBase5D (num_inner={base_samples.num_inner:,}), "
            f"elapsed {elapsed:.2f}s"
        )
    elif isinstance(base_samples, torch.Tensor):
        print(f"  base_samples shape={tuple(base_samples.shape)}, elapsed {elapsed:.2f}s")
        if base_samples.ndim != 2 or base_samples.shape[1] != 5:
            raise RuntimeError(
                f"Peano5D base_samples must be (S,5); got shape={tuple(base_samples.shape)}"
            )
        if int(base_samples.shape[0]) != int(s):
            raise RuntimeError(
                f"Peano5D base_samples rows ({base_samples.shape[0]}) != S ({s})"
            )
    else:
        raise RuntimeError(f"Unexpected base_samples type: {type(base_samples)}")

    stop_threshold: List[Any] = [True, torch.tensor([stop_mae], dtype=torch.float32)]

    peano_globals: Dict[str, Any] = {
        "rect_l": torch.tensor(rect_l, dtype=torch.float32),
        "order": torch.tensor(order, dtype=torch.int64),
        "dim": torch.tensor(PEANO_DIM, dtype=torch.int32),
    }

    aux_template: Dict[str, Any] = {
        "rect_l": torch.tensor([rect_l], dtype=torch.float32),
        "order": torch.tensor([float(order)], dtype=torch.float32),
        "class_max": torch.tensor([class_max], dtype=torch.float32),
        "loss_max": torch.tensor([loss_max], dtype=torch.float32),
        "loss_hope": torch.tensor([loss_hope], dtype=torch.float32),
        "stop_threshold": stop_threshold,
        "base_samples": base_samples,
    }
    return peano_globals, aux_template, stop_threshold


def compress_one_tensor_peano5d(
    tensor_name: str,
    tensor: torch.Tensor,
    kind: str,
    aux_template: Dict[str, Any],
    stop_threshold: List[Any],
    encode_tensor_torch_version,
    device: str,
    index: int,
) -> Dict[str, Any]:
    tensor_gpu = tensor.to(device, non_blocking=True).float()
    triple = [tensor_name, tensor_gpu, kind]

    aux = {
        "rect_l": aux_template["rect_l"].to(device),
        "order": aux_template["order"].to(device),
        "class_max": aux_template["class_max"].to(device),
        "loss_max": aux_template["loss_max"].to(device),
        "loss_hope": aux_template["loss_hope"].to(device),
        "stop_threshold": [stop_threshold[0], stop_threshold[1].clone().to(device)],
        "base_samples": _base_samples_for_device(aux_template["base_samples"], device),
    }

    try:
        item = encode_tensor_torch_version(triple, aux, device, index)
    finally:
        del tensor_gpu, triple, aux
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    lt = item["load_type"]
    load_type = int(lt.item()) if isinstance(lt, torch.Tensor) else int(lt)

    if load_type == 0:
        origin_cpu = tensor.detach().to("cpu").contiguous()
        mae_val = float(item.get("mae", 0.0)) if "mae" in item else None
        return {
            "tensor_name": tensor_name,
            "load_type": torch.tensor([0], dtype=torch.uint8),
            "origin_param": origin_cpu,
            "status": "fallback",
            "mae": mae_val,
            "max_abs_error": None,
        }

    back = item["back_tensor"].to("cpu").float()
    orig = tensor.detach().to("cpu").float()
    diff = (back - orig).abs()
    mae = float(diff.mean().item())
    max_abs = float(diff.max().item())

    item["status"] = "compressed"
    item["mae"] = mae
    item["max_abs_error"] = max_abs
    item["back_tensor"] = back.to(orig.dtype) if orig.dtype != torch.float32 else back
    return item


def reconstruction_check_summary(
    state_dict: Dict[str, torch.Tensor],
    back_dict: Dict[str, torch.Tensor],
    per_tensor: List[Dict[str, Any]],
) -> Dict[str, Any]:
    orig_keys = set(state_dict.keys())
    recon_keys = set(back_dict.keys())
    missing = sorted(orig_keys - recon_keys)
    unexpected = sorted(recon_keys - orig_keys)
    shape_mismatches: List[Dict[str, Any]] = []

    for k in sorted(orig_keys & recon_keys):
        if tuple(state_dict[k].shape) != tuple(back_dict[k].shape):
            shape_mismatches.append(
                {"name": k, "original_shape": list(state_dict[k].shape), "reconstructed_shape": list(back_dict[k].shape)}
            )

    tied_mae_ok = True
    if HEAD_KEY in back_dict and EMBED_KEY in back_dict:
        try:
            tied_mae_ok = bool(torch.allclose(back_dict[HEAD_KEY], back_dict[EMBED_KEY], rtol=0, atol=0))
        except Exception:
            tied_mae_ok = False

    compressed_maes = [t["mae"] for t in per_tensor if t.get("status") == "compressed" and t.get("mae") is not None]
    return {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
        "lm_head_alias_matches_embed": tied_mae_ok if HEAD_KEY in state_dict else None,
        "compressed_tensor_count": len(compressed_maes),
        "compressed_tensor_average_mae": float(np.mean(compressed_maes)) if compressed_maes else None,
        "compressed_tensor_max_mae": float(np.max(compressed_maes)) if compressed_maes else None,
        "ok": not missing and not unexpected and not shape_mismatches and tied_mae_ok,
    }


def save_compress_checkpoint(
    out_dir: Path,
    encoded_dict: Dict[str, Any],
    back_dict: Dict[str, torch.Tensor],
    per_tensor: List[Dict[str, Any]],
    encode_idx: int,
) -> None:
    torch.save(
        {
            "encoded_dict": encoded_dict,
            "back_dict": back_dict,
            "per_tensor": per_tensor,
            "encode_idx": encode_idx,
        },
        out_dir / CHECKPOINT_NAME,
    )
    meta = {
        "encode_idx": encode_idx,
        "num_completed": len(per_tensor),
        "last_tensor": per_tensor[-1]["tensor_name"] if per_tensor else None,
    }
    with open(out_dir / CHECKPOINT_META_NAME, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def load_compress_checkpoint(out_dir: Path) -> Optional[Dict[str, Any]]:
    path = out_dir / CHECKPOINT_NAME
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def clear_compress_checkpoint(out_dir: Path) -> None:
    for name in (CHECKPOINT_NAME, CHECKPOINT_META_NAME):
        p = out_dir / name
        if p.exists():
            p.unlink()


def clone_payload_dict_for_disk(encoded_dict: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in encoded_dict.items():
        if (
            isinstance(k, str)
            and k.endswith(".origin_param")
            and isinstance(v, torch.Tensor)
            and v.dtype == torch.float32
        ):
            out[k] = v.detach().to(torch.float16).contiguous()
        else:
            out[k] = v
    return out


def default_output_dir(model_path: Path, embedmem: bool) -> Path:
    suffix = "compressed_peano5d_maxcover_3p4_embedmem_run" if embedmem else "compressed_peano5d_maxcover_3p4_run"
    return model_path / suffix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Peano-5D (order=4, n=81, S=81^5, dim=5) LLM compression — MaxCover + lm_head tie alias"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/home/ET/fgao/HyperCompression/models_7B/Qwen1.5-7B"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: {model_path}/compressed_peano5d_maxcover_3p4_run (or _embedmem_run with --embedmem)",
    )
    parser.add_argument(
        "--hc-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="HyperCompression repo root (for resnet18/Peano backend)",
    )
    parser.add_argument("--order", type=int, default=PEANO_ORDER, help="Peano order (must be 4)")
    parser.add_argument("--rect-l", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--class-max", type=int, default=6)
    parser.add_argument("--loss-max", type=float, default=0.002)
    parser.add_argument("--loss-hope", type=float, default=0.001)
    parser.add_argument("--stop-mae", type=float, default=0.006)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--max-tensors", type=int, default=0)
    parser.add_argument("--skip-embedding", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--embedmem",
        action="store_true",
        default=False,
        help="Load spacefill_peano5D_embedmem.py (peak-memory backend; same encode API)",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = default_output_dir(args.model_path, args.embedmem)
    return args


def main() -> None:
    args = parse_args()

    if args.order != PEANO_ORDER:
        raise ValueError(f"Peano-5D MaxCover requires order={PEANO_ORDER}, got order={args.order}")

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    peano_mod, peano_dir, backend_name = load_peano5d_module(args.hc_root, embedmem=args.embedmem)
    encode_tensor_torch_version = peano_mod.encode_tensor_torch_version
    get_Try_peano_torch_base = peano_mod.get_Try_peano_torch_base

    n = PEANO_N
    s = PEANO_S

    run_config = {
        "script": "compress_llm_peano5d_maxcover.py",
        "policy": "maxcover_with_lm_head_tie_alias",
        "method": "Peano5D",
        "dim": PEANO_DIM,
        "order": args.order,
        "n": n,
        "S": s,
        "S_label": f"{n}^5",
        "backend": backend_name,
        "backend_dir": str(peano_dir),
        "embedmem": args.embedmem,
        "model_path": str(args.model_path),
        "output_dir": str(out_dir),
        "rect_l": args.rect_l,
        "seed": args.seed,
        "class_max": args.class_max,
        "loss_max": args.loss_max,
        "loss_hope": args.loss_hope,
        "stop_mae": args.stop_mae,
        "num_gpus": args.num_gpus,
        "skip_embedding": args.skip_embedding,
        "alias_lm_head_when_tied": not args.skip_embedding,
        "resume": args.resume,
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    safetensors_bytes = safetensors_size_bytes(args.model_path)

    print("=" * 72)
    print("[Peano-5D maxcover] configure")
    print("=" * 72)
    for k in (
        "method", "dim", "order", "n", "S", "S_label", "backend", "backend_dir", "embedmem",
        "rect_l", "seed", "class_max", "loss_max", "loss_hope", "stop_mae",
        "num_gpus", "skip_embedding", "alias_lm_head_when_tied", "resume",
        "model_path", "output_dir",
    ):
        print(f"  {k}: {run_config[k]}")
    print("=" * 72)
    print("[Peano-5D maxcover] embed+bias+small tensors; tie lm_head alias")
    print("=" * 72)
    print("[1/6] Model file sizes …")
    print(f"  path: {args.model_path}")
    if safetensors_bytes is not None:
        print(f"  model.safetensors bytes: {safetensors_bytes} ({safetensors_bytes / 1e9:.4f} GB)")

    print("=" * 72)
    print("[2/6] Load model to CPU …")
    from transformers import AutoModelForCausalLM

    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    state_dict = model.state_dict()
    param_names = {n for n, _ in model.named_parameters()}
    tied = lm_head_tied_embed(state_dict)
    print(f"  loaded in {time.perf_counter() - t0:.2f}s")
    print(f"  state_dict keys: {len(state_dict)}")
    print(f"  lm_head tied to embed: {tied}")

    print("=" * 72)
    print("[3/6] Build MaxCover encode list …")
    ready2encode, enc_pre, back_pre = build_ready2encode_maxcover(
        model,
        state_dict,
        param_names,
        skip_embedding=args.skip_embedding,
        skip_extra=[],
    )
    del model
    import gc

    gc.collect()

    if args.max_tensors > 0:
        ready2encode = ready2encode[: args.max_tensors]
    print(f"  tensors to encode: {len(ready2encode)}")
    for triple in ready2encode:
        if "embed_tokens" in triple[0]:
            w = triple[1]
            print(
                f"  [embed] {triple[0]} shape={tuple(w.shape)} numel={w.numel():,}",
                flush=True,
            )
            break

    num_gpus = max(1, min(args.num_gpus, torch.cuda.device_count()))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for Peano-5D encoding.")
    print(f"  GPUs: {num_gpus} (cuda:0 .. cuda:{num_gpus - 1})")

    print("=" * 72)
    print(f"[4/6] Build Peano-5D template order={args.order}, n={n}, S={s:,} …")
    template_device = "cuda:0"
    peano_globals, aux_template, stop_threshold = build_peano_globals_and_aux(
        rect_l=args.rect_l,
        order=args.order,
        class_max=args.class_max,
        loss_max=args.loss_max,
        loss_hope=args.loss_hope,
        stop_mae=args.stop_mae,
        get_Try_peano_torch_base=get_Try_peano_torch_base,
        template_device=template_device,
    )

    encoded_dict: Dict[str, Any] = dict(enc_pre)
    encoded_dict.update(peano_globals)
    back_dict: Dict[str, torch.Tensor] = dict(back_pre)
    per_tensor: List[Dict[str, Any]] = []
    encode_idx = 0

    if args.resume:
        ckpt = load_compress_checkpoint(out_dir)
        if ckpt is not None:
            encoded_dict = ckpt["encoded_dict"]
            back_dict = ckpt["back_dict"]
            per_tensor = ckpt["per_tensor"]
            encode_idx = int(ckpt["encode_idx"])
            print("=" * 72)
            print(
                f"[resume] Loaded checkpoint: encode_idx={encode_idx}, "
                f"last={per_tensor[-1]['tensor_name'] if per_tensor else 'n/a'}"
            )
        else:
            print("=" * 72)
            print("[resume] No checkpoint found; starting from scratch.")

    print("=" * 72)
    print("[5/6] Encode tensors …")
    t_all = time.perf_counter()
    for i, triple in enumerate(tqdm(ready2encode, desc=f"Peano-5D maxcover order={args.order}")):
        if i < encode_idx:
            continue
        name, tensor_cpu, kind = triple
        dev = f"cuda:{i % num_gpus}"
        t_i = time.perf_counter()
        item = compress_one_tensor_peano5d(
            name,
            tensor_cpu,
            kind,
            aux_template,
            stop_threshold,
            encode_tensor_torch_version,
            dev,
            i,
        )
        dt = time.perf_counter() - t_i
        merge_encode_item_into_payload(encoded_dict, back_dict, item)

        lt = item["load_type"]
        load_type = int(lt.item()) if isinstance(lt, torch.Tensor) else int(lt)
        status = item.get("status", "fallback" if load_type == 0 else "compressed")
        per_tensor.append(
            {
                "tensor_name": name,
                "shape": list(tensor_cpu.shape),
                "numel": int(tensor_cpu.numel()),
                "status": status,
                "encode_kind": kind,
                "load_type": load_type,
                "mae": item.get("mae"),
                "max_abs_error": item.get("max_abs_error"),
                "payload_bytes": estimate_tensor_payload_bytes(encoded_dict, name),
                "device": dev,
                "seconds": round(dt, 4),
            }
        )
        encode_idx = i + 1
        save_compress_checkpoint(out_dir, encoded_dict, back_dict, per_tensor, encode_idx)

        mae_s = f"{item.get('mae'):.6f}" if item.get("mae") is not None else "n/a"
        print(
            f"  [{encode_idx}/{len(ready2encode)}] {name}  {status}  MAE={mae_s}  {dt:.2f}s",
            flush=True,
        )

        torch.cuda.empty_cache()

    if tied and not args.skip_embedding:
        print("=" * 72)
        print("[5b/6] Alias lm_head to embed …")
        alias_lm_head_to_embed(encoded_dict, back_dict)
        per_tensor.append(
            {
                "tensor_name": HEAD_KEY,
                "shape": list(state_dict[HEAD_KEY].shape),
                "numel": int(state_dict[HEAD_KEY].numel()),
                "status": "aliased_to_embed",
                "encode_kind": None,
                "load_type": int(encoded_dict.get(EMBED_KEY + ".load_type", torch.tensor([0])).item()),
                "mae": None,
                "max_abs_error": 0.0,
                "payload_bytes": 0,
                "payload_bytes_shared_with_embed": estimate_tensor_payload_bytes(encoded_dict, EMBED_KEY),
                "device": "cpu",
                "seconds": 0.0,
            }
        )

    print(f"  encoding finished in {(time.perf_counter() - t_all) / 60:.2f} min")

    num_compressed = sum(1 for t in per_tensor if t["status"] == "compressed")
    num_fallback = sum(1 for t in per_tensor if t["status"] == "fallback")
    num_aliased = sum(1 for t in per_tensor if t["status"] == "aliased_to_embed")
    total_original_numel = sum(int(t.numel()) for t in state_dict.values())
    logical_fp16_size_bytes = total_original_numel * 2
    payload_for_disk = clone_payload_dict_for_disk(encoded_dict)
    logical_payload_dedupe_bytes = tensor_dict_storage_bytes_dedupe(payload_for_disk)
    recon_check = reconstruction_check_summary(state_dict, back_dict, per_tensor)

    print("=" * 72)
    print("[6/6] Save outputs and validate …")
    payload_path = out_dir / "compressed_payload.pt"
    recon_path = out_dir / "reconstructed_state.pt"
    torch.save(payload_for_disk, payload_path)
    torch.save(back_dict, recon_path)
    clear_compress_checkpoint(out_dir)

    compressed_payload_size_bytes = payload_path.stat().st_size
    reconstructed_state_size_bytes = recon_path.stat().st_size
    file_size_cr = (
        safetensors_bytes / compressed_payload_size_bytes
        if safetensors_bytes and compressed_payload_size_bytes
        else None
    )

    compression_report = {
        "script": "compress_llm_peano5d_maxcover.py",
        "policy": "maxcover_with_lm_head_tie_alias",
        "backend": backend_name,
        "method": "Peano5D",
        "dim": PEANO_DIM,
        "embedmem": args.embedmem,
        "alias_lm_head_when_tied": tied and not args.skip_embedding,
        "lm_head_tied_to_embed": tied,
        "peano_order": args.order,
        "peano_n": n,
        "peano_S": s,
        "rect_l": args.rect_l,
        "seed": args.seed,
        "class_max": args.class_max,
        "stop_mae": args.stop_mae,
        "payload_origin_param_disk_dtype": "float16",
        "payload_origin_param_note": "fp16 only in compressed_payload.pt on save; encode/back_dict unchanged",
        "model_safetensors_bytes": safetensors_bytes,
        "compressed_payload_size_bytes": compressed_payload_size_bytes,
        "reconstructed_state_size_bytes": reconstructed_state_size_bytes,
        "compression_ratio_file_size_safetensors_over_payload": round(file_size_cr, 6) if file_size_cr else None,
        "logical_fp16_size_bytes": logical_fp16_size_bytes,
        "logical_payload_tensor_bytes_dedupe_storage": logical_payload_dedupe_bytes,
        "compression_ratio_logical_fp16_over_payload_dedupe": round(
            logical_fp16_size_bytes / logical_payload_dedupe_bytes, 6
        )
        if logical_payload_dedupe_bytes
        else None,
        "num_tensors_to_encode": len(ready2encode),
        "num_compressed_tensors": num_compressed,
        "num_fallback_tensors": num_fallback,
        "num_aliased_tensors": num_aliased,
        "reconstruction_check": recon_check,
        "per_tensor": per_tensor,
    }
    with open(out_dir / "compression_report.json", "w", encoding="utf-8") as f:
        json.dump(compression_report, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("Peano-5D maxcover summary")
    print(f"  payload: {compressed_payload_size_bytes / 1e9:.4f} GB  ->  {payload_path}")
    print(f"  reconstructed: {reconstructed_state_size_bytes / 1e9:.4f} GB  ->  {recon_path}")
    if file_size_cr:
        print(f"  safetensors / payload: {file_size_cr:.4f}x")
    if logical_payload_dedupe_bytes:
        print(
            f"  logical fp16 / payload (dedupe): "
            f"{logical_fp16_size_bytes / logical_payload_dedupe_bytes:.4f}x"
        )
    print(f"  compressed / fallback / aliased: {num_compressed} / {num_fallback} / {num_aliased}")
    print(f"  reconstruction ok: {recon_check['ok']}")


if __name__ == "__main__":
    main()
