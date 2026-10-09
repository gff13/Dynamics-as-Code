#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QMC3D LLM compression — Hyper-Compression broad coverage + tied lm_head alias.

Same as compress_llm_qmc3d_hcstyle.py, except when embed_tokens and lm_head share
storage (tie_word_embeddings), lm_head reuses embed's compressed payload keys
instead of storing a second full copy.

Coverage policy:
  - Iterate model.state_dict() keys in order.
  - named_parameters() + numel >= 10 → attempt QMC3D encode.
  - Buffers, numel < 10, MAE > stop_mae → direct store.
  - When lm_head is tied to embed: defer lm_head, then alias after embed.

Method: spacefill_qmc3D.py (D=3, S=m=65536 Halton candidates, bases=(2,3,5)).
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


def resolve_dtype(name: str) -> torch.dtype:
    m = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in m:
        raise ValueError(f"Unsupported --dtype {name!r}; choose from {list(m)}")
    return m[name]


def resolve_qmc3d_dir(methods_root: Path) -> Path:
    candidates = [
        methods_root / "QMC3D",
        Path("/home/ET/fgao/HyperCompression/resnet18/QMC"),
        Path("/home/ET/fgao/Dynamics_as_Code/Resnet18/QMC"),
        methods_root.parent / "resnet18" / "QMC",
        methods_root.parent / "Resnet18" / "QMC",
    ]
    for d in candidates:
        if d.is_dir() and (d / "spacefill_qmc3D.py").is_file():
            return d
    raise FileNotFoundError(
        "spacefill_qmc3D.py not found under: " + str([str(c) for c in candidates])
    )


def load_qmc3d_module(methods_root: Path):
    qmc_dir = resolve_qmc3d_dir(methods_root)
    path = qmc_dir / "spacefill_qmc3D.py"
    if str(qmc_dir) not in sys.path:
        sys.path.insert(0, str(qmc_dir))
    spec = importlib.util.spec_from_file_location("spacefill_qmc3d_hcstyle_tie", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    """When tied, lm_head shares embed's compressed keys and reconstructed tensor."""
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


def collect_named_parameter_names(model: nn.Module) -> Set[str]:
    return {n for n, _ in model.named_parameters()}


def collect_linear_weight_names(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> Set[str]:
    names: Set[str] = set()
    for mod_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            key = f"{mod_name}.weight" if mod_name else "weight"
            if key in state_dict:
                names.add(key)
    return names


def infer_encode_kind(tensor_name: str, tensor: torch.Tensor, linear_weights: Set[str]) -> str:
    if tensor_name in linear_weights and tensor.ndim == 2:
        return "linear"
    return "non-linear"


def should_try_compress(
    tensor_name: str,
    tensor: torch.Tensor,
    parameter_names: Set[str],
    args: argparse.Namespace,
    compress_attempts: int,
    tied: bool,
) -> Tuple[bool, str]:
    if tensor_name not in parameter_names:
        if tied and tensor_name == HEAD_KEY and not args.skip_embedding:
            return False, "tied_alias_deferred"
        return False, "buffer"
    if tensor.numel() < 10:
        return False, "small"
    if args.max_tensors > 0 and compress_attempts >= args.max_tensors:
        return False, "debug_max_tensors"
    return True, ""


def estimate_tensor_payload_bytes(encoded_dict: Dict[str, Any], tensor_name: str) -> int:
    prefix = tensor_name + "."
    total = 0
    for k, v in encoded_dict.items():
        if isinstance(k, str) and k.startswith(prefix) and isinstance(v, torch.Tensor):
            total += tensor_storage_bytes(v)
    return total


def fallback_direct_store(
    tensor_name: str,
    tensor: torch.Tensor,
    status: str,
) -> Dict[str, Any]:
    t_cpu = tensor.detach().to("cpu").contiguous()
    return {
        "tensor_name": tensor_name,
        "load_type": torch.tensor([0], dtype=torch.uint8),
        "origin_param": t_cpu,
        "status": status,
        "mae": None,
        "max_abs_error": 0.0,
    }


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


def compress_one_tensor_qmc3d(
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
        "m": aux_template["m"].to(device),
        "bases": aux_template["bases"],
        "class_max": aux_template["class_max"].to(device),
        "loss_max": aux_template["loss_max"].to(device),
        "loss_hope": aux_template["loss_hope"].to(device),
        "stop_threshold": [stop_threshold[0], stop_threshold[1].clone().to(device)],
        "base_samples": aux_template["base_samples"].to(device),
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
                {
                    "name": k,
                    "original_shape": list(state_dict[k].shape),
                    "reconstructed_shape": list(back_dict[k].shape),
                }
            )

    compressed_maes = [
        t["mae"] for t in per_tensor if t.get("status") == "compressed" and t.get("mae") is not None
    ]
    return {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
        "compressed_tensor_count": len(compressed_maes),
        "compressed_tensor_average_mae": float(np.mean(compressed_maes)) if compressed_maes else None,
        "compressed_tensor_max_mae": float(np.max(compressed_maes)) if compressed_maes else None,
        "ok": not missing and not unexpected and not shape_mismatches,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QMC3D (S=2^16) LLM compression — HC broad coverage + lm_head tie alias"
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("/home/ET/fgao/HyperCompression/models/Qwen2.5-1.5B"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/home/ET/fgao/HyperCompression/models/qwen2.5-1.5b_qmc3d_2p16_hcstyle_tie"),
    )
    parser.add_argument(
        "--methods_root",
        type=Path,
        default=Path(__file__).resolve().parent / "methods",
        help="Directory containing methods/QMC3D/, or any root used to locate spacefill_qmc3D.py",
    )
    parser.add_argument("--D", type=int, default=3)
    parser.add_argument("--S", type=int, default=65536, help="QMC candidates; must be 2^16=65536")
    parser.add_argument("--rect_l", type=float, default=0.1)
    parser.add_argument("--bases", type=int, nargs=3, default=[2, 3, 5])
    parser.add_argument("--class_max", type=int, default=10)
    parser.add_argument("--loss_max", type=float, default=0.002)
    parser.add_argument("--loss_hope", type=float, default=0.001)
    parser.add_argument("--stop_mae", type=float, default=0.006)
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--max_tensors", type=int, default=0)
    parser.add_argument(
        "--skip_embedding",
        action="store_true",
        help="Store embed at full size; do not alias lm_head to embed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.D != 3:
        raise ValueError(f"--D must be 3 for QMC3D, got {args.D}")
    if args.S != 65536:
        raise ValueError(f"--S must be 65536 (2^16), got {args.S}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    bases: Tuple[int, int, int] = (int(args.bases[0]), int(args.bases[1]), int(args.bases[2]))
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    load_dtype = resolve_dtype(args.dtype)

    run_config = {
        "script": "compress_llm_qmc3d_hcstyle_tie.py",
        "policy": "hyper_compression_broad_parameter_coverage_with_lm_head_tie_alias",
        "model_path": str(args.model_path),
        "output_dir": str(out_dir),
        "methods_root": str(args.methods_root),
        "D": args.D,
        "S": args.S,
        "m": args.S,
        "rect_l": args.rect_l,
        "bases": list(bases),
        "class_max": args.class_max,
        "loss_max": args.loss_max,
        "loss_hope": args.loss_hope,
        "stop_mae": args.stop_mae,
        "num_gpus": args.num_gpus,
        "seed": args.seed,
        "dtype": args.dtype,
        "max_tensors": args.max_tensors,
        "skip_embedding": args.skip_embedding,
        "alias_lm_head_when_tied": not args.skip_embedding,
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    sq = load_qmc3d_module(args.methods_root)
    get_Try_qmc_torch_base = sq.get_Try_qmc_torch_base
    encode_tensor_torch_version = sq.encode_tensor_torch_version

    print("=" * 72)
    print("[1/6] Model file sizes …")
    print(f"  path: {args.model_path}")
    safetensors_bytes = safetensors_size_bytes(args.model_path)
    if safetensors_bytes is not None:
        print(f"  model.safetensors bytes: {safetensors_bytes} ({safetensors_bytes / 1e9:.4f} GB)")
    original_model_size_bytes = get_dir_size(args.model_path)
    print(f"  directory bytes: {original_model_size_bytes} ({original_model_size_bytes / 1e9:.4f} GB)")

    print("=" * 72)
    print(f"[2/6] Load model (dtype={args.dtype}) …")
    from transformers import AutoModelForCausalLM

    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        dtype=load_dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    state_dict = model.state_dict()
    parameter_names = collect_named_parameter_names(model)
    linear_weights = collect_linear_weight_names(model, state_dict)
    tied = lm_head_tied_embed(state_dict)
    print(f"  loaded in {time.perf_counter() - t0:.2f}s")
    print(f"  state_dict keys: {len(state_dict)}")
    print(f"  named_parameters: {len(parameter_names)}")
    print(f"  lm_head tied to embed: {tied}")

    import gc

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("=" * 72)
    print("[3/6] Plan tensor coverage (HC style + tie alias) …")
    work_order: List[Tuple[str, torch.Tensor, bool, str, str]] = []
    compress_attempts = 0
    deferred_tie_alias = False
    for name, tensor in state_dict.items():
        try_compress, reason = should_try_compress(
            name, tensor, parameter_names, args, compress_attempts, tied=tied
        )
        kind = infer_encode_kind(name, tensor, linear_weights) if try_compress else ""
        if try_compress:
            compress_attempts += 1
        if reason == "tied_alias_deferred":
            deferred_tie_alias = True
        work_order.append((name, tensor, try_compress, reason, kind))

    n_compress = sum(1 for _, _, tc, _, _ in work_order if tc)
    n_direct = len(work_order) - n_compress
    print(f"  compress attempts planned: {n_compress}")
    print(f"  direct store planned: {n_direct}")
    for reason in ("buffer", "small", "debug_max_tensors", "tied_alias_deferred"):
        n = sum(1 for _, _, tc, r, _ in work_order if not tc and r == reason)
        if n:
            print(f"    {reason}: {n}")

    num_gpus = max(1, min(args.num_gpus, torch.cuda.device_count()))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for QMC3D encoding.")
    print(f"  GPUs: {num_gpus} (cuda:0 .. cuda:{num_gpus - 1})")

    print("=" * 72)
    print(f"[4/6] Generate QMC3D base samples S={args.S} …")
    t_base = time.perf_counter()
    base_samples = get_Try_qmc_torch_base(args.S, args.rect_l, "cuda:0", bases=bases)
    print(f"  shape={tuple(base_samples.shape)}, time={time.perf_counter() - t_base:.2f}s")

    stop_threshold = [True, torch.tensor([args.stop_mae], dtype=torch.float32)]
    aux_template: Dict[str, Any] = {
        "rect_l": torch.tensor([args.rect_l], dtype=torch.float32),
        "m": torch.tensor([float(args.S)], dtype=torch.float32),
        "bases": bases,
        "class_max": torch.tensor([args.class_max], dtype=torch.float32),
        "loss_max": torch.tensor([args.loss_max], dtype=torch.float32),
        "loss_hope": torch.tensor([args.loss_hope], dtype=torch.float32),
        "stop_threshold": stop_threshold,
        "base_samples": base_samples,
    }

    encoded_dict: Dict[str, Any] = {}
    back_dict: Dict[str, torch.Tensor] = {}
    per_tensor: List[Dict[str, Any]] = []
    skipped_tensors: List[Dict[str, Any]] = []

    print("=" * 72)
    print("[5/6] Process tensors in state_dict order …")
    t_all = time.perf_counter()
    encode_idx = 0
    for name, tensor, try_compress, reason, kind in tqdm(work_order, desc="QMC3D hcstyle_tie"):
        if not try_compress:
            if reason == "tied_alias_deferred":
                skipped_tensors.append(
                    {"tensor_name": name, "reason": reason, "numel": int(tensor.numel())}
                )
                continue
            item = fallback_direct_store(name, tensor, reason)
            merge_encode_item_into_payload(encoded_dict, back_dict, item)
            entry = {
                "tensor_name": name,
                "shape": list(tensor.shape),
                "numel": int(tensor.numel()),
                "status": reason,
                "encode_kind": None,
                "load_type": 0,
                "mae": None,
                "max_abs_error": 0.0,
                "payload_bytes": estimate_tensor_payload_bytes(encoded_dict, name),
                "device": "cpu",
                "seconds": 0.0,
            }
            per_tensor.append(entry)
            skipped_tensors.append({"tensor_name": name, "reason": reason, "numel": int(tensor.numel())})
            continue

        dev = f"cuda:{encode_idx % num_gpus}"
        t_i = time.perf_counter()
        item = compress_one_tensor_qmc3d(
            name,
            tensor,
            kind,
            aux_template,
            stop_threshold,
            encode_tensor_torch_version,
            dev,
            encode_idx,
        )
        dt = time.perf_counter() - t_i
        merge_encode_item_into_payload(encoded_dict, back_dict, item)

        lt = item["load_type"]
        load_type = int(lt.item()) if isinstance(lt, torch.Tensor) else int(lt)
        status = item.get("status", "fallback" if load_type == 0 else "compressed")
        entry = {
            "tensor_name": name,
            "shape": list(tensor.shape),
            "numel": int(tensor.numel()),
            "status": status,
            "encode_kind": kind,
            "load_type": load_type,
            "mae": item.get("mae"),
            "max_abs_error": item.get("max_abs_error"),
            "payload_bytes": estimate_tensor_payload_bytes(encoded_dict, name),
            "device": dev,
            "seconds": round(dt, 4),
        }
        per_tensor.append(entry)
        if status == "fallback":
            skipped_tensors.append(
                {
                    "tensor_name": name,
                    "reason": "fallback_stop_mae",
                    "numel": int(tensor.numel()),
                    "mae": item.get("mae"),
                }
            )

        encode_idx += 1
        if encode_idx % 4 == 0 or encode_idx == n_compress:
            print(f"  [{encode_idx}/{n_compress}] {name}  {status}  {dt:.2f}s", flush=True)

    num_aliased = 0
    if tied and not args.skip_embedding and deferred_tie_alias:
        alias_lm_head_to_embed(encoded_dict, back_dict)
        num_aliased = 1
        print("  aliased lm_head.weight -> model.embed_tokens.weight", flush=True)

    encoded_dict["rect_l"] = torch.tensor(args.rect_l, dtype=torch.float32)
    encoded_dict["m"] = torch.tensor(args.S, dtype=torch.int64)
    encoded_dict["bases"] = bases

    print(f"  encoding finished in {(time.perf_counter() - t_all) / 60:.2f} min")

    num_total = len(state_dict)
    num_parameter = sum(1 for n in state_dict if n in parameter_names)
    num_buffer = num_total - num_parameter
    num_small = sum(1 for t in per_tensor if t["status"] == "small")
    num_direct = sum(1 for t in per_tensor if t["status"] in ("buffer", "small", "debug_max_tensors"))
    num_fallback = sum(1 for t in per_tensor if t["status"] == "fallback")
    num_compressed = sum(1 for t in per_tensor if t["status"] == "compressed")

    total_original_numel = sum(int(t.numel()) for t in state_dict.values())
    total_compressed_numel = sum(t["numel"] for t in per_tensor if t["status"] == "compressed")
    total_direct_numel = sum(t["numel"] for t in per_tensor if t["status"] != "compressed")
    logical_fp16_size_bytes = total_original_numel * 2
    logical_payload_bytes = tensor_dict_storage_bytes(encoded_dict)
    logical_payload_dedupe_bytes = tensor_dict_storage_bytes_dedupe(encoded_dict)

    recon_check = reconstruction_check_summary(state_dict, back_dict, per_tensor)

    print("=" * 72)
    print("[6/6] Save outputs and validate …")
    payload_path = out_dir / "compressed_payload.pt"
    recon_path = out_dir / "reconstructed_state.pt"
    torch.save(encoded_dict, payload_path)
    torch.save(back_dict, recon_path)

    compressed_payload_size_bytes = payload_path.stat().st_size
    reconstructed_state_size_bytes = recon_path.stat().st_size
    file_size_cr = (
        (safetensors_bytes or original_model_size_bytes) / compressed_payload_size_bytes
        if compressed_payload_size_bytes
        else None
    )

    compression_report = {
        "script": "compress_llm_qmc3d_hcstyle_tie.py",
        "policy": "hyper_compression_broad_parameter_coverage_with_lm_head_tie_alias",
        "alias_lm_head_when_tied": tied and not args.skip_embedding,
        "lm_head_tied_to_embed": tied,
        "num_aliased_tensors": num_aliased,
        "model_safetensors_bytes": safetensors_bytes,
        "original_model_size_bytes": original_model_size_bytes,
        "compressed_payload_size_bytes": compressed_payload_size_bytes,
        "reconstructed_state_size_bytes": reconstructed_state_size_bytes,
        "compression_ratio_file_size_safetensors_over_payload": round(file_size_cr, 6)
        if file_size_cr
        else None,
        "logical_fp16_size_bytes": logical_fp16_size_bytes,
        "logical_payload_tensor_bytes": logical_payload_bytes,
        "logical_payload_tensor_bytes_dedupe_storage": logical_payload_dedupe_bytes,
        "compression_ratio_logical_fp16_over_payload_dedupe": round(
            logical_fp16_size_bytes / logical_payload_dedupe_bytes, 6
        )
        if logical_payload_dedupe_bytes
        else None,
        "num_total_tensors": num_total,
        "num_parameter_tensors": num_parameter,
        "num_buffer_tensors": num_buffer,
        "num_compressed_tensors": num_compressed,
        "num_direct_tensors": num_direct,
        "num_fallback_tensors": num_fallback,
        "num_small_tensors": num_small,
        "total_original_numel": total_original_numel,
        "total_compressed_numel": total_compressed_numel,
        "total_direct_numel": total_direct_numel,
        "reconstruction_check": recon_check,
        "per_tensor": per_tensor,
    }

    with open(out_dir / "compression_report.json", "w", encoding="utf-8") as f:
        json.dump(compression_report, f, indent=2, ensure_ascii=False)
    with open(out_dir / "tensor_report.json", "w", encoding="utf-8") as f:
        json.dump(per_tensor, f, indent=2, ensure_ascii=False)
    with open(out_dir / "skipped_tensors.json", "w", encoding="utf-8") as f:
        json.dump(skipped_tensors, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("Summary [QMC3D hcstyle_tie]")
    print(f"  compressed / direct / fallback / aliased: "
          f"{num_compressed} / {num_direct} / {num_fallback} / {num_aliased}")
    print(f"  reconstruction OK: {recon_check['ok']}")
    print(f"  payload: {payload_path}")
    print(f"  report:  {out_dir / 'compression_report.json'}")
    print("=" * 72)

    if not recon_check["ok"]:
        raise RuntimeError(f"Reconstruction check failed: {recon_check}")


if __name__ == "__main__":
    main()
