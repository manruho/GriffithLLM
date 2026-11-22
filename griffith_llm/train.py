"""学習スクリプト。評価とチェックポイント選択は常に hard(離散化後)で行う。"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import yaml

from .data import get_batch, load_dataset
from .model import GPT, GPTConfig, set_lut_hard

DEFAULTS = {
    "dataset": "tiny_shakespeare",
    "data_dir": "data",
    "vocab_size": 4096,  # dataset=tinystories の BPE 語彙数
    "train_bytes": None,  # dataset=tinystories で学習テキストの先頭だけ使う場合のバイト数
    "out_dir": "runs/default",
    "seed": 0,
    "batch_size": 64,
    "max_steps": 5000,
    "eval_every": 500,
    "eval_batches": 50,
    "lr": 1e-3,
    "lut_lr": 1e-2,
    "weight_decay": 0.1,
    "warmup_steps": 100,
    "min_lr_ratio": 0.1,
    "grad_clip": 1.0,
    "device": "auto",
    "threads": None,
    "resume": False,  # out_dir/last.pt があれば続きから学習する
    "max_seconds": None,  # この秒数を超えたら評価・保存して止める(resume で続行できる)
    "model": {},
}


def load_config(path: str | None, overrides: list[str]) -> dict:
    cfg = dict(DEFAULTS)
    if path:
        cfg.update(yaml.safe_load(Path(path).read_text()) or {})
    for item in overrides:
        key, _, value = item.partition("=")
        parsed = yaml.safe_load(value)
        if key.startswith("model."):
            cfg["model"] = {**cfg["model"], key[len("model.") :]: parsed}
        elif key in cfg:
            cfg[key] = parsed
        else:
            raise KeyError(f"未知の設定キー: {key}")
    return cfg


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_scale(step: int, cfg: dict) -> float:
    if step < cfg["warmup_steps"]:
        return (step + 1) / cfg["warmup_steps"]
    progress = (step - cfg["warmup_steps"]) / max(1, cfg["max_steps"] - cfg["warmup_steps"])
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return cfg["min_lr_ratio"] + (1.0 - cfg["min_lr_ratio"]) * cosine


def build_optimizer(model: GPT, cfg: dict) -> torch.optim.Optimizer:
    lut_ids = {id(p) for p in model.lut_parameters()}
    decay, no_decay = [], []
    for p in model.parameters():
        if id(p) in lut_ids:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "lr": cfg["lr"], "weight_decay": cfg["weight_decay"]},
        {"params": no_decay, "lr": cfg["lr"], "weight_decay": 0.0},
    ]
    if lut_ids:
        groups.append({"params": model.lut_parameters(), "lr": cfg["lut_lr"], "weight_decay": 0.0})
    for g in groups:
        g["base_lr"] = g["lr"]
    return torch.optim.AdamW(groups, betas=(0.9, 0.99))


@torch.no_grad()
def evaluate(
    model: GPT, ids: torch.Tensor, cfg: dict, device: torch.device, *, hard: bool
) -> float:
    """検証 loss(nat)。hard=True は離散化後の回路、False は連続緩和。"""
    was_training = model.training
    model.eval()
    set_lut_hard(model, hard)
    gen = torch.Generator().manual_seed(1234)
    total = 0.0
    for _ in range(cfg["eval_batches"]):
        x, y = get_batch(ids, cfg["batch_size"], model.cfg.block_size, device, gen)
        total += model(x, y)[1].item()
    model.train(was_training)
    return total / cfg["eval_batches"]


def save_checkpoint(
    path: Path, model: GPT, tok_state: dict, cfg: dict, metrics: dict, **extra
) -> None:
    tmp = path.with_name(path.name + ".part")
    torch.save(
        {
            "model_cfg": model.cfg.to_dict(),
            "state_dict": model.state_dict(),
            "tokenizer": tok_state,
            "train_cfg": cfg,
            "metrics": metrics,
            **extra,
        },
        tmp,
    )
    tmp.replace(path)


def train(cfg: dict) -> dict:
    if cfg["threads"]:
        torch.set_num_threads(int(cfg["threads"]))
    torch.manual_seed(cfg["seed"])
    device = pick_device(cfg["device"])
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ids, val_ids, tok = load_dataset(cfg)
    model = GPT(GPTConfig(vocab_size=tok.vocab_size, **cfg["model"])).to(device)
    opt = build_optimizer(model, cfg)
    print(
        f"device={device} params={model.count_parameters()} "
        f"train_tokens={len(train_ids)} val_tokens={len(val_ids)}",
        flush=True,
    )

    log_path = out_dir / "log.jsonl"
    last_path = out_dir / "last.pt"
    best = {"step": -1, "val_hard": float("inf")}
    batch_gen = torch.Generator().manual_seed(cfg["seed"])
    first_step, elapsed_before = 0, 0.0
    if cfg["resume"] and last_path.exists():
        last = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(last["state_dict"])
        opt.load_state_dict(last["optimizer"])
        batch_gen.set_state(last["batch_gen"])
        best, first_step, elapsed_before = last["metrics"], last["step"], last["elapsed"]
        print(f"resumed from step {first_step}", flush=True)
    else:
        log_path.write_text("")
    session_start = time.time()
    start = session_start - elapsed_before
    train_loss, train_count = 0.0, 0
    model.train()
    for step in range(first_step, cfg["max_steps"] + 1):
        out_of_time = cfg["max_seconds"] and time.time() - session_start > cfg["max_seconds"]
        stop = step == cfg["max_steps"] or out_of_time
        is_eval_step = step % cfg["eval_every"] == 0 or stop
        # 再開直後のステップは中断前に評価・記録済み
        if is_eval_step and not (step == first_step and step > 0):
            val_hard = evaluate(model, val_ids, cfg, device, hard=True)
            val_soft = evaluate(model, val_ids, cfg, device, hard=False)
            row = {
                "step": step,
                "train_loss": train_loss / train_count if train_count else None,
                "val_hard": val_hard,
                "val_soft": val_soft,
                "ppl_hard": math.exp(val_hard),
                "ppl_soft": math.exp(val_soft),
                "elapsed": time.time() - start,
            }
            train_loss, train_count = 0.0, 0
            with log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(
                f"step {step:>6}  hard ppl {row['ppl_hard']:.3f}  soft ppl {row['ppl_soft']:.3f}"
                f"  {row['elapsed']:.0f}s",
                flush=True,
            )
            if val_hard < best["val_hard"]:
                best = {"step": step, "val_hard": val_hard, "val_soft": val_soft}
                save_checkpoint(out_dir / "best.pt", model, tok.state(), cfg, best)
            save_checkpoint(
                last_path,
                model,
                tok.state(),
                cfg,
                best,
                optimizer=opt.state_dict(),
                batch_gen=batch_gen.get_state(),
                step=step,
                elapsed=row["elapsed"],
            )
        if stop:
            break

        scale = lr_scale(step, cfg)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * scale
        x, y = get_batch(train_ids, cfg["batch_size"], model.cfg.block_size, device, batch_gen)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg["grad_clip"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        train_loss += loss.item()
        train_count += 1

    best = {**best, "ppl_hard": math.exp(best["val_hard"]), "last_step": step}
    (out_dir / "result.json").write_text(json.dumps(best, indent=2))
    print(f"best: step {best['step']}  hard ppl {best['ppl_hard']:.3f}", flush=True)
    return best


def main() -> None:
    p = argparse.ArgumentParser(description="ハイブリッド GPT の学習")
    p.add_argument("--config", help="YAML 設定ファイル")
    p.add_argument("overrides", nargs="*", help="key=value(model.* はモデル設定)")
    args = p.parse_args()
    train(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
