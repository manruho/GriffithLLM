"""チェックポイントを検証データ全体(重なりのない窓)で評価する。hard = 離散化後の値。"""

from __future__ import annotations

import argparse
import json
import math

import torch

from .data import load_dataset
from .generate import load_model
from .model import GPT


@torch.no_grad()
def full_loss(
    model: GPT, ids: torch.Tensor, device: str = "cpu", batch_size: int = 128
) -> tuple[float, int]:
    """(1 トークンあたりの loss[nat], 評価したトークン数)。model は eval() 済みであること。"""
    t = model.cfg.block_size
    n = (len(ids) - 1) // t
    x = ids[: n * t].view(n, t)
    y = ids[1 : n * t + 1].view(n, t)
    total = 0.0
    for i in range(0, n, batch_size):
        xb, yb = x[i : i + batch_size].to(device).long(), y[i : i + batch_size].to(device).long()
        total += model(xb, yb)[1].item() * len(xb)
    return total / n, n * t


def main() -> None:
    p = argparse.ArgumentParser(description="検証データ全体での ppl(離散化後)")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()

    model, _ = load_model(args.ckpt, args.device)
    cfg = torch.load(args.ckpt, map_location="cpu", weights_only=True)["train_cfg"]
    _, val_ids, _ = load_dataset(cfg)
    loss, tokens = full_loss(model, val_ids, args.device, args.batch_size)
    print(json.dumps({"ckpt": args.ckpt, "loss": loss, "ppl": math.exp(loss), "tokens": tokens}))


if __name__ == "__main__":
    main()
