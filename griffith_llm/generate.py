"""チェックポイントから文章を生成する(hard = 離散化後の回路で動かす)。"""

from __future__ import annotations

import argparse

import torch

from .data import BpeTokenizer, CharTokenizer, tokenizer_from_state
from .model import GPT, GPTConfig


def load_model(path: str, device: str = "cpu") -> tuple[GPT, CharTokenizer | BpeTokenizer]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = GPT(GPTConfig(**ckpt["model_cfg"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, tokenizer_from_state(ckpt["tokenizer"])


def generate_text(
    model: GPT,
    tok: CharTokenizer | BpeTokenizer,
    prompt: str | None = None,
    max_new_tokens: int = 400,
    *,
    temperature: float = 0.8,
    top_k: int | None = None,
    seed: int = 0,
) -> str:
    """prompt の続きを生成する。BPE では物語の先頭(EOT の直後)から始め、次の EOT で切る。"""
    device = next(model.parameters()).device
    eot = getattr(tok, "eot_id", None)
    if prompt is None:
        prompt = "\n" if eot is None else ""
    ids = tok.encode(prompt)
    if eot is not None:
        ids = [eot, *ids]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    gen = torch.Generator(device=device).manual_seed(seed)
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k, generator=gen)[
        0
    ].tolist()
    if eot is not None:
        out = out[1:]
        if eot in out:
            out = out[: out.index(eot)]
    return tok.decode(out)


def main() -> None:
    p = argparse.ArgumentParser(description="ハイブリッド GPT で文章生成")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", default=None)
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    model, tok = load_model(args.ckpt, args.device)
    print(
        generate_text(
            model,
            tok,
            args.prompt,
            args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
