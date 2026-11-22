"""実数 attention + 論理ゲート(LUT)FFN のハイブリッド GPT。

`ffn='fp'` は通常の MLP、`ffn='lut'` は Griffith の LutLayer で FFN を置き換える。
LUT FFN は train() で連続緩和(soft)、eval() で Boolean 回路(hard)として動く。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from griffith_logic.binarize import BinarizeFunction
from griffith_logic.layers.lut import LutLayer


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    mlp_ratio: int = 4
    dropout: float = 0.0
    weight_tying: bool = True
    ffn: str = "fp"  # 'fp' | 'lut' | 'none'(FFN なし。LUT FFN の寄与を測る対照用)
    # --- ffn='lut' のときだけ使う ---
    lut_n_inputs: int = 2
    lut_depth: int = 2
    lut_in_bits: int | None = None  # None → d_model * 2
    lut_width: int | None = None  # None → d_model * mlp_ratio
    lut_parameterization: str = "softmax"
    lut_selection: str = "soft"  # 'soft' | 'hard_st' | 'gumbel_st'(学習時のゲート選択)
    lut_gumbel_noise: bool = False  # selection='soft' のときだけ有効
    lut_readout: str = "groupsum"  # 'groupsum'(popcount)| 'linear'(実数行列)
    rezero: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        q, k, v = (z.view(b, t, self.n_head, d // self.n_head).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(b, t, d)
        return self.resid_drop(self.proj(y))


class FloatFFN(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        hidden = cfg.d_model * cfg.mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LutFFN(nn.Module):
    """実数 → ビット → LUT 層 → 実数 の FFN。

    入口は Linear + 閾値二値化(STE)なので、LUT 層へ入るのは train/eval とも {0,1}。
    eval() では各 LUT が 1 つの真理値表に確定し、LUT 部分は純粋な Boolean 回路になる。
    出口は ReZero(0 始まりの係数 alpha)を掛けて residual に足す。
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        d = cfg.d_model
        in_bits = cfg.lut_in_bits or d * 2
        width = cfg.lut_width or d * cfg.mlp_ratio
        if cfg.lut_readout == "groupsum" and width % d != 0:
            raise ValueError(f"lut_width({width}) は d_model({d}) の倍数が必要")
        if cfg.lut_readout not in ("groupsum", "linear"):
            raise ValueError(f"lut_readout が不正: {cfg.lut_readout}")

        self.d_model = d
        self.width = width
        self.readout = cfg.lut_readout
        self.to_bits = nn.Linear(d, in_bits)
        self.luts = nn.Sequential(
            *[
                LutLayer(
                    in_dim=in_bits if i == 0 else width,
                    out_dim=width,
                    n_inputs=cfg.lut_n_inputs,
                    parameterization=cfg.lut_parameterization,
                    selection=cfg.lut_selection,
                    gumbel_noise=cfg.lut_gumbel_noise,
                )
                for i in range(cfg.lut_depth)
            ]
        )
        if self.readout == "linear":
            self.from_bits = nn.Linear(width, d)
        self.alpha = nn.Parameter(torch.zeros(1)) if cfg.rezero else None
        self.drop = nn.Dropout(cfg.dropout)

    def bits(self, x: torch.Tensor) -> torch.Tensor:
        """LUT 層の出力ビット(eval では {0,1})。x: (N, d_model)。"""
        h = BinarizeFunction.apply(self.to_bits(x), 0.0, -1.0, 1.0)
        return self.luts(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        h = self.bits(x.reshape(-1, self.d_model))
        if self.readout == "linear":
            y = self.from_bits(h - 0.5)
        else:
            group = self.width // self.d_model
            y = (h.view(-1, self.d_model, group).sum(-1) - group / 2) / math.sqrt(group)
        if self.alpha is not None:
            y = self.alpha * y
        return self.drop(y.view(*lead, self.d_model))


class NoFFN(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        if cfg.ffn == "fp":
            self.ffn: nn.Module = FloatFFN(cfg)
        elif cfg.ffn == "lut":
            self.ffn = LutFFN(cfg)
        elif cfg.ffn == "none":
            self.ffn = NoFFN()
        else:
            raise ValueError(f"ffn が不正: {cfg.ffn}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("net.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))
        if cfg.weight_tying:
            self.head.weight = self.tok_emb.weight

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        t = idx.size(1)
        if t > self.cfg.block_size:
            raise ValueError(f"系列長 {t} が block_size {self.cfg.block_size} を超えています")
        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size :])
            logits = logits[:, -1, :]
            if temperature <= 0.0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1, generator=generator)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    def lut_parameters(self) -> list[nn.Parameter]:
        return [p for m in self.modules() if isinstance(m, LutLayer) for p in m.parameters()]

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        lut = sum(p.numel() for p in self.lut_parameters())
        gates = sum(m.out_dim for m in self.modules() if isinstance(m, LutLayer))
        return {"total": total, "lut": lut, "gates": gates}


def set_lut_hard(model: nn.Module, hard: bool) -> None:
    """LUT 層だけを hard(eval)/ soft(train)に切り替える。dropout 等には触れない。"""
    for m in model.modules():
        if isinstance(m, LutLayer):
            m.train(not hard)
