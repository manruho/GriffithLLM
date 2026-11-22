"""コーパスの取得とトークナイズ。"""

from __future__ import annotations

import json
import shutil
import urllib.request
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

CORPORA = {
    "tiny_shakespeare": (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    ),
}
TINYSTORIES_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-{split}.txt"
)
EOT = "<|endoftext|>"
BPE_TRAIN_CHARS = 100_000_000  # BPE の学習に使う先頭の文字数


class CharTokenizer:
    def __init__(self, chars: list[str]) -> None:
        self.chars = chars
        self.stoi = {c: i for i, c in enumerate(chars)}

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.chars[i] for i in ids)

    def state(self) -> dict:
        return {"type": "char", "chars": self.chars}

    @classmethod
    def from_state(cls, state: dict) -> CharTokenizer:
        if state.get("type") != "char":
            raise ValueError(f"未対応のトークナイザ: {state.get('type')}")
        return cls(state["chars"])


class BpeTokenizer:
    """バイト単位 BPE。EOT(物語の区切り)は 1 トークン。"""

    def __init__(self, tok: Tokenizer) -> None:
        self.tok = tok
        self.eot_id = tok.token_to_id(EOT)

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int) -> BpeTokenizer:
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=[EOT],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        tok.train_from_iterator(texts, trainer)
        return cls(tok)

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self.tok.encode_batch(texts)]

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)

    def state(self) -> dict:
        return {"type": "bpe", "json": self.tok.to_str()}

    @classmethod
    def from_state(cls, state: dict) -> BpeTokenizer:
        return cls(Tokenizer.from_str(state["json"]))


def tokenizer_from_state(state: dict) -> CharTokenizer | BpeTokenizer:
    kind = state.get("type")
    if kind == "char":
        return CharTokenizer.from_state(state)
    if kind == "bpe":
        return BpeTokenizer.from_state(state)
    raise ValueError(f"未対応のトークナイザ: {kind}")


def load_text(name: str, data_dir: str | Path = "data") -> str:
    path = Path(data_dir) / f"{name}.txt"
    if not path.exists():
        if name not in CORPORA:
            raise FileNotFoundError(f"{path} が無く、既知のコーパスでもありません: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CORPORA[name], path)
    return path.read_text(encoding="utf-8")


def load_char_dataset(
    name: str, data_dir: str | Path = "data", val_fraction: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor, CharTokenizer]:
    """(train ids, val ids, tokenizer)。末尾 val_fraction を検証用にする。"""
    text = load_text(name, data_dir)
    tok = CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    n_val = int(len(ids) * val_fraction)
    return ids[:-n_val], ids[-n_val:], tok


def download(url: str, path: Path, max_bytes: int | None = None) -> None:
    """url を path に保存する。max_bytes を指定すると先頭だけ取得する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"Range": f"bytes=0-{max_bytes - 1}"} if max_bytes else {}
    tmp = path.with_name(path.name + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as r:
        with tmp.open("wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
    tmp.rename(path)


def iter_stories(path: Path, chunk_chars: int = 1 << 24) -> Iterator[str]:
    """EOT 区切りのテキストから物語を 1 つずつ返す。末尾の途切れた物語は捨てる。"""
    carry = ""
    with path.open(encoding="utf-8", errors="ignore") as f:
        while chunk := f.read(chunk_chars):
            *stories, carry = (carry + chunk).split(EOT)
            for story in stories:
                if story := story.strip():
                    yield story


def _head(stories: Iterable[str], max_chars: int) -> Iterator[str]:
    seen = 0
    for story in stories:
        yield story
        seen += len(story)
        if seen >= max_chars:
            return


def encode_stories(
    stories: Iterable[str], tok: BpeTokenizer, out_path: Path, batch: int = 2000
) -> int:
    """各物語を「本文 + EOT」として uint16 で out_path に書き出す。トークン数を返す。"""
    if tok.vocab_size > 1 << 15:
        raise ValueError("vocab_size は 32768 以下が必要(int16 で読み込むため)")
    tmp = out_path.with_name(out_path.name + ".part")
    count = 0
    buf: list[str] = []

    def flush(f) -> int:
        ids = [i for seq in tok.encode_batch(buf) for i in (*seq, tok.eot_id)]
        np.asarray(ids, dtype=np.uint16).tofile(f)
        buf.clear()
        return len(ids)

    with tmp.open("wb") as f:
        for story in stories:
            buf.append(story)
            if len(buf) >= batch:
                count += flush(f)
        if buf:
            count += flush(f)
    tmp.rename(out_path)
    return count


def load_tinystories(
    data_dir: str | Path = "data", vocab_size: int = 4096, train_bytes: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, BpeTokenizer]:
    """TinyStories を取得・BPE 化して (train ids, val ids, tokenizer) を返す。結果はキャッシュする。

    train_bytes を指定すると学習テキストの先頭だけを使う。検証は公式の valid 全体。
    """
    root = Path(data_dir) / "tinystories"
    size = train_bytes or "full"
    raw = {"train": root / f"train-{size}.txt", "val": root / "valid.txt"}
    cache = root / f"bpe{vocab_size}-{size}"
    meta_path = cache / "meta.json"
    if not meta_path.exists():
        for split, name in (("train", "train"), ("val", "valid")):
            if not raw[split].exists():
                limit = train_bytes if split == "train" else None
                download(TINYSTORIES_URL.format(split=name), raw[split], limit)
        cache.mkdir(parents=True, exist_ok=True)
        tok = BpeTokenizer.train(_head(iter_stories(raw["train"]), BPE_TRAIN_CHARS), vocab_size)
        counts = {
            split: encode_stories(iter_stories(raw[split]), tok, cache / f"{split}.bin")
            for split in ("train", "val")
        }
        meta_path.write_text(json.dumps({"tokenizer": tok.state(), "tokens": counts}))
    tok = BpeTokenizer.from_state(json.loads(meta_path.read_text())["tokenizer"])
    # 語彙は 32768 以下なので uint16 をそのまま int16 として読める(コピーなし)
    train, val = (
        torch.from_numpy(np.fromfile(cache / f"{split}.bin", dtype=np.int16))
        for split in ("train", "val")
    )
    return train, val, tok


def load_dataset(cfg: dict) -> tuple[torch.Tensor, torch.Tensor, CharTokenizer | BpeTokenizer]:
    if cfg["dataset"] == "tinystories":
        return load_tinystories(cfg["data_dir"], cfg["vocab_size"], cfg["train_bytes"])
    return load_char_dataset(cfg["dataset"], cfg["data_dir"])


def get_batch(
    ids: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(len(ids) - block_size - 1, (batch_size,), generator=generator)
    offsets = torch.arange(block_size)
    x = ids[starts[:, None] + offsets]
    y = ids[starts[:, None] + offsets + 1]
    return x.to(device).long(), y.to(device).long()
