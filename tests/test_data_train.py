import torch

from griffith_llm import data
from griffith_llm.data import EOT, BpeTokenizer, iter_stories, tokenizer_from_state
from griffith_llm.evaluate import full_loss
from griffith_llm.generate import generate_text, load_model
from griffith_llm.train import DEFAULTS, evaluate, train

STORIES = [
    f"Once upon a time there was a {w} cat. It liked to play." for w in "big red sad old".split()
]
MODEL = dict(block_size=16, n_layer=1, n_head=2, d_model=32)


def write_corpus(path, n=60):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"\n{EOT}\n".join(STORIES[i % len(STORIES)] for i in range(n))
    path.write_text(text + f"\n{EOT}\nA story cut off in the mid")


def fake_download(url, path, max_bytes=None):
    write_corpus(path, 60 if "train" in url else 12)


def cfg_for(tmp_path, **kw):
    return {
        **DEFAULTS,
        "data_dir": str(tmp_path / "data"),
        "out_dir": str(tmp_path / "run"),
        "dataset": "tinystories",
        "vocab_size": 300,
        "batch_size": 8,
        "eval_batches": 2,
        "eval_every": 5,
        "warmup_steps": 2,
        "device": "cpu",
        **kw,
    }


def test_iter_stories_drops_partial_tail_across_chunks(tmp_path):
    path = tmp_path / "s.txt"
    write_corpus(path, 5)
    for chunk in (7, 1 << 20):
        stories = list(iter_stories(path, chunk))
        assert stories == [STORIES[i % 4] for i in range(5)]


def test_bpe_roundtrip_and_eot_is_one_token():
    tok = BpeTokenizer.train(STORIES, 300)
    text = "The red cat liked\nto play.\n\nOK?"
    assert tok.decode(tok.encode(text)) == text
    assert tok.encode(EOT) == [tok.eot_id]
    restored = tokenizer_from_state(tok.state())
    assert restored.encode(text) == tok.encode(text)
    assert tok.vocab_size <= 300


def test_load_tinystories_encodes_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "download", fake_download)
    train_ids, val_ids, tok = data.load_tinystories(tmp_path, vocab_size=300)
    assert train_ids.dtype == torch.int16 and train_ids.min() >= 0
    assert (train_ids == tok.eot_id).sum() == 60 and (val_ids == tok.eot_id).sum() == 12
    first = train_ids[: (train_ids == tok.eot_id).nonzero()[0, 0]].tolist()
    assert tok.decode(first) == STORIES[0]

    def fail(*a, **k):
        raise AssertionError("キャッシュがあるのに再取得した")

    monkeypatch.setattr(data, "download", fail)
    again, _, _ = data.load_tinystories(tmp_path, vocab_size=300)
    assert torch.equal(again, train_ids)


def test_train_bpe_lut_then_generate_and_full_eval(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "download", fake_download)
    cfg = cfg_for(tmp_path, max_steps=10, model={**MODEL, "ffn": "lut", "lut_selection": "hard_st"})
    best = train(cfg)
    assert best["ppl_hard"] < 300
    model, tok = load_model(str(tmp_path / "run" / "best.pt"))
    text = generate_text(model, tok, "Once upon", 20, seed=0)
    assert text.startswith("Once upon") and EOT not in text
    _, val_ids, _ = data.load_dataset(cfg)
    loss, tokens = full_loss(model, val_ids, batch_size=4)
    assert tokens == (len(val_ids) - 1) // 16 * 16 and 0 < loss < 6


def test_resume_matches_uninterrupted_run(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "download", fake_download)
    model_cfg = {**MODEL, "ffn": "lut", "lut_selection": "hard_st"}
    full = cfg_for(tmp_path, max_steps=10, model=model_cfg, out_dir=str(tmp_path / "full"))
    train(full)
    # 5 step で中断した状態を作る(スケジュールは 10 step のまま)
    part = {**full, "out_dir": str(tmp_path / "part")}
    monkeypatch.setattr("griffith_llm.train.lr_scale", stop_after(5))
    try:
        train(part)
    except KeyboardInterrupt:
        pass
    monkeypatch.undo()
    monkeypatch.setattr(data, "download", fake_download)
    train({**part, "resume": True})
    a = torch.load(tmp_path / "full" / "last.pt", weights_only=True)
    b = torch.load(tmp_path / "part" / "last.pt", weights_only=True)
    assert a["step"] == b["step"] == 10
    assert all(torch.equal(a["state_dict"][k], b["state_dict"][k]) for k in a["state_dict"])
    steps = [
        __import__("json").loads(line)["step"]
        for line in (tmp_path / "part" / "log.jsonl").read_text().splitlines()
    ]
    assert steps == [0, 5, 10]


def stop_after(step_limit):
    from griffith_llm.train import lr_scale

    def scale(step, cfg):
        if step >= step_limit:
            raise KeyboardInterrupt
        return lr_scale(step, cfg)

    return scale


def test_evaluate_restores_training_state_and_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "download", fake_download)
    cfg = cfg_for(tmp_path, max_steps=0, model={**MODEL, "ffn": "lut", "dropout": 0.1})
    train(cfg)
    model, _ = load_model(str(tmp_path / "run" / "best.pt"))
    _, val_ids, _ = data.load_dataset(cfg)
    model.train()
    a = evaluate(model, val_ids, cfg, torch.device("cpu"), hard=True)
    assert model.training and all(m.training for m in model.modules())
    assert a == evaluate(model, val_ids, cfg, torch.device("cpu"), hard=True)


def test_max_seconds_stops_early_and_resume_finishes(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "download", fake_download)
    cfg = cfg_for(tmp_path, max_steps=10, model={**MODEL, "ffn": "fp"})
    assert train({**cfg, "max_seconds": 1e-9})["last_step"] == 0
    assert train({**cfg, "resume": True})["last_step"] == 10
