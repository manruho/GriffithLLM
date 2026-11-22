import pytest
import torch

from griffith_llm.data import CharTokenizer, get_batch
from griffith_llm.model import GPT, GPTConfig, LutFFN, set_lut_hard

SMALL = dict(vocab_size=17, block_size=16, n_layer=2, n_head=2, d_model=32)


def make(ffn: str, **kw) -> GPT:
    torch.manual_seed(0)
    return GPT(GPTConfig(**SMALL, ffn=ffn, **kw))


@pytest.mark.parametrize("ffn", ["fp", "lut", "none"])
def test_forward_shape_and_loss(ffn):
    model = make(ffn)
    x = torch.randint(0, 17, (3, 16))
    logits, loss = model(x, x)
    assert logits.shape == (3, 16, 17)
    assert torch.isfinite(loss)


@pytest.mark.parametrize("ffn", ["fp", "lut"])
def test_causal(ffn):
    model = make(ffn).eval()
    x = torch.randint(0, 17, (1, 16))
    y = x.clone()
    y[0, -1] = (y[0, -1] + 1) % 17
    a, _ = model(x)
    b, _ = model(y)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)


@pytest.mark.parametrize("readout", ["groupsum", "linear"])
def test_lut_bits_are_boolean_in_eval(readout):
    model = make("lut", lut_readout=readout).eval()
    ffn = model.blocks[0].ffn
    assert isinstance(ffn, LutFFN)
    bits = ffn.bits(torch.randn(50, 32))
    assert set(bits.unique().tolist()) <= {0.0, 1.0}


def test_hard_st_training_forward_matches_eval():
    model = make("lut", lut_selection="hard_st")
    for block in model.blocks:
        torch.nn.init.ones_(block.ffn.alpha)
    x = torch.randint(0, 17, (3, 16))
    train_loss = model.train()(x, x)[1]
    eval_loss = model.eval()(x, x)[1]
    assert torch.allclose(train_loss, eval_loss, atol=1e-6)


def test_rezero_starts_at_zero():
    model = make("lut")
    out = model.blocks[0].ffn(torch.randn(4, 16, 32))
    assert torch.count_nonzero(out) == 0
    assert make("lut", rezero=False).blocks[0].ffn.alpha is None


def test_set_lut_hard_only_touches_lut_layers():
    model = make("lut", dropout=0.1).train()
    set_lut_hard(model, True)
    ffn = model.blocks[0].ffn
    assert all(not layer.training for layer in ffn.luts)
    assert ffn.drop.training and model.training


def test_lut_gradients_reach_lut_and_input_projection():
    model = make("lut")
    for block in model.blocks:
        torch.nn.init.ones_(block.ffn.alpha)
    x = torch.randint(0, 17, (4, 16))
    model(x, x)[1].backward()
    ffn = model.blocks[0].ffn
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in ffn.luts.parameters())
    assert ffn.to_bits.weight.grad.abs().sum() > 0


def test_lut_model_learns_in_hard_mode():
    model = make("lut")
    lut_ids = {id(p) for p in model.lut_parameters()}
    rest = [p for p in model.parameters() if id(p) not in lut_ids]
    opt = torch.optim.AdamW(
        [{"params": model.lut_parameters(), "lr": 1e-2}, {"params": rest}], lr=3e-3
    )
    data = torch.arange(400) % 17
    gen = torch.Generator().manual_seed(0)
    for _ in range(150):
        x, y = get_batch(data, 16, 16, "cpu", gen)
        loss = model(x, y)[1]
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    x, y = get_batch(data, 16, 16, "cpu", gen)
    assert model(x, y)[1].item() < 0.5


def test_generate_and_block_size_limit():
    model = make("lut").eval()
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), 40, temperature=0.0)
    assert out.shape == (1, 41)
    with pytest.raises(ValueError):
        model(torch.zeros((1, 17), dtype=torch.long))


def test_count_parameters():
    counts = make("lut").count_parameters()
    assert counts["gates"] == 2 * 2 * 128
    assert make("fp").count_parameters()["gates"] == 0


def test_char_tokenizer_roundtrip():
    tok = CharTokenizer.from_text("hello world")
    assert tok.decode(tok.encode("low roll")) == "low roll"
    assert CharTokenizer.from_state(tok.state()).chars == tok.chars
