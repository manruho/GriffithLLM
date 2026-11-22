# GriffithLLM

[Griffith](https://github.com/tsukushi-gate/Griffith)(微分可能論理ゲート NN)を使ったハイブリッド言語モデル。
attention・埋め込み・LayerNorm は実数のまま、Transformer の FFN だけを論理ゲート(LUT)に置き換える。

## セットアップ

```bash
uv sync --extra cpu   # CPU 版 PyTorch。CUDA 機での構成は未検証
```

## 使い方

```bash
uv run python -m griffith_llm.train --config configs/shakespeare_char_lut.yaml
uv run python -m griffith_llm.generate --ckpt runs/shakespeare_char_lut/best.pt --prompt "ROMEO:"
uv run python -m griffith_llm.evaluate --ckpt runs/shakespeare_char_lut/best.pt   # 検証データ全体の ppl
uv run pytest
```

設定は `key=value` で上書きできる(モデル設定は `model.` を付ける)。

```bash
# TinyStories + BPE(初回は約 1.9GB を取得して BPE 化する。train_bytes で先頭だけ使える)
uv run python -m griffith_llm.train --config configs/tinystories_lut.yaml train_bytes=30000000 max_steps=200
```

- `resume: true` なら `out_dir/last.pt` から続きを学習する。`max_seconds` を超えると評価・保存して止まる。
- TinyStories の BPE は学習テキストの先頭 1 億文字で学習し、`<|endoftext|>` を物語の区切りの 1 トークンにする。

## 数値の読み方

報告する ppl は常に `eval()`(各 LUT が 1 つの真理値表に確定した Boolean 回路)での値。
既定の `lut_selection: hard_st` は学習時の forward も argmax で 1 ゲートに確定させるので、
学習時と離散化後の出力が一致する。Griffith 標準の `soft`(softmax 混合)で学習すると、
同じ構成で soft ppl 6.3 に対し離散化後 13.5(1.5k step 時点)まで悪化した。

## 結果(Tiny Shakespeare 文字単位)

d_model=128・4層・4ヘッド・文脈64・バッチ64・15k step・seed 1本。検証は末尾 10%。
チェックポイントは同じ検証データで選んでいるので、数値はやや楽観的。

| 構成 | 検証 ppl(学習中の最良) | 検証全体の ppl | パラメータ数 |
|---|---|---|---|
| 実数 FFN + dropout 0.1 | 4.38 | 4.35 | 818k |
| 実数 FFN(dropout なし、6.5k step 以降は過学習) | 4.80 | 4.76 | 818k |
| LUT FFN、出口を実数行列(`lut_readout=linear`、`lut_lr=0.05`) | 4.83 | 4.80 | 780k |
| LUT FFN、出口を popcount(`configs/shakespeare_char_lut.yaml`) | 4.92 | 4.87 | 517k |
| FFN なし(`model.ffn=none`、attention のみ) | 4.95 | 4.91 | 291k |

- LUT FFN の寄与は FFN なしに対して 0.04(popcount)〜 0.12(実数行列)ppl。実数 FFN は 0.57。
  いまの構成では品質のほとんどを attention が担っている。
- 1.5k step の短い比較では、LUT の入力ビットと幅を 2 倍(512 / 1024)にすると 6.23 → 5.88、
  3 層にすると 6.37 に悪化した(同条件の実数 FFN 5.40、FFN なし 6.86)。15k step では未確認。
- CPU(4 スレッド)での学習速度は実数 FFN が約 0.1 秒/step、LUT FFN が約 0.65 秒/step。

## 結果(TinyStories + BPE 4096)

d_model=256・6層・8ヘッド・文脈256・バッチ32・16k step(約 0.27 エポック)・seed 1本。
Kaggle の T4 で学習。検証は TinyStories の valid 全体(480 万トークン)。
どの構成も 16k step でまだ下がり続けていて、収束前の比較になる。

| 構成 | 検証全体の ppl | パラメータ数 | 学習速度(T4) |
|---|---|---|---|
| 実数 FFN(`configs/tinystories_fp.yaml`) | 5.50 | 5.85M | 0.11 秒/step |
| LUT FFN 幅 2 倍(`model.lut_in_bits=1024 model.lut_width=2048`) | 6.64 | 7.82M | 0.75 秒/step |
| LUT FFN(`configs/tinystories_lut.yaml`、入力 512 ビット・幅 1024) | 6.72 | 5.26M | 0.40 秒/step |
| FFN なし(`configs/tinystories_noffn.yaml`) | 7.27 | 2.70M | 0.06 秒/step |

- LUT FFN は「FFN なし」と「実数 FFN」の差 1.77 ppl のうち 0.55(31%)を埋める。幅 2 倍で 0.63(35%)。
- LUT FFN のパラメータの大半は入口と出口の実数行列(1 層あたり約 39 万)で、ゲート自体は 1 層あたり 2048 個。
- `hard_st` なので学習時の forward と離散化後の出力は同じで、soft と hard の ppl は一致する。
