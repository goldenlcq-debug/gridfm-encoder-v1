# gridfm-encoder-v1

Frozen weights of **encoder V1**, the shared graph encoder of our unified power-grid computation model.
This repository distributes the weights and the input-normalization statistics only. The model definition
and data-assembly code are **not** part of this release.

## Code

The encoder model definition, pretraining and probe code, the task-specific reasoners that read this encoder and
their weights are published at <https://github.com/goldenlcq-debug/unified-grid-representation>.
A loading example for the weights in this repository is `encoder/examples/encode_with_public_weights.py` there.

## Download

| File | Size | SHA-256 |
|---|---|---|
| [`encoder_v1_weights_20260907.zip`](https://github.com/goldenlcq-debug/gridfm-encoder-v1/releases/download/v1.0/encoder_v1_weights_20260907.zip) | 37,390,584 bytes | `885b2c2318812eb4a29cde89340441bc270975905fe96bcd718fd595c8001982` |

The zip unpacks to a single directory `encoder_v1/` with four files:

| File | Size | SHA-256 |
|---|---|---|
| `encoder_v1_weights.pt` | 40,470,780 bytes | `f8fb5856dcc013ad0bb815e8f80de16e1f59d83504955e08643a5b818d49def3` |
| `normalizer.npz` | 1,912 bytes | `117294fa3224b4fca42feee9735c562fb02d2c96a2b377761c9aa796c4a8ca76` |
| `SHA256SUMS.txt` | 171 bytes | checksums of the two files above |
| `MANIFEST.txt` | 1,695 bytes | plain-text manifest |

Verify after download:

```bash
sha256sum encoder_v1_weights_20260907.zip          # Linux / macOS
certutil -hashfile encoder_v1_weights_20260907.zip SHA256   # Windows
```

## What is inside `encoder_v1_weights.pt`

A plain PyTorch checkpoint, loadable with `torch.load(path, map_location="cpu", weights_only=True)`
(no custom classes are pickled). Top-level keys:

| Key | Content |
|---|---|
| `model_state` | 283 tensors, 10,096,706 float32 parameters |
| `config` | architecture only: `d=192, n_layers=8, n_global=16, n_heads=4, ffn_mult=2, lape_k=16, rwse_steps=16, random_dim=8, branch_random_dim=8, rand_proj_split=true` |
| `schema_version` | `"v0.2"` |
| `global_step` | `6000` |
| `source_checkpoint_blake2b16` | `ccf4260f2a8f5db5b6650a33b17d7a0f` (fingerprint of the frozen source checkpoint) |

Architecture in one line: a graph transformer with 8 layers of width 192, 4 attention heads, FFN multiplier 2,
16 learned global tokens, node tokens and branch (edge) tokens.

### Expected input widths (from the projection layers in `model_state`)

| Projection | Weight shape | Input = |
|---|---|---|
| `node_proj` | (192, 73) | 25 node features + 16 Laplacian-eigenvector PE + 16 PE-validity bits + 16 RWSE steps |
| `node_rand_proj` | (192, 8) | 8 random node channels (re-drawn per forward; separate projection, `rand_proj_split=true`) |
| `edge_proj` | (192, 14) | 14 branch features |
| `edge_rand_proj` | (192, 8) | 8 random branch channels |

## Input schema (v0.2)

Node features, 25 columns in this order. The *norm* tag says how the column is normalized before entering the model.

| # | Column | Dim | Group | Norm |
|---|---|---|---|---|
| 1–3 | `node_type` | 3 | topology | one-hot `[PQ, PV, Slack]` |
| 4 | `is_null` | 1 | topology | bool |
| 5 | `has_gen` | 1 | topology | bool |
| 6 | `has_load` | 1 | topology | bool |
| 7 | `vbase` | 1 | topology | z |
| 8 | `vmax` | 1 | static | pu_z |
| 9 | `vmin` | 1 | static | pu_z |
| 10 | `pmax` | 1 | static | pu_z |
| 11 | `pmin` | 1 | static | pu_z |
| 12 | `qmax` | 1 | static | pu_z |
| 13 | `qmin` | 1 | static | pu_z |
| 14 | `tj` | 1 | dynamic | log_z |
| 15 | `xdp` | 1 | dynamic | z |
| 16 | `sh` | 1 | dynamic | pu_z |
| 17 | `ph` | 1 | dynamic | pu_z |
| 18 | `pg` | 1 | state | pu_z |
| 19 | `qg` | 1 | state | pu_z |
| 20 | `pl` | 1 | state | pu_z |
| 21 | `ql` | 1 | state | pu_z |
| 22 | `v` | 1 | state | pu_z |
| 23 | `theta` | 1 | state | rad |
| 24 | `v_set` | 1 | state | pu_z |
| 25 | `adjustable` | 1 | meta | bool |

Branch features, 14 columns in this order:

| # | Column | Dim | Group | Norm |
|---|---|---|---|---|
| 1–6 | `branch_type` | 6 | topology | one-hot `[acline, series_comp, xfmr_2w, phase_shifter, xfmr_3w, shunt_comp]` |
| 7 | `r` | 1 | static | log_z |
| 8 | `x` | 1 | static | z |
| 9 | `b_half` | 1 | static | z |
| 10 | `mark` | 1 | state | raw (1 = in service, 0 = out of service) |
| 11 | `pi` | 1 | state | pu_z |
| 12 | `qi` | 1 | state | pu_z |
| 13 | `pj` | 1 | state | pu_z |
| 14 | `qj` | 1 | state | pu_z |

Normalization tags: `z` and `pu_z` are z-scores using the per-column `mean`/`std` stored in `normalizer.npz`
(per-unit conversion on a 100 MVA base is done when the case is parsed); `log_z` applies `sign(x)·log1p(|x|)`
first and then the z-score; `rad`, `raw`, `bool` and `one-hot` columns pass through unchanged.
Dynamic-group columns are zero for buses without a generator. `normalizer.npz` holds
`version` (`"v0.2"`), `node_mean`, `node_std` (25), `branch_mean`, `branch_std` (14), all float64.
**Use these statistics as shipped; do not re-fit them.**

## Loading sketch

```python
import numpy as np, torch

ck = torch.load("encoder_v1/encoder_v1_weights.pt", map_location="cpu", weights_only=True)
cfg = ck["config"]          # architecture hyper-parameters (see table above)
sd  = ck["model_state"]     # 283 tensors

nz = np.load("encoder_v1/normalizer.npz")
node_mean, node_std     = nz["node_mean"], nz["node_std"]        # shape (25,)
branch_mean, branch_std = nz["branch_mean"], nz["branch_std"]    # shape (14,)
```

## Provenance

The tensors are extracted unchanged (tensor-for-tensor identical) from the frozen encoder checkpoint
(training step 6000, frozen 2026-07-11, blake2b-128 `ccf4260f2a8f5db5b6650a33b17d7a0f`).
Removed from the original checkpoint: pre-training decoder head, tag head, optimizer state, AMP scaler
state, RNG states, early-stopping bookkeeping and training hyper-parameters.

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Citation

To be added when the accompanying paper is published.
