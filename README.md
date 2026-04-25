# Resource-Adaptive KV-Cache Compression

A controlled, strict-byte-accounted Pareto comparison of KV-cache
compressors (quantization, low-rank, eviction, hybrid) on small
GPT-style transformer substrates, prepared for the **ICML 2026
Workshop on Resource-Adaptive Foundation Models (AdaptFM)**.

The empirical leaderboard with every experiment is in
[`results.tsv`](results.tsv); generated figures are in
[`figures/`](figures/).

## What's in here

```
.
├── train.py             ─ model + KV-cache compressor zoo + training loop + eval
├── prepare.py           ─ data download, BPE tokenizer, evaluate_bpb (read-only eval)
├── figures.py           ─ regenerates every figure in the paper from results.tsv
├── measure_realized.py  ─ realized GPU-memory measurement vs closed-form (App. B.2.b)
├── analysis.ipynb       ─ small notebook for inspecting results.tsv
├── results.tsv          ─ leaderboard; one row per experiment (108 rows)
├── figures/             ─ generated PNGs
├── pyproject.toml       ─ dependencies (PyTorch 2.6, matplotlib, pyarrow, ...)
├── uv.lock              ─ locked dependency versions for reproducibility
└── .python-version      ─ Python 3.10
```

## Reproducing the leaderboard

### 1. Install dependencies

The project uses [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

This installs PyTorch 2.6 (CUDA 12.4 wheel on Linux, MPS wheel on
Apple Silicon).

### 2. One-time data prep

```bash
uv run prepare.py
```

Downloads 11 FineWeb-Edu shards into `~/.cache/kvcompress/`, trains
an 8192-token BPE tokenizer, and caches the resulting `token_bytes`
lookup. Takes ~2 min.

### 3. Run a single experiment

```bash
uv run train.py
```

Trains the substrate from scratch on a 300-second wall-clock budget,
then runs the compression eval on the compressor pinned in
`agent_compressor = ...` near the bottom of `train.py`. On a 4090
this takes ~6 min and prints metrics like:

```
val_bpb:                  1.0937
baseline_bpb:             1.0937
compressed_bpb:           1.0973
val_bpb_delta:            0.00357
baseline_bytes_per_tok:   1536.0
compressed_bytes_per_tok: 400.0
compression_ratio:        3.8400
compressor_name:          int4_sym_per_tok_per_head
compression_score:        3.8043
```

To compare a different compressor, edit the `agent_compressor` line
near the end of `train.py` (e.g. `MixedKVCompressor(config, k_bits=4,
v_bits=2)`) and re-run.

### 4. Substrate scale

`train.py` auto-scales the substrate to the detected device. Override
via env vars:

```bash
DEPTH=10 ASPECT_RATIO=80 DEVICE_BATCH_SIZE=8 uv run train.py    # large substrate
DEPTH=6  ASPECT_RATIO=64 HEAD_DIM=128         uv run train.py    # hd128 sweep
```

The three substrates reported in the paper (small / medium / large)
correspond to (D=3, A=40), (D=6, A=64), (D=10, A=80).

### 5. Realized memory measurement (Appendix B.2.b)

```bash
uv run measure_realized.py
```

Allocates each compressor's state on GPU, calls
`torch.cuda.memory_allocated()` before and after, and reports
realized vs closed-form bytes-per-token-per-layer. Used to verify the
strict byte-accounting contract.

### 6. Regenerate figures

```bash
uv run figures.py
```

Reads `results.tsv` and rewrites all PNGs in `figures/`.

## Compressor families implemented

Every compressor in `train.py` inherits from `KVCompressor` and
implements `compress(K, V) -> (state, n_bytes)` and
`decompress(state) -> (K_hat, V_hat)`. The honest-byte contract:
`n_bytes` must equal the sum of stored tensor `numel × element_size`
plus all auxiliary scalars and indexing metadata.

| Family | Class | Description |
|---|---|---|
| Uniform symmetric INT-N | `INTNSymPerTokPerHeadCompressor` | per-(B, T, H) BF16 scale, no zero-point |
| Group-wise INT-N | `INTNGroupCompressor` | one BF16 scale per group of `group_size` channels along head_dim |
| Asymmetric INT-N | `AsymINTNCompressor` | adds BF16 zero-point per slice |
| Mixed K/V precision | `MixedKVCompressor` | independent bit-widths for K and V |
| Sliding window | `SlidingWindowCompressor` | keep last W tokens, drop the rest |
| StreamingLLM sink+window | `StreamingLLMCompressor` | first S sink tokens + last W |
| Top-k by `‖K‖` | `TopKKnormCompressor` | keep top-k tokens by row-wise K-norm |
| H₂O heavy-hitter | `H2OCompressor` | recent R + top-k by accumulated attention |
| Per-token SVD | `SVDLowRankCompressor` | rank-r approximation across heads |
| Random projection | `LowRankCompressor` | fixed Gaussian projection (Johnson-Lindenstrauss) |
| Head pruning | `HeadPruneCompressor` | drop K, V on a fraction of KV heads |
| Recency-tier hybrid | `HybridRecentFullOldQuantCompressor` | recent BF16 + old INT-N |
| Stack | `StackedCompressor` | outer eviction × inner quantization |

Identity (no compression) is the reference baseline: 4·H·D bytes per
token-layer.

## Hardware

Designed for a single 24 GB GPU (e.g. RunPod RTX 4090, A10, etc.).
Apple Silicon (MPS) is supported but auto-scales to a smaller default
substrate so it fits in unified memory. CPU works for `prepare.py`
but training on CPU is impractical.
