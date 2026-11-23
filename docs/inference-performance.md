# Efficient inference for small requests

The LC0, Allie, UniMaia, Maia2, ChessCLIP, text-generation, and Carlini
pipelines choose their preprocessing strategy
according to the number of requested games. Small requests run in the calling
process, which avoids DataLoader and process-pool startup. Larger requests use
worker-backed preprocessing and parallel post-processing to improve throughput.

The default cutoff is 32 games. It is deliberately conservative and can be
overridden per request:

```python
predictions = pipeline.make_prediction(
    pgns,
    batch_size=16,
    parallel_threshold=64,
)
```

`num_workers` controls the choice explicitly:

- `num_workers=None` (the default) selects zero workers up to the threshold,
  and the package's normal preprocessing worker count above it.
- `num_workers=0` always uses synchronous preprocessing and post-processing.
- A positive `num_workers` forces that many workers, even for a small request.

`batch_size` still controls the number of positions passed to the model at
once. It is independent of the worker decision and must be positive.

LC0 converts PGNs and board histories eagerly. Consequently, its worker count
applies to the conversion and post-processing stages; already-converted
samples are batched locally without DataLoader workers. The `persistent_workers`
argument therefore has no effect for LC0. Allie and UniMaia parse or prepare
their examples lazily, so larger requests can use DataLoader workers and may
benefit from `persistent_workers=True` when making repeated calls. The
`persistent_workers` option requires a positive resolved worker count; use an
explicit positive `num_workers` when enabling it for a small request.

Pipeline-specific behavior:

| Pipeline family | Small-request path | Large-request work |
| --- | --- | --- |
| LC0 | Eager preparation and local batching | PGN/board preparation and post-processing pools |
| Allie | Local PGN parsing and token collation | DataLoader workers and post-processing pool |
| UniMaia | Local board/prompt preparation | DataLoader workers and post-processing pool |
| Maia2 | Local FEN encoding and batching | DataLoader workers and post-processing pool |
| ChessCLIP | Local history preparation and batching | Legal-move expansion in DataLoader workers |
| Text generation | Local prompt preparation and generation batches | Adaptive prompt preparation and post-processing |
| Carlini | Local preparation and sequential API calls | Adaptive preparation; async API batching remains separate |

For Carlini's synchronous API variants, `num_workers` affects preparation and
post-processing only; remote requests remain sequential. The async Carlini
variant uses `batch_size` for request deduplication and API concurrency, while
`num_workers` only controls CPU-side dataset preparation.

The pipelines preserve the existing iteration order unless `shuffle=True` is
requested (UniMaia and LC0). Empty input returns an empty prediction dataset.
Invalid PGNs and model errors continue to propagate normally. For a request
containing only one or a few games, the recommended settings are the defaults
or `num_workers=0`.

## Benchmarking overhead

Run the lightweight batching benchmark from the repository root:

```shell
python benchmarks/benchmark_pipeline_batching.py
```

It measures dataset batching and adaptive post-processing for representative
request sizes without downloading a model. Use `--sizes` to focus on the
range relevant to an application:

```shell
python benchmarks/benchmark_pipeline_batching.py --sizes 1 4 16 32 64 256
```

The benchmark is intended to identify the process-startup crossover, not to
predict model throughput. For end-to-end measurements, run representative
requests through a loaded local pipeline with `num_workers=0` and with the
default adaptive setting, after warming the model once. For remote Carlini
pipelines, measure preparation and network/API latency separately.
