# Provenance and license notices

This is an independent derivative of the HIQL reference implementation in
[OGBench](https://github.com/seohongpark/ogbench), based on commit
`1d4140997f60c52c6fb0702ec100dc988b18c548` (OGBench 1.2.1).
It is not an official OGBench or HIQL release and does not imply endorsement by
their authors.

## Copyright and license

Copyright (c) 2024 OGBench Authors.

The complete upstream MIT license and copyright notice are retained verbatim in
`LICENSE`. The OGBench-derived source files remain subject to that notice.

Copyright (c) 2026 xzh-xzh, for the local rank-value modifications, tests,
configuration, and documentation, also provided under the MIT terms in `LICENSE`.
Repository visibility is managed separately through GitHub access controls.

## Included source and modifications

- `impls/agents/hiql.py`: OGBench HIQL with local rank loss, an optional upper-gap
  penalty (disabled by default), configuration validation, actor diagnostics, and
  conditional target updates. The exported algorithm file is unchanged from the
  local research source at export time.
- `impls/utils/datasets.py`: OGBench dataset utilities with local same-trajectory
  rank sampling, validity masks, and shared rank-image augmentation. Exported
  unchanged from the local research source.
- `impls/utils/{__init__,encoders,env_utils,evaluation,flax_utils,log_utils,networks}.py`:
  unchanged OGBench support code from the pinned commit.
- `impls/main.py`: the upstream training loop with rank configuration selected by
  default, a GPU requirement, local-only default logging, videos disabled by
  default, and an ignored output directory. Local experimental monitoring is not
  included.
- `impls/agents/__init__.py`: upstream registry reduced to the included HIQL agent.
- `impls/agents/hiql_rank.py`: local configuration selecting the original one-sided
  rank loss without changing the base HIQL configuration.
- `tests/reference_hiql.py`: unmodified upstream `impls/agents/hiql.py`, retained
  solely as the baseline for the TD and actor regression test.
- `tests/test_hiql_rank.py`: local synthetic-data tests. The export permits CPU
  unit tests; explicit CUDA validation uses `JAX_PLATFORMS=cuda`.

`UPSTREAM.json` records the base revision and SHA-256 hashes of the copied source
files, with text normalized to LF line endings. This repository starts with a source-only history; research artifacts and
the local research checkout's Git history are not included.

## Acknowledgments

The hierarchical method is based on
[HIQL](https://github.com/seohongpark/HIQL). OGBench credits
[jaxrl_m](https://github.com/dibyaghosh/jaxrl_m),
[rlbase](https://github.com/kvfrans/rlbase_stable), HIQL, and
[cmd-notebook](https://github.com/vivekmyers/cmd-notebook) for its JAX-based
algorithm implementations; those acknowledgments are retained here.

OGBench environments, robot assets, and datasets are installed or downloaded
separately and are not redistributed in this repository. Their own notices and
terms remain applicable. See the upstream OGBench repository for environment and
asset provenance.
