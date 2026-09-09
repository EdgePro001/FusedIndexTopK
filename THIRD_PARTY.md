# Third-party software

The release is designed to run against frozen upstream checkouts. It does not
copy FlashInfer or DeepGEMM source into this repository.

| Project | Role | Qualified revision | License |
|---|---|---|---|
| [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | MQA indexer dependency and producer lineage | `7c95b14aa4a66edd7b682e5acdde62351ca81197` | MIT |
| [FlashInfer](https://github.com/flashinfer-ai/flashinfer) | exact Top-K performance baseline | `a0a6b019b9b27d49d209f85d028a1ae5a9b347d7` (`v0.6.17`) | Apache-2.0 |
| [PyTorch](https://github.com/pytorch/pytorch) | tensor runtime and independent correctness baseline | `2.10.0+cu130` in the reported environment | BSD-style |

DeepGEMM's current main branch may differ from the frozen qualified revision.
Porting to current upstream and rerunning the full qualification matrix is a
separate release gate.
