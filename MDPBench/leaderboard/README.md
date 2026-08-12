---
title: MDPBench Leaderboard
emoji: 📄
colorFrom: blue
colorTo: yellow
sdk: static
pinned: false
---

# MDPBench Leaderboard

Static, official leaderboard for **MDPBench: A Benchmark for Multilingual
Document Parsing in Real-World Scenarios**.

The deployed page needs no server or API key. Update `leaderboard.json` to
publish a new verified result. The initial entries are the results in the
repository's main-results table; `source_name` keeps the original table label
where the displayed official release/API identifier is more specific. The
initial board contains all 31 models from that table.

Before publishing an updated result table, synchronize the data and verify the
result:

```bash
python tools/sync_leaderboard.py --write
python tools/sync_leaderboard.py --check
```

## Deployment

This directory is the shared static-site source for both public deployments:

- Hugging Face: <https://huggingface.co/spaces/Delores-Lin/MDPBench-leaderboard>
- GitHub Pages: `https://delores-lin.github.io/MultimodalOCR/` (after GitHub
  Pages is enabled for the repository)

The `deploy-mdpbench-leaderboard` GitHub Actions workflow publishes this same
directory to both services on each `main`-branch update. GitHub Pages needs no
credential. For Hugging Face, add an `HF_TOKEN` Actions secret with write access
to `Delores-Lin/MDPBench-leaderboard`; the workflow skips that deployment until
the secret is present.

## Result policy

Public scores can be reproduced using the released evaluation set. Private
scores are evaluated by the MDPBench maintainers; a model is ranked by private
score only after that verification. See the benchmark README for the private-set
submission procedure.
