# From Alignment to Balance — Project Page

Project homepage for **From Alignment to Balance: Mode Coverage and Mode Seeking in Autoregressive Video Distillation**.

Deployed via GitHub Pages at **https://lijiaxing0213.github.io/DistillAlign**.

## Structure

- `index.html` — the entire single-page site (inline CSS/JS; fonts & icons via CDN).
- `static/imgs/` — figures exported from the paper:
  - `hypothesis1.png`, `hypothesis2.png` — the two hypotheses illustrations
  - `distribution_evolution.png` — Causal CD / DMD / joint distillation evolution in shared V-JEPA2 PCA space
  - `renoise.png` — teacher-normalized re-noising
  - `comparison.png`, `comparison_diversity.png` — qualitative comparisons
  - `pipeline_dist.png`, `pipeline_coverage.png` — pipeline distribution/coverage (spare)
  - `rd_logo.png` — Riemann Dynamics logo (nav)

## Local preview

```bash
# bundled PowerShell static server:
powershell -NoProfile -ExecutionPolicy Bypass -File serve.ps1
# then open http://localhost:8321
```

## TODO before publishing

- Fill the **Paper** and **Code** links in the hero `.actions` block of `index.html` (currently `#`).
