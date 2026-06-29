# Computational code — *Generalized Rapidity in Relativistic Electrodynamics*

This repository contains all the numerical/analytical code used to produce the
figures and quantitative results of the paper

> N. S. Akintsov, A. P. Nevecheria, S. N. Andreev, Q.-H. Qin,
> *Generalized Rapidity in Relativistic Electrodynamics: Particle Trajectories
> and Radiation Signatures in Intense Laser Fields*, submitted to
> European Physical Journal D.

## Contents

| File | Purpose |
|------|---------|
| `make_figures.py` | Self-contained script generating all 10 figures (Figs. 1–10) as vector PDF + EPS. Set `FIG_PNG=1` to also emit PNG previews; set `FIG_OUTDIR` to redirect output. |
| `flatten.py` | Builds the single-file Springer submission `article/main_submission.tex` by inlining the section files and the generated `main.bbl`. |
| `requirements.txt` | Python dependencies. |

## Method summary

- **Figs. 1–4, 8** — closed-form analytical evaluation of the generalized-rapidity
  trajectories and Sarachik–Schappert harmonic spectra (Appendices A–C).
- **Figs. 5, 6** — direct numerical integration of the modified equations of motion
  with `scipy.integrate.solve_ivp` (adaptive RK45, `rtol=1e-8`, `atol=1e-11`).
- **Figs. 7, 9** — analytical harmonic/Bessel-sideband synthesis.
- **Figs. 7(b), 10** — the universal scaling law $\Delta\omega/\omega=2\xi$ with
  reproducible synthetic scatter (fixed seeds 7 and 42 via `numpy.random.default_rng`).

All quantities use reduced units: lengths in $r_{\rm osc}=c/\omega$, momenta in
$m_ec$, energies in $m_ec^2$, frequencies in $\omega$.

## Reproducing the figures

```bash
python make_figures.py
```

By default the figures are written to `../article/figures/`. To write elsewhere,
set the environment variable `FIG_OUTDIR`.

## Requirements

- Python ≥ 3.9
- numpy, scipy, matplotlib

Tested with Python 3.13, numpy 2.4.4, scipy 1.17.1, matplotlib 3.10.9.

## License / citation

Released to accompany the article above. If you use this code, please cite the
paper (and the archived Zenodo record, DOI to be added upon deposit).
