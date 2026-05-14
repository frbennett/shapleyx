# Discussion Papers

All discussion papers use the JSS journal style.

## Case Study Reports

Detailed case study reports in PDF format for key examples in the
ShapleyX package.

| Report | Description |
|---|---|
| [Cantilever Beam](../case_study_reports/cantilever_case_study.pdf) | Structural reliability analysis of a rectangular cantilever beam under two orthogonal tip forces. Six mixed-distribution inputs (LogNormal + Normal) with correlated dimensional parameters. Reproduces Demange-Chryst et al. (2022) Example 4.2. |
| [Fire Spread Model](../case_study_reports/fire_spread_case_study.pdf) | Reliability-oriented sensitivity analysis of the Rothermel wildland fire spread model. Ten mixed-distribution inputs (LogNormal + Normal) with truncation and correlation. Target Shapley effects estimated without importance sampling — demonstrates the limits of standard MC for rare events ($p_f \approx 10^{-4}$). Reproduces Demange-Chryst et al. (2022) Example 4.3. |
| [SAC-SMA Model](../case_study_reports/sac_sma_case_study.pdf) | Sensitivity analysis of the Sacramento Soil Moisture Accounting (SAC-SMA) conceptual rainfall-runoff model for flood forecasting. High-dimensional inputs spanning basin moisture storage, drainage, and routing parameters. Demonstrates RS-HDMR with ARD on a operational hydrologic model. |
| [Wing Weight Function](../case_study_reports/wing_weight_case_study.pdf) | Sensitivity analysis of the Forrester et al. (2008) wing weight model. Ten independent Uniform inputs across five orders of magnitude. Validated against OpenTURNS reference Sobol indices. Includes both independent and correlated-input analyses. |

## Methodology Papers

| Paper | Description |
|---|---|
| [ARD-CV Paper](../case_study_reports/ard_cv_paper.pdf) | Variable selection in RS-HDMR using automatic relevance determination and cross-validation. Covers the ARD-CV algorithm, model selection criteria, and comparative benchmarks against OMP and LASSO. |
| [MC Shapley Paper](../case_study_reports/mc_shapley_paper.pdf) | Monte Carlo estimation of Shapley effects for correlated inputs using conditional sampling with Gaussian copulas. Derivation, algorithm, variance analysis, and case studies including the cantilever beam, fire spread, and wing weight models. |

## See Also

- [Distribution Classes](distributions.md) — guide to writing custom distribution classes
- [Tutorials](../tutorials/cantilever_beam.ipynb) — Jupyter notebook examples including the Cantilever Beam tutorial
- [MC Shapley How-to](../how-to-guides/mc-shapley.md) — usage instructions
- [Example Notebooks](https://github.com/frbennett/shapleyx/tree/main/Examples) — all Jupyter notebooks
