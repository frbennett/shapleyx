# Case Study Reports

Detailed case study reports in PDF format for key examples in the
ShapleyX package.

## Available Reports

| Report | Description |
|---|---|
| [Cantilever Beam](../case_study_reports/cantilever_case_study.pdf) | Structural reliability analysis of a rectangular cantilever beam under two orthogonal tip forces.  Six mixed-distribution inputs (LogNormal + Normal) with correlated dimensional parameters.  Reproduces Demange-Chryst et al. (2022) Example 4.2. |
| [Fire Spread Model](../case_study_reports/fire_spread_case_study.pdf) | Reliability-oriented sensitivity analysis of the Rothermel wildland fire spread model.  Ten mixed-distribution inputs (LogNormal + Normal) with truncation and correlation.  Target Shapley effects estimated without importance sampling — demonstrates the limits of standard MC for rare events ($p_f \approx 10^{-4}$).  Reproduces Demange-Chryst et al. (2022) Example 4.3. |
| [Wing Weight Function](../case_study_reports/wing_weight_case_study.pdf) | Sensitivity analysis of the Forrester et al. (2008) wing weight model.  Ten independent Uniform inputs across five orders of magnitude.  Validated against OpenTURNS reference Sobol indices.  Includes both independent and correlated-input analyses. |

## See Also

- [Distribution Classes](distributions.md) — guide to writing custom distribution classes
- [MC Shapley How-to](../how-to-guides/mc-shapley.md) — usage instructions
- [Example Notebooks](https://github.com/frbennett/shapleyx/tree/main/Examples) — all Jupyter notebooks
