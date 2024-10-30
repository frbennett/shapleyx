---
title: 'ShapleyX: A Python Package for Meta-Model Based Parameter Sensitivity Analysis'
tags:
  - Python
  - sensitivity analysis
  - high dimensional model representation
authors:
  - name: Frederick R. Bennett
    orcid: 0000-0001-8868-6737
    affiliation: 1
affiliations:
 - name: Department of Environment, Science and Innovation, Queensland, Australia
   index: 1
date: 28 October 2024
bibliography: paper/paper.bib
---


# Summary

<!--Statement of need-->

SALib contains Python implementations of commonly used global sensitivity
analysis methods, including Sobol [@Sobol2001,@Saltelli2002a,@Saltelli2010a],
Morris [@Morris1991,@Campolongo2007], FAST [@Cukier1973,@Saltelli1999],
Delta Moment-Independent Measure [@Borgonovo2007,@Plischke2013]
Derivative-based Global Sensitivity Measure (DGSM) [@Sobol2009]
, and Fractional Factorial Sensitivity Analysis [@Saltelli2008b]
 methods.
SALib is useful in simulation, optimisation and systems modelling to calculate
the influence of model inputs or exogenous factors on outputs of interest.

<!--Target Audience-->

SALib exposes a range of global sensitivity analysis techniques to the
scientist, researcher and modeller, making it very easy to easily implement
the range of techniques into typical modelling workflows.

The library facilitates the generation of samples associated with a model's
inputs, and then provides functions to analyse the outputs from a model and
visualise those results [@saltelli_sensitivity_2023].

# References
