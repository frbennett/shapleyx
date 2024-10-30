---
title: 'colorspace: A Python Toolbox for Manipulating and Assessing Colors and Palettes'
tags:
  - Python
  - color palettes
  - color vision
  - visualization
  - assesment
authors:
  - name: Reto Stauffer
    orcid: 0000-0002-3798-5507
    affiliation: "1, 2"
  - name: Achim Zeileis
    orcid: 0000-0003-0918-3766
    affiliation: 1
affiliations:
 - name: Department of Statistics, Universität Innsbruck, Austria
   index: 1
 - name: Digital Science Center, Universität Innsbruck, Austria
   index: 2
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
visualise those results.

# References
