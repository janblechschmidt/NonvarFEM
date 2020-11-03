# Error estimation for second-order PDEs in non-variational form
Authors: Jan Blechschmidt, Roland Herzog, Max Winkler

## Abstract
Second-order partial differential equations in non-divergence form are considered. Equations of this kind typically arise as subproblems for the solution of Hamilton-Jacobi-Bellman equations in the context of stochastic optimal control, or as the linearization of fully nonlinear second-order PDEs. The non-divergence form in these problems is natural. If the coefficients of the diffusion matrix are not differentiable, the problem can not be transformed into the more convenient variational form.
We investigate tailored non-conforming finite element approximations of second-order PDEs in non-divergence form, utilizing finite element Hessian recovery strategies to approximate second derivatives in the equation. We study both approximations with continuous and discontinuous trial functions. Of particular interest are a priori and a posteriori error estimates as well as adaptive finite element methods. In numerical experiments our method is compared with other approaches known from the literature.

## Link arxiv.org 
https://arxiv.org/abs/1909.12676

## NonvarFEM
This repository contains Python code to solve second-order partial differential equations in nonvariational form using FEniCS.
Currently, the following solvers are implemented:
- Direct solver using a finite element Hessian recovery 
- Iterative solver (using gmres) using a finite element Hessian recovery 
- Direct method by Neilan, https://doi.org/10.1515/jnma-2016-1017
- Direct method by Feng, Salgado, Zhang, https://arxiv.org/abs/1610.07992
