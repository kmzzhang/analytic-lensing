# Analytic Simplifications to Planetary Microlensing under the Generalized Perturbative Picture

https://arxiv.org/abs/2207.12412

Python implementation of the two methods to acquire analytic microlensing solutions proposed in Zhang 2022. Comments are welcome and should be directed to kemingz@berkeley.edu.

### updates

2022/08/02: optimized code (~40% speed-up for semi-analytic)

- use simplified coefficients in solve_four()

- evaluate only one cubic root in multi_quartic()

- only evaluate points above threshold for root-refinement

- further optimization still possible (e.g. numba; SG12)