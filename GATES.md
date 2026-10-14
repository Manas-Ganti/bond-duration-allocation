# GATES.md — pass criteria, stated before any number was computed

Written 2026-10-14, before the curve generator, the bond model, any baseline or the agent had
been run even once. Each gate records PASS / FAIL and the date it was evaluated in the
**Status** table at the bottom. The machine-readable record of every evaluation is
`results/gates.json`, written by the code itself. Thresholds are mirrored in
`configs/config.yaml` under `gates:`; if the two ever disagree, this file is the statement of
intent and the config is a bug.

Nothing about the learned policy is reported until A, B and C pass. No historical yield is
downloaded, loaded or looked at until A–D pass (Gate E is enforced in code, not just policy).

Units: yields and returns are decimals (0.0001 = 1 bp). One period is one month (Δt = 1/12).
"Episode value" is the sum of monthly rewards over the 24-rebalance horizon.

---

## Gate A — the generator

**Claim tested:** `curve.py` samples the two-factor process it says it samples, and the curve
built from it gives back the factors through the same projection the policies observe.

Simulate 2,000 paths × 600 months from the stationary distribution (seeded). Recover the level
and slope factors from the simulated curve by projecting yields at the observation tenors
(2, 5, 10, 30y) onto the two Nelson-Siegel loadings. On the recovered factors estimate, pooled
across paths:

| moment | estimator | tolerance |
|---|---|---|
| mean level, mean slope | sample mean | ±10 bp absolute |
| innovation vol σ (each factor) | AR(1) residual sd mapped back to continuous time | ±5% relative |
| mean-reversion speed κ (each factor) | κ = −ln(φ̂)/Δt from pooled AR(1) | ±10% relative |
| shock correlation ρ | correlation of AR(1) residuals | ±0.05 absolute |

**Pass:** every moment inside its tolerance.

The targets are the priors in `configs/config.yaml`, chosen as round numbers *before* any
historical data was seen. Calibrating to FRED happens only after Gate E opens and is reported
as a model-error diagnostic, not used to retrain (see README).

## Gate B — pricing

**Claim tested:** the first-order return `r ≈ y·Δt − D·Δy` is close to the exact bond pricer
for small moves, and we know where it stops being close.

Buckets are constant-maturity zero-coupon exposures at 2, 7 and 20 years (modified duration =
maturity under continuous compounding). The exact one-month holding return of a zero is
`P(y+Δy, m−Δt) / P(y, m) − 1` with `P(y, m) = exp(−y·m)`. Evaluate at y = 5%.

**Pass:** for every bucket, |exact − approx| ≤ **5 bp** of return for all |Δy| ≤ **10 bp**.

**Recorded, not gated:** for each bucket, the smallest |Δy| at which the error exceeds 5 bp
(the "break point"), and the fraction of simulated monthly yield moves under the training
process that exceed it. This is the measured size of the missing convexity term.

## Gate C — trivial case

**Claim tested:** with nothing to plan for, the agent reproduces the greedy answer.

Train the fitted-Q agent with **zero transaction cost and a one-period horizon** (same sample
size and features as the main experiment). Compare its action to the myopic optimiser's on an
evaluation grid: every in-band previous allocation × a 21×21 grid of (level, slope) covering
±2 stationary standard deviations.

**Pass, both:**
- the two choose the identical allocation in **≥ 90%** of grid states, and
- mean one-period expected-return regret of the agent (true model) ≤ **0.5 bp**.

(Disagreement is expected only where two allocations have near-equal expected return, which
is why regret, not agreement alone, is the second criterion.)

## Gate D — agreement with the DP optimum

**Claim tested:** under the training process with costs (5 bp per unit traded), the learned
policy is as good as the dynamic-programming optimum and chooses the same allocations.

1. **Value.** On 50,000 common simulated episodes (stationary start, static allocation as the
   starting portfolio), compute the paired difference DP − FQI. **Pass:** the upper end of its
   95% CI is ≤ **5%** of DP's mean advantage over the static allocation.
2. **Policy map.** Over every in-band previous allocation × the DP factor-grid points within
   ±2 stationary sd, at t = 0 and t = 12, compare the long-bucket weight chosen by each policy.
   **Pass:** they are within one grid step (0.1) in **≥ 90%** of states.

**Recorded, not gated:** exact-match fraction; mean regret of FQI's action under DP's own Q on
the grid; DP grid convergence (value change from a 31×31/7-node grid to a 51×51/9-node grid on
the same paths); DP's predicted V₀ against its own Monte Carlo value.

## Gate E — historical data is sealed

`data/fred.py` refuses to download or read any yield file unless `results/gates.json` records
A, B, C and D as passing. Calibration to history, historical evaluation, and every historical
figure go through that loader. There is a test for the refusal.

---

## Status

| Gate | Result | Date | Notes |
|---|---|---|---|
| A | PASS | 2026-10-14 | max errors: mean 1.8 bp, sigma 0.1%, kappa 0.8%, rho 0.001 |
| B | PASS | 2026-10-14 | short: err≤10bp move 0.86 bp, breaks at 57 bp (12% of sim moves); intermediate: err≤10bp move 0.87 bp, breaks at 35 bp (25% of sim moves); long: err≤10bp move 2.09 bp, breaks at 16 bp (59% of sim moves) |
| C | PASS | 2026-10-14 | agreement 93.9%, mean regret 0.026 bp |
| D | PASS | 2026-10-14 | shortfall -0.09 bp/yr (CI high 0.42) vs threshold 2.51; map within one step 98.4% |
| E | PASS | 2026-10-14 | FRED first loaded after A-D passed; 1977-02-28 to 2026-10-31 |
