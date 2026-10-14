# CLAUDE.md

## Project: RL for Duration-Aware Bond Portfolio Allocation

An agent shifts allocation across short, intermediate and long duration exposure as the yield
curve moves, under position limits, a duration budget and transaction costs.

**The deliverable is not a better allocation. It is a learned policy proven correct against a
computable optimum, and then an honest measurement of how fast that correctness decays when the
curve stops behaving like the model it was trained on.**

Same structure as the optimal stopping project, different domain: build a world simple enough
that the right answer is knowable, prove the agent finds it, then break the world on purpose.

---

## HOW TO WORK ON THIS PROJECT (read first)

- **Scaffold the whole codebase in this session**, runnable end to end.
- Build in this order: yield-curve simulator -> bond return model -> classical baselines
  including the dynamic-programming optimum -> gates written down -> RL agent -> off-model
  evaluation -> historical evaluation -> writeup. **The DP baseline comes before the agent.**
- **Never report a number that was not produced by a real run.** No illustrative figures.
- **Write GATES.md before running anything.**

---

## THE INVIOLABLE RULE

**The state and action spaces stay small enough that the optimal policy remains computable by
dynamic programming.**

Three assets, a discretised allocation grid, and a two-factor curve state discretised on a
grid. That is the whole design constraint, and it is what separates this from the large pile of
"RL for portfolio allocation" projects that cannot be checked.

If the DP baseline becomes intractable because the state space grew, **the project has lost its
reason to exist.** Shrink the state space rather than dropping the baseline.

Corollary, inherited from the optimal stopping work: **no learned policy is believed until it
has reproduced a known-correct answer in a regime where one exists.**

---

## The decision problem

The portfolio holds three duration buckets. Call them short (~2y), intermediate (~7y) and long
(~20y) modified duration, plus optional cash.

**Single-period return**, first-order:

```
r_i  ≈  y_i · Δt  −  D_i · Δy_i
```

carry minus the duration effect. Convexity (`+ ½ C_i (Δy_i)²`) is **out of scope for v1** and
must be named as a limitation in the README, because ignoring it systematically understates
long-duration returns in large rate moves, which is exactly the regime you will be stressing.

**State:** current weights, portfolio duration, the two curve factors (level, slope), a
trailing volatility estimate, and last-period turnover.

**Actions:** target weights on a discretised simplex, subject to maximum position weight,
maximum turnover per rebalance, and a portfolio duration band.

**Reward:** total return, minus proportional transaction cost on turnover, minus a penalty for
duration-band violations. Report the decomposition, never just the total.

---

## Modules

### 1. `curve.py` — the generator
Two mean-reverting (Ornstein-Uhlenbeck) factors, level and slope, mapped to yields across
maturities in the Nelson-Siegel spirit. Seeded, vectorised.

Variants for stress testing only, never for training in the main experiment:
- a third factor (curvature)
- regime switching between high and low volatility states
- a jump in level

Calibration helper: fit the two-factor parameters to historical constant-maturity Treasury
yields.

### 2. `bonds.py`
Modified duration, yield to maturity, the return approximation above, and an **exact bond
pricer** used only to validate that approximation.

### 3. `baselines/`
- **`static.py`** — fixed allocation. The do-nothing baseline, and harder to beat after costs
  than people expect.
- **`immunize.py`** — duration matching to a target. The classical fixed-income answer.
- **`myopic.py`** — single-period optimiser: maximise expected return subject to the duration
  band, ignoring the future. The greedy baseline that isolates how much of the agent's value
  comes from being *dynamic* rather than from being *optimised*.
- **`dp.py`** — **backward-induction dynamic programming on the discretised state.** Under the
  training process this is the optimum. It is the reason this project is checkable, and it is
  the thing the agent is measured against.

### 4. `env.py`
Gymnasium interface over `curve.py` and `bonds.py`. One episode is a fixed horizon of
rebalances.

### 5. `agent/`
Start with **fitted Q-iteration**, for the same reason as the stopping project: closer to DP,
easier to debug, and if it already matches you may not need anything deeper. PPO only as a
second method if the first works.

### 6. `policy_map.py`
Extract the allocation policy as a function of curve state, for **any** policy, learned or
classical. This is the comparison surface and the figure generator. Write it early; it is how
you will debug everything else.

The headline figure: allocation to long duration, plotted over the (level, slope) plane, drawn
for both the DP policy and the learned one. If they look the same, the project worked.

### 7. `evaluate/`
- **`simulated.py`** — policy value over many simulated paths with confidence intervals
- **`offmodel.py`** — trained on two factors, evaluated on the stress variants
- **`historical.py`** — evaluated on actual yield-change paths
- **`costs.py`** — sweep the transaction cost parameter and find where the advantage over the
  static baseline disappears

---

## Real data

Constant-maturity Treasury yields from FRED (`DGS2`, `DGS5`, `DGS10`, `DGS30`), free CSV
download, decades of daily history.

Used for exactly three things: **calibrating** the simulator, **evaluating** a policy trained on
simulated curves, and **legibility** in the writeup.

**Never train on historical curves.** Forty years of monthly rebalancing gives roughly 500
decisions, dominated by a handful of regimes. Overlapping windows are the same episode counted
repeatedly. A policy trained on that memorises 1994, 2008 and 2022 rather than learning a rule.

This is the single most common failure in this project category, and avoiding it deliberately
is worth stating in the README.

---

## Pre-stated gates

Write into `GATES.md` **before running anything**, with pass or fail and the date.

- **Gate A (generator)** — simulated curves reproduce the target moments (mean level, mean
  slope, factor volatilities, mean-reversion speeds) within stated tolerance.
- **Gate B (pricing)** — the first-order return approximation matches the exact bond pricer
  within tolerance for small yield changes, and the tolerance is **recorded along with the yield
  move at which it breaks**.
- **Gate C (trivial case)** — with zero transaction costs and a one-period horizon, the agent's
  allocation matches the myopic optimiser.
- **Gate D (agreement)** — under the training process with costs, the agent's policy value is
  within stated tolerance of the DP optimum, and the policy maps agree over the state grid.
- **Gate E** — no historical data is loaded or looked at until A through D pass.

Gate E is the one that is easy to break and the one that matters. Looking at historical
performance early is how the configuration that happened to suit the last decade gets selected
without anyone deciding to select it.

---

## Findings to target

1. **Agreement with DP.** The validation result. Everything downstream depends on it.
2. **A no-trade band emerges under costs.** With zero costs the policy rebalances constantly.
   With costs it should develop a region where it holds. Does it, and does the band match the
   DP policy's?
3. **Where the advantage dies.** Sweep transaction costs and report the level at which the
   dynamic policy stops beating the static allocation. That number is more useful to a
   practitioner than any Sharpe ratio.
4. **Off-model degradation, measured relatively.** Train on two factors, evaluate under
   curvature, regime switching, and real history. Report **how much the learned policy degrades
   relative to how much DP degrades.** If the agent wins in-model and loses off-model, it
   overfit the generator, and that is the finding.

"The learned policy matches the computable optimum in-model and loses its advantage above N
basis points of cost" is a complete, honest, useful conclusion. A win is not required.

---

## Repository Structure

```
duration-allocation-rl/
|- README.md                   # what it is, REAL results, the policy-map figure, limitations
|- GATES.md                    # gates stated before any numbers
|- pyproject.toml
|- configs/config.yaml         # factors, OU params, durations, limits, costs, grids, seeds
|- src/duralloc/
|  |- curve.py                 # 2-factor OU generator + stress variants + calibration
|  |- bonds.py                 # duration, returns, exact pricer
|  |- baselines/
|  |  |- static.py
|  |  |- immunize.py
|  |  |- myopic.py
|  |  |- dp.py                 # the computable optimum
|  |- env.py
|  |- agent/
|  |  |- fitted_q.py           # build this first
|  |  |- ppo.py                # only if fitted Q works
|  |- policy_map.py            # policy extraction + figures
|  |- evaluate/
|  |  |- simulated.py
|  |  |- offmodel.py
|  |  |- historical.py
|  |  |- costs.py
|  |- data/fred.py
|  |- cli.py                   # `duralloc baseline`, `train`, `policymap`, `evaluate`
|- tests/
|  |- test_curve.py            # moments vs targets
|  |- test_bonds.py            # approximation vs exact pricer, and where it breaks
|  |- test_dp.py               # DP optimal on a tiny hand-solvable case
|  |- test_constraints.py      # no action ever violates limits
|  |- test_policy_map.py       # extraction correct on a known policy
|- results/
|- figures/
```

---

## Conventions

- **Seeded and reproducible.** Seed in config, recorded in `results/metrics.json`.
- Config-driven. No hardcoded durations, cost levels, or grid sizes.
- Every Monte Carlo number carries a confidence interval. Historical evaluation uses a
  **calendar-year cluster bootstrap**, because yield changes within a year are correlated and an
  ordinary bootstrap will understate uncertainty.
- Every figure generated by code from `results/metrics.json`.
- Constraint checks are tests, not assertions buried in the env. A policy that violates the
  duration band should fail CI.
- Dependencies: `numpy`, `scipy`, `pandas`, `matplotlib`, `gymnasium`, `scikit-learn`,
  `pydantic`, `pytest`. Ask before adding more.

---

## Definition of Done

1. `duralloc baseline` produces all four classical policies, with Gates A and B passing.
2. `duralloc train` produces a learned policy, with Gate C passing.
3. `duralloc policymap` emits the headline figure: long-duration allocation over the
   (level, slope) plane, DP against learned.
4. `duralloc evaluate` gives in-model performance with intervals, the cost sweep with the
   break-even level, off-model degradation for both policies, and historical results as a
   distribution rather than a single number.
5. `results/metrics.json` carries real numbers with seed and date.
6. README leads with the policy-map figure and the DP agreement, then the cost break-even, then
   off-model degradation, then a limitations section that names the missing convexity term and
   the discretisation.
7. `pytest` passes, including every gate's test.

---

## Scope and cut line

Target four to six days for the simplified prototype. Much of the scaffolding (gate structure,
bootstrap intervals, figure generation from recorded metrics) transfers from the optimal
stopping project.

- **Cut first:** PPO, the curvature factor, cash as a fourth asset, convexity.
- **Never cut:** the DP baseline, any gate, the policy-map figure, the cost sweep.

If the DP baseline is what is slowing you down, shrink the grid rather than removing it. A
coarse DP optimum is still an optimum. No DP means no project.

---

## Shared agent knowledge base

This machine runs an MCP knowledge base ("agent call centre") that other agents write to. The
tools are:

`mcp__callcentre__search`, `mcp__callcentre__get`, `mcp__callcentre__submit`,
`mcp__callcentre__report_outcome`, `mcp__callcentre__open_issue`, `mcp__callcentre__claim_issue`,
`mcp__callcentre__log_attempt`, `mcp__callcentre__resolve_issue`

Use them:

- **`search` first, before spending time on any error** — OU calibration instability, DP memory
  blowup on fine grids, Gymnasium API changes, FRED download failures, fitted-Q divergence.
- **`get`** to pull the full write-up of a matching entry.
- **`submit`** when you solve something non-obvious that cost real time, especially anything
  about discretisation choices, constraint handling in the action space, or sign conventions on
  duration, which are easy to get subtly backwards.
- **`report_outcome`** after trying a suggested fix, whether or not it worked.
- **`open_issue`** for a blocker you cannot resolve, with what you tried.
- **`claim_issue`** before working an open issue so two agents do not duplicate effort.
- **`log_attempt`** while working a claimed issue, so the trail survives if you stop.
- **`resolve_issue`** when it is fixed, with the resolution.

---

## Why this project exists (keep this framing in the README)

Reinforcement learning for portfolio allocation is a crowded category with a bad reputation,
and deservedly: the historical record is too short to train on, so most results are
memorisation reported as skill.

This project takes the opposite route. The world is simulated and deliberately simple enough
that the optimal policy can be computed exactly, so the learned policy can be **verified rather
than believed**. Only then is it taken somewhere the optimum is unknown, and the question
becomes not whether it wins but how quickly being correct stops meaning anything.