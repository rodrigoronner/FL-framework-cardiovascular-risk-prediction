# Headroom study — protocol (frozen before any run of this study)

Frozen: 2026-11-13 06:18 -03
Code: `experiments/headroom_study.py`, SHA-256 `fa53741dfc1b7fccb7e75074e0d7a374bdfb8a384ddcbe5365ed5c45b56ce9ca`
Design table: `results_revision/headroom_study_design.csv`

## Question
Clients whose budget exceeds the cap must shrink their lots. Should the
other clients spend their headroom (larger lots, larger budget, less noise
in the aggregate), and how much? Is a fuzzy policy that decides this per
federation better than fixed policies and than a crisp threshold rule?

## What was known before freezing
Results on the four-site federation (F01) and on one graded 8-client
federation were seen before writing the rules: spending headroom bought no
significant AUC with four sites and about +0.013 AUC in the graded
federation. The rules below were written from reasoning informed by these
two observations; F01 is therefore placed in the development set.

## Setting
sigma = 2.5, C = 1, T = 100, E = 1, delta = 1e-5, base lot 32, lots 8–128,
FedAdam with literature-default hyperparameters. Cap: every client takes at
least 12 lots per epoch, eps* = f(12) = 5.977 (CAP_EPS = 5.9773).
Partitions 0–9 x training seeds 0–2 for every arm.

## Federations (fractions of each site's folds; sizes in the design table)
F01 [1][1][1][1]; F02 [.5,.5][1][1][1]; F03 [1][.3,.7][1][1];
F04 [.5,.5][.5,.5][1][1]; F05 [.5,.5][1][.5,.5][1]; F06 [1][1/3 x3][1][1];
F07 [1][1/3 x3][1/3 x3][1]; F08 [.5,.5][.15,.15,.7][1][.15,.15,.7];
F09 [1][1][.25,.75][.25,.75]; F10 [1][.2 x5][1][1];
F11 [.5,.5][1/3,2/3][1/3,2/3][1/3,2/3]; F12 [1][1/6 x6][1/6 x6][1].
(F05 was first [.5,.5] at every site; it was replaced before any run because
every client was already at the cap, leaving no headroom to decide on.)

Split (seed 2026, F01 forced into development):
development F01 F02 F04 F05 F06 F12; confirmation F03 F07 F08 F09 F10 F11.

## Arms and allocation
Constrained clients (budget under base lots > eps*) take the largest lot
whose budget is <= eps*. Each other client takes
L = round(32 + frac x (L_cap − 32)), L_cap the largest lot allowed by eps*.
Arms: frac = 0 (no harm), 0.5, 1 (exact equalization), and the fuzzy frac.

## Inputs (public: client sizes and the accountant)
c = share of training records held by constrained clients.
r = 1 − V(frac=1) / V(frac=0), V = sum_k p_k^2 S_k / Lbar_k^2 (Eq. 7).

## Fuzzy policy (Mamdani, min/max, centroid)
c on [0, 0.5]: low (0,0,0.2), medium (0.1,0.25,0.4), high (0.3,0.5,0.5).
r on [0, 1]: small (0,0,0.5), moderate (0.3,0.6,0.9), large (0.7,1,1).
frac on [0, 1]: none (0,0,0.5), half (0.25,0.5,0.75), all (0.5,1,1).
Rules: low&small→none, low&moderate→none, low&large→half,
medium&small→none, medium&moderate→half, medium&large→all,
high&small→half, high&moderate→all, high&large→all.
Rationale: when much of the data sits in constrained (noisy) clients, the
federation leans on the cleaner updates of the others, so they should spend
headroom; when spending would barely reduce the aggregate noise, it is not
worth the privacy.

## Crisp threshold policy (PT)
Spend all (frac = 1) if c >= X, else none. X is chosen on the development
federations by minimizing the mean regret of validation macro AUC
(regret = best of the three fixed arms minus the chosen arm); ties go to
the larger X. Test folds are not used to choose X.

## Evaluation (confirmation federations only, test folds)
Unit: (federation, partition) block, training seeds averaged; 60 blocks.
Policies: P0, P05, P1 (fixed), PT, PF.
Reported: mean test macro AUC (site level), mean regret against the best
fixed arm per federation, mean client budget (privacy spent).

Success criteria for the fuzzy policy (all three must hold):
S1  non-inferior to always equalizing: one-sided 95% lower bound of
    (PF − P1) > −0.005 AUC.
S2  spends less privacy than always equalizing: mean client budget PF < P1.
S3  better than never spending: one-sided Wilcoxon p < 0.05 for (PF − P0).
Secondary (descriptive): PF − PT with 95% CI and mean budgets.
If any criterion fails, the fuzzy policy is not claimed to add value, and
the paper is repositioned around the audit and the allocation rules.
