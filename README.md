# Adaptive Pivot-Based Shrinkage Estimator

## Overview

The adaptive pivot-based shrinkage estimator is designed for Poisson regression
with correlated predictors. It combines a convex coefficient penalty with
shrinkage toward a data-driven ridge-regression pivot. The estimator is evaluated
against Poisson Lasso, Elastic Net, and MNet using independently generated test
data.

The simulation study examines:

- predictive performance measured by Poisson deviance, $D$;
- coefficient recovery measured by
  $\operatorname{MSE}_{\boldsymbol\beta}$;
- prediction error measured by $\operatorname{MAE}$ and
  $\operatorname{RMSE}$;
- similarity of estimated coefficients within strongly correlated groups;
- sensitivity to the amount of trust placed in the pivot;
- sensitivity to perturbations of the pivot;
- performance in classical $n>p$ and high-dimensional $p\geq n$ regimes; and
- coefficient shrinkage paths.

## Estimator

Let $\widehat{\boldsymbol\beta}^{\,R}$ denote the ridge Poisson estimate used as
the pivot. For fixed $(\alpha,d,\lambda_1,\lambda_2)$, the proposed estimator is

$$
\widehat{\boldsymbol\beta}
=
\arg\min_{\boldsymbol\beta}
\left\{
\sum_{i=1}^{n}
\left[
\exp(\mathbf{x}_i^\top\boldsymbol\beta)
-y_i\mathbf{x}_i^\top\boldsymbol\beta
\right]
+
\lambda_1\sum_{j=1}^{p}
\left(|\beta_j|+\varepsilon\right)^\alpha
+
\lambda_2
\left\|
\boldsymbol\beta-d\widehat{\boldsymbol\beta}^{\,R}
\right\|_2^2
\right\}.
$$

The parameters have the following interpretations:

- $\alpha$ controls the shape of the convex coefficient penalty, with
  $1<\alpha\leq2$;
- $d$ scales the ridge pivot and therefore controls how strongly its magnitude
  influences the target;
- $\lambda_1$ controls coefficient shrinkage; and
- $\lambda_2$ controls shrinkage toward the scaled pivot
  $d\widehat{\boldsymbol\beta}^{\,R}$.

The numerical constant is $\varepsilon=10^{-5}$.

## Hyperparameter grids

The complete grids are

$$
\begin{aligned}
\mathcal A
&=\{1.1,\,1.25,\,1.4,\,1.6,\,1.8,\,2.0\},\\
\mathcal D
&=\{10^{-4},\,0.3,\,0.5,\,0.7,\,1-10^{-4}\},\\
\Lambda_1=\Lambda_2
&=\left\{10^{-5/2+5k/14}:k=0,\ldots,7\right\},\\
\Lambda_{\mathrm{ridge}}
&=\left\{10^{-2+3k/5}:k=0,\ldots,5\right\}.
\end{aligned}
$$

The implementation can use a coarse screen formed by taking every second entry
from the $\alpha$, $d$, $\lambda_1$, and $\lambda_2$ grids.

## Simulation scenarios

| Scenario | Training size, $n$ | Dimension, $p$ | $(p_1,p_2,p_3,p_0)$ | $(\rho_1,\rho_2,\rho_3)$ | $\rho_{\mathrm{ext}}$ |
|---|---:|---:|---|---|---:|
| $\mathrm{A}$ | $140$ | $60$ | $(10,10,10,30)$ | $(0.90,0.90,0.90)$ | $0.10$ |
| $\mathrm{B}$ | $140$ | $60$ | $(12,10,8,30)$ | $(0.90,0.85,0.90)$ | $0.30$ |
| $\mathrm{C}$ | $140$ | $76$ | $(14,12,10,40)$ | $(0.90,0.70,0.90)$ | $0.50$ |
| $\mathrm{D}$ | $120$ | $160$ | $(20,20,20,100)$ | $(0.85,0.85,0.90)$ | $0.30$ |

Each scenario contains three correlated signal blocks followed by one noise block.
The within-noise and signal-noise correlations are
$\rho_0=0.05$.

## Data-generating mechanism

Predictor rows are generated from

$$
\mathbf{x}_i\sim\mathcal N_p(\mathbf 0,\boldsymbol\Sigma),
$$

where $\boldsymbol\Sigma$ is determined by the scenario-specific block
correlations. Small covariance eigenvalues are truncated to maintain numerical
positive definiteness.

If $g(i)$ identifies the original block containing predictor $i$, the covariance
entries are

$$
\Sigma_{ij}
=
\begin{cases}
1,
& i=j,\\
\rho_g,
& i\neq j,\quad g(i)=g(j)=g\in\{1,2,3\},\\
\rho_{\mathrm{ext}},
& g(i),g(j)\in\{1,2,3\},\quad g(i)\neq g(j),\\
\rho_0,
& \text{otherwise},
\end{cases}
\qquad \rho_0=0.05.
$$

Training and test samples are generated independently. Both samples are
standardized using the training-sample means and standard deviations, and the
same random column permutation is applied to both.

For signal proportion $\kappa$, the number of active coefficients is

$$
q=\max\{\operatorname{round}(\kappa p),1\}.
$$

Active coordinates are sampled without replacement. Each active coefficient has
magnitude $0.6$ and a randomly selected sign; all remaining coefficients are
zero. A new active set, coefficient signs, training sample, and test sample are
generated for every Monte Carlo replication.

The response model is

$$
Y_i\mid\mathbf{x}_i
\sim\operatorname{Poisson}(\mu_i),
\qquad
\mu_i=
\exp\!\left\{
\operatorname{clip}
\left(
\mathbf{x}_i^\top\boldsymbol\beta^\star,-3.5,3.5
\right)
\right\}.
$$

The implemented data generator does not add an intercept.

## Parameter selection

The proposed estimator is tuned on an independent selection dataset using
repeated $K$-fold cross-validation. The default design uses
$R=10$ repetitions and $K=5$ folds.

The ridge pivot is selected and re-estimated inside every outer training fold.
Consequently, validation observations do not influence either the pivot or the
estimator fitted for that fold.

For validation observation $i$, the Poisson deviance contribution is

$$
D_i
=
2\left[
y_i\log\left(\frac{y_i}{\widehat\mu_i}\right)
+\widehat\mu_i-y_i
\right],
$$

where the logarithmic term is defined as zero when $y_i=0$. The validation
score is the deviance per observation. Scores are averaged first across folds
within each repeat and then across repeats. The selected candidate minimizes the
mean repeated-CV deviance. Repeated-CV standard deviation is used only as a
deterministic tie-break.

Writing $D_{rf}$ and $n_{rf}$ for the validation deviance and validation size in
fold $f$ of repeat $r$, respectively, the repeat-level and overall selection
scores are

$$
\overline D_r
=
\frac{1}{K}
\sum_{f=1}^{K}
\frac{D_{rf}}{n_{rf}},
\qquad
\overline D_{\mathrm{CV}}
=
\frac{1}{R}
\sum_{r=1}^{R}
\overline D_r.
$$

## Reference estimators

The proposed estimator is compared with:

- **Poisson Lasso**, using an $\ell_1$ penalty;
- **Poisson Elastic Net**, combining $\ell_1$ and quadratic penalties; and
- **MNet**, using a minimax-concave component together with quadratic
  regularization.

Reference-estimator penalties are selected using cross-validation on training
data.

## Evaluation metrics

### Poisson deviance per observation

$$
\frac{D_{\mathrm{test}}}{n_{\mathrm{test}}}
=
\frac{2}{n_{\mathrm{test}}}
\sum_{i=1}^{n_{\mathrm{test}}}
\left[
y_i\log\left(\frac{y_i}{\widehat\mu_i}\right)
+\widehat\mu_i-y_i
\right].
$$

### Coefficient mean squared error

$$
\operatorname{MSE}_{\beta}
=
\frac{1}{p}
\sum_{j=1}^{p}
\left(\widehat\beta_j-\beta_j^\star\right)^2.
$$

### Prediction errors

$$
\operatorname{MAE}
=
\frac{1}{n_{\mathrm{test}}}
\sum_i|y_i-\widehat\mu_i|,
$$

$$
\operatorname{RMSE}
=
\left[
\frac{1}{n_{\mathrm{test}}}
\sum_i(y_i-\widehat\mu_i)^2
\right]^{1/2}.
$$

### Within-group coefficient dispersion

$$
G_{\mathrm{within}}
=
\frac{1}{|\mathcal P_{\mathrm{within}}|}
\sum_{(i,j)\in\mathcal P_{\mathrm{within}}}
|\widehat\beta_i-\widehat\beta_j|,
$$

where $\mathcal P_{\mathrm{within}}$ contains predictor pairs from the same
signal group whose absolute training-sample correlation exceeds $0.8$.
Smaller values indicate more similar coefficient estimates among strongly
correlated predictors.

Analogous measures are calculated for strongly correlated predictors from
different signal groups and for pairs within the noise block.

## Simulation experiments

### Pivot-scaling sensitivity

This experiment varies $d$ while holding the other selected parameters fixed.
It measures how the amount of trust placed in the ridge pivot affects prediction,
coefficient recovery, and grouping.

### Pivot-quality sensitivity

The ridge pivot is compared with noisy versions of itself and with a
Lasso-derived pivot. This experiment evaluates whether performance gains depend
on the pivot carrying useful coefficient information.

### Classical sample-size analysis

The number of predictors is fixed while the training sample size increases into
the $n>p$ regime. This experiment studies how estimation error changes as more
information becomes available.

### High-dimensional finite-sample analysis

The number of predictors remains fixed while the training size varies across
$p>n$, $p\approx n$, and $p=n$ settings.

### Main estimator comparison

The proposed estimator, Lasso, Elastic Net, and MNet are compared over independent
Monte Carlo replications using the same data-generating settings and evaluation
metrics.

### Coefficient shrinkage paths

Coefficient estimates are traced over a decreasing sequence of
$\lambda_1$ values. Because the proposed estimator uses $\alpha>1$, its paths
represent continuous shrinkage rather than exact sparsity paths.

## Statistical inference

Percentile-bootstrap confidence intervals are calculated for mean performance
metrics. Paired two-sided Wilcoxon signed-rank tests compare the proposed
estimator with each reference estimator. Holm adjustment controls multiplicity
within each reported metric family.

## Reproducibility and computation

- Random seeds are derived from named experiment components using
  $\mathrm{SHA}\text{-}256$.
- Results remain reproducible when replications are executed in parallel.
- Independent training, selection, validation, and test roles are maintained.
- Numba-compiled coordinate-descent kernels are used when Numba is available.
- Joblib parallelism is used for Monte Carlo replications.
- Native BLAS threads are restricted within each worker to reduce
  oversubscription.
- Replication-level statistics, summary tables, confidence intervals,
  significance tests, and figures are retained as outputs.

## Main outputs

The simulation produces:

- repeated-CV fold and candidate summaries;
- replication-level and aggregated results for each experimental axis;
- bootstrap confidence intervals;
- raw and Holm-adjusted significance tests;
- coefficient-path data;
- PDF figures; and
- a run report containing the configuration, selected parameters, timing, and
  principal findings.

## Author

Ibrahim Bakari
