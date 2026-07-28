# Simulation methods

## Data generation

Each scenario partitions the predictors into three correlated signal blocks and
one noise block. Predictor rows are sampled from a multivariate Gaussian
distribution with scenario-specific within-block and between-signal correlations.
Training and test samples are generated independently. Both are standardized using
the training-sample means and standard deviations.

For signal proportion $\kappa$, the number of active coefficients is

```math
q=\max\{\operatorname{round}(\kappa p),1\}.
```

Active coordinates are sampled without replacement and assigned random signs with
magnitude $0.6$. Responses follow

```math
Y_i\mid X_i\sim\operatorname{Poisson}(\mu_i),\qquad
\mu_i=\exp\{\operatorname{clip}(X_i^\top\beta^\star,-3.5,3.5)\}.
```

## Hyperparameter grids

```math
\begin{aligned}
\mathcal A
&=\{1.1,1.25,1.4,1.6,1.8,2.0\},\\
\mathcal D
&=\{10^{-4},0.3,0.5,0.7,1-10^{-4}\},\\
\Lambda_1=\Lambda_2
&=\left\{10^{-5/2+5k/14}:k=0,\ldots,7\right\},\\
\Lambda_{\mathrm{ridge}}
&=\left\{10^{-2+3k/5}:k=0,\ldots,5\right\}.
\end{aligned}
```

## Parameter selection

The proposed estimator is selected using repeated five-fold cross-validation on
one independent selection dataset. The ridge pivot is re-estimated inside every
training fold, preventing validation observations from influencing the pivot.
Candidates are ranked by mean Poisson deviance per observation across repeat-level
fold means, with repeated-CV standard deviation used only as a deterministic
tie-break.

## Evaluation

The main independent-test metrics are:

- Poisson deviance per observation;
- coefficient mean squared error;
- mean absolute prediction error;
- root mean squared prediction error; and
- coefficient dispersion among highly correlated predictors.

Confidence intervals use percentile bootstrap resampling. Paired Wilcoxon
signed-rank tests compare the proposed estimator with each baseline, followed by
Holm multiplicity adjustment.
