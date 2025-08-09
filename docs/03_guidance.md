# Guidance Equation

We implement an additive guidance strategy for candidate actions.
For a state $s$ and action $a$ the guided logits are

$$\log \hat{p}(a\mid s) = \log p_\theta(a\mid s) - \lambda\,\hat{A}(s \oplus a)$$

where $\hat{A}$ is the approximate assembly index of the successor state and
$\lambda$ is a non-negative weight. During diffusion sampling $\lambda$ is
increased linearly from $0$ to $\lambda_{\max}$ across the $T$ diffusion steps.

The function `additive_guidance` in `assembly_diffusion.guidance` applies the
above transformation, and `linear_lambda_schedule` provides the step-dependent
$\lambda_t = \lambda_{\max}\, t/(T-1)$ schedule.
