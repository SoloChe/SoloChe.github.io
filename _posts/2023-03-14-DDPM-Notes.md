---
layout: post
title: Notes of Diffusion Models 
date: 2023-03-05 20:00:00-0400
description: basics of diffusion models
comments: true
tags: Bayes Deep-Learning Diffusion-Model
---

**Last update on 05/18/2024:** Updated Guidance and the next update would be the connection with score-based DM and neural ODEs.


**Table of Contents**
- [Revisit of Denoising Diffusion Probabilistic Models (DDPM)](#revisit-of-denoising-diffusion-probabilistic-models-ddpm)
  - [DDPM Formulation](#ddpm-formulation)
  - [DDPM Forward Process (Encoding)](#ddpm-forward-process-encoding)
  - [DDPM Reverse Process (Decoding)](#ddpm-reverse-process-decoding)
  - [Training: ELBO](#training-elbo)
    - [Parameterization on $L\_t$](#parameterization-on-l_t)
    - [$L\_T$ and $L\_0$](#l_t-and-l_0)
  - [Implementation](#implementation)
- [Connection with DDIM](#connection-with-ddim)
  - [Accelerated Inference](#accelerated-inference)
- [Connection with Score-based DM](#connection-with-score-based-dm)
  - [Score Matching](#score-matching)
  - [DDPM to Score Matching (Tweedie's Formula)](#ddpm-to-score-matching-tweedies-formula)
- [Connection with Guidance](#connection-with-guidance)
  - [Classifier Guidance](#classifier-guidance)
  - [Classifier-free Guidance](#classifier-free-guidance)
  - [Noise Conditional Score Networks (NCSN)](#noise-conditional-score-networks-ncsn)
  - [Langevin Dynamics for Sampling](#langevin-dynamics-for-sampling)
    - [Langevin Dynamics](#langevin-dynamics)
- [Unified Framework by Stochastic Differential Equations (SDE)](#unified-framework-by-stochastic-differential-equations-sde)
  - [Forward](#forward)
  - [Reverse](#reverse)
    - [Sampling](#sampling)

# Revisit of Denoising Diffusion Probabilistic Models (DDPM)

Some good reviews:
1. [How diffusion models work: the math from scratch](https://theaisummer.com/diffusion-models/?fbclid=IwAR1BIeNHqa3NtC8SL0sKXHATHklJYphNH-8IGNoO3xZhSKM_GYcvrrQgB0o) 
2. [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) 
3. [Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/)
4. [Understanding Diffusion Models: A Unified Perspective](https://ar5iv.labs.arxiv.org/html/2208.11970)
   
## DDPM Formulation
Given the data distribution $\x_0\sim q(\x_0)$ which is unknown, we want to learn an approximation $p_\theta (\x_0)$ that we can sample from. It is similar to variational autoencoder (VAE) or hierarchical VAE in the form, e.g., it also has encoding process (forward process) and decoding process (reverse process) and minimizes ELBO, but with multiple high dimensional latent variables.

Diffusion models are latent variable models with the formulation (Markov chain)

$$
p_\theta(\x_0) = \int p_\theta(\x_{0:T})d\x_{1:T} \space \text{where} \space p_\theta(\x_{0:T}) = p(\x_T)\prod_{t=1}^{T}p_\theta(\x_{t-1}\mid\x_t).
$$

The latent variables are $\{\x_1,...,\x_T\}$ with the same dimensionality as the data $\x_0$ and their joint $p_\theta(\x_{0:T})$ is called the reverse process (decoding) starting at $p(\x_T)=\N(0,\I)$. The approximate posterior $q(\x_{1:T}\mid\x_0)$, called the forward process, is fixed to another Markov chain

$$
q(\x_{1:T}\mid\x_0) = \prod_{t=1}^{T}q(\x_t\mid\x_{t-1}).
$$

Compared to other latent variable models, e.g., VAE, it has no learnable parameters in the approximate posterior $q$ (encoding). In this process, the data $\x_0$ is transformed to $\x_T$ by gradually adding Gaussian noise. To recover the data distribution, the ELBO is maximized as

$$
\begin{align*}
\max_{\theta} \log p_\theta(\x_0) &= \log \int \frac{p_\theta(\x_{0:T})q(\x_{1:T}\mid\x_0)}{q(\x_{1:T}\mid\x_0)}d\x_{1:T}\\
&= \log \mathbb{E}_{\x_{1:T}\sim q(\x_{1:T}\mid\x_0)}\sbr{\frac{p_\theta(\x_{0:T})}{q(\x_{1:T}\mid\x_0)}} \\
&\geq \mathbb{E}_{\x_{1:T}\sim q(\x_{1:T}\mid\x_0)}\sbr{\log \frac{p_\theta(\x_{0:T})}{q(\x_{1:T}\mid\x_0)}}\\
&= -KL\sbr{q(\x_{1:T}\mid\x_0)\mid p_\theta(\x_{0:T})}.
\end{align*}
$$

## DDPM Forward Process (Encoding)
The original data $\x_0\sim q(\x_0)$ and the Markov chain assumes we add noise to the data $\x_0$ in each time step $t\in[1,T]$ with transition kernel $q(\x_t\mid\x_{t-1})$ which is usually handcrafted as 

$$
q(\x_t\mid\x_{t-1}) = \N\nbr{\x_t;\sqrt{1-\beta_t}\x_{t-1},\beta_t\I}
$$

where $\beta_t\in (0,1)$ is a hyperparameter. A closed form of dependence according to the reparameterization trick:

$$
\begin{align*}
\x_t &= \sqrt{1-\beta_t}\x_{t-1} + \sqrt{\beta_t}\bm{\epsilon}_{t-1} \quad \text{where} \quad \bm{\epsilon}_{t-1}\sim \N\nbr{\bm{0},\bm{I}}\\
&= \sqrt{\alpha_t}\x_{t-1} + \sqrt{1-\alpha_t}\bm{\epsilon}_{t-1} \quad \text{where} \quad \alpha_t = 1-\beta_t
\end{align*}
$$

Furthermore, with the addition of two Gaussian random variable, we can trace back to the data $\x_0$ 

$$
\begin{align}
\x_t &= \sqrt{\alpha_t\alpha_{t-1}}\x_{t-2} + \sqrt{\alpha_t-\alpha_t\alpha_{t-1}}\bm{\epsilon}_{t-2} + \sqrt{1-\alpha_t}\bm{\epsilon}_{t-1} \nonumber\\
    &= \sqrt{\alpha_t\alpha_{t-1}}\x_{t-2} + \sqrt{1-\alpha_{t}\alpha_{t-1}} \bm{\epsilon} \quad \text{where} \quad \bm{\epsilon}\sim \N\nbr{\bm{0},\bm{I}}\nonumber\\
    & \qquad ...\nonumber\\
    &= \sqrt{\bar{\alpha}_t}\x_{0} + \sqrt{1-\bar{\alpha}_{t}} \bm{\epsilon}
\end{align}\\ 
$$

where $\bar{\alpha}_ t = \prod_{i=1}^t \alpha_i $. Usually, $\alpha_i$ will decrease along with $t$, and therefore $\bar{\alpha}_t \rightarrow 0$ when $t \rightarrow \infty$.


## DDPM Reverse Process (Decoding)
To generate a new sample or reverse from $\x_T\sim\N(0,\I)$, we need to know $q(\x_{t-1} \mid \x_t)$ which is unavailable in practice for decoding. However, we know it is also Gaussian according to the Bayes' theorem. To make it tractable, we use $q(\x_{t-1} \mid \x_t, \x_0)$ which is conditioned on $\x_0$, which can be written as

$$
q(\x_{t-1} \vert \x_t, \x_0) = q(\x_t\mid\x_{t-1},\x_0)\frac{q(\x_{t-1}\mid\x_0)}{q(\x_t\mid\x_0)} = \mathcal{N}(\x_{t-1}; \color{blue}{\tilde{\bm{\mu}}}(\x_t, \x_0), \color{red}{\tilde{\beta}_t} \mathbf{I}),
$$ 

where

$$
\begin{align}
\tilde{\beta}_t 
&= \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t \nonumber\\
\tilde{\bm{\mu}}_t (\x_t, \x_0)
&= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \x_0 \nonumber\\
&= \frac{1}{\sqrt{\alpha_t}} \Big( \x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \bm{\epsilon}_t \Big)
= \tilde{\bm{\mu}}_t
\end{align}
$$
with $\x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(\x_t - \sqrt{1 - \bar{\alpha}_t}\bm{\epsilon}_t)$ (derived from Eq. (1)). The details of derivations can be found [here](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) (complete the square). 

Note that $q(\x_{t-1} \mid \x_t, \x_0) = q(\x_{t-1} \mid \x_t)$ due to Markovian property. The decoder $p_\theta$ with parameters $\theta$ is used to approximate $q(\x_{t-1} \mid \x_t, \x_0)$ with the same form as $q(\x_{t-1} \mid \x_t, \x_0)$ (Gaussian), i.e.,

$$
p_\theta(\x_{t-1} \mid \x_{t}) = \N\nbr{\x_{t-1}\mid \bm{\mu}_\theta(\x_t, t), \bm{\Sigma}_\theta(\x_t, t)}.
$$




## Training: ELBO

Our objective is to maximize the ELBO $-KL\sbr{q(\x_{1:T}\mid\x_0)\mid p_\theta(\x_{0:T})}$ which is equivalent to minimize the negative ELBO 

$$
\begin{align*}
L &= \mathbb{E}_{\x_{1:T}\sim q(\x_{1:T}\mid\x_0)}\sbr{\log \frac{q(\x_{1:T}\mid\x_0)}{p_\theta(\x_{0:T})}}\\
  &= \mathbb{E}_{\x_{1:T}\sim q(\x_{1:T}\mid\x_0)}\sbr{\log \frac{\prod_{t=1}^{T} q(\x_{t}\mid\x_{t-1})}{p_\theta(\x_T)\prod_{t=1}^{T}  p_\theta(\x_{t-1}\mid\x_{t})}}\\
  &= \mathbb{E}_{\x_{1:T}\sim q(\x_{1:T}\mid\x_0)}\sbr{-\log p_\theta(\x_T) + \sum_{t=2}^T\log \frac{q(\x_t\mid\x_{t-1})}{p_\theta(\x_{t-1}\mid\x_t)} + \log\frac{q(\x_1\mid\x_0)}{p_\theta(\x_0\mid\x_1)}}\\
  &= \mathbb{E}_{\x_{1:T}\sim q(\x_{1:T}\mid\x_0)}\sbr{-\log p_\theta(\x_T) + \sum_{t=2}^T\log \nbr{\frac{q(\x_{t-1}\mid\x_{t},\x_0)}{p_\theta(\x_{t-1}\mid\x_t)}\frac{q(\x_t\mid\x_0)}{q(\x_{t-1}\mid\x_0)}} + \log\frac{q(\x_1\mid\x_0)}{p_\theta(\x_0\mid\x_1)}}\\
  &= \mathbb{E}_{\x_{1:T}\sim q(\x_{1:T}\mid\x_0)}\sbr{-\log p_\theta(\x_T) + \sum_{t=2}^T\log \frac{q(\x_{t-1}\mid\x_{t},\x_0)}{p_\theta(\x_{t-1}\mid\x_t)} + \log \frac{q(\x_T\mid\x_0)}{q(\x_{1}\mid\x_0)} + \log\frac{q(\x_1\mid\x_0)}{p_\theta(\x_0\mid\x_1)}}\\
  &= \mathbb{E}_{\x_{1:T}\sim q(\x_{1:T}\mid\x_0)}\sbr{\log \frac{q(\x_T\mid\x_0)}{p(\x_T)} + \sum_{t=2}^T\log \frac{q(\x_{t-1}\mid\x_{t},\x_0)}{p_\theta(\x_{t-1}\mid\x_t)}  - \log p_\theta(\x_0\mid\x_1)}\\
  &= \underbrace{KL\sbr{q(\x_T\mid\x_0)\mid p(\x_T)}}_{L_T} + \sum_{t=2}^T \underbrace{\mathbb{E}_{\x_t\sim q(\x_t\mid\x_0)}\sbr{KL\sbr{q(\x_{t-1}\mid\x_{t},\x_0)\mid p_\theta(\x_{t-1}\mid\x_t)}}}_{L_{t}} - \underbrace{\mathbb{E}_{\x_{1}\sim q(\x_{1}\mid\x_0)}\sbr{\log p_\theta(\x_0\mid\x_1)}}_{L_0}
\end{align*}
$$

### Parameterization on $L_t$

For $L_t$, we assume the decoder $p_\theta$ has the same form as $q(\x_{t-1}\mid\x_t, \x_0)$ (see Eq. (2)), and the mean $\bm{\mu}_\theta$ is parameterized as

$$
\bm{\mu}_\theta = \frac{1}{\sqrt{\alpha_t}} \nbr{\x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \bm{\epsilon}_\theta(\x_t, t)}
$$

$$
p_\theta(\x_{t-1}\mid\x_t) = \N\nbr{\x_{t-1}; \frac{1}{\sqrt{\alpha_t}} \Big( \x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \bm{\epsilon}_\theta(\x_t, t) \Big), \bm{\Sigma}_\theta(\x_t, t)}
$$

In the above case, the $\bm{\epsilon}_\theta(\x_t, t)$ is the output of the backbone model (usually U-net) which is used to approximate the added noise $\bm{\epsilon}_t$ in the forward process. 

The KL divergence between two Gaussian: 

$$
_{KL}(p||q) = \frac{1}{2}\left[\log\frac{|\Sigma_q|}{|\Sigma_p|} - d + (\bm{\mu_p}-\bm{\mu_q})^T\Sigma_q^{-1}(\bm{\mu_p}-\bm{\mu_q}) + tr\left\{\Sigma_q^{-1}\Sigma_p\right\}\right]
$$

We set $\bm{\Sigma}_\theta(\x_t, t) = \sigma_t^2\bm{I}$ where $\sigma_t^2 = \tilde{\beta}_t$ or $\sigma_t^2 = \beta_t$
$$
\begin{align}
L_t &= \mathbb{E}_{\x_0, \bm{\epsilon}} \sbr{\frac{1}{2\sigma_t^2} \| \tilde{\bm{\mu}}_t(\x_t, \x_0) - \bm{\mu}_\theta(\x_t, t) \|^2 } \nonumber\\
&= \mathbb{E}_{\x_0, \bm{\epsilon}} \sbr{\frac{ (1 - \alpha_t)^2 }{2 \alpha_t (1 - \bar{\alpha}_t) \sigma_t^2} \|\bm{\epsilon}_t - \bm{\epsilon}_\theta(\x_t, t)\|^2} \nonumber\\
&= \mathbb{E}_{\x_0, \bm{\epsilon}} \sbr{\frac{ (1 - \alpha_t)^2 }{2 \alpha_t (1 - \bar{\alpha}_t) \sigma_t^2} \|\bm{\epsilon}_t - \bm{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\x_{0} + \sqrt{1-\bar{\alpha}_{t}} \bm{\epsilon}_t, t)\|^2} \\
\end{align}
$$

Eq. (3) is further reduced to 

$$
\begin{equation}
L_{\text{simple}}(\theta) := \mathbb{E}_{\x_0, \bm{\epsilon}} \sbr{\|\bm{\epsilon}_t - \bm{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\x_{0} + \sqrt{1-\bar{\alpha}_{t}} \bm{\epsilon}_t, t)\|^2},
\end{equation}
$$
where the weight term is removed for better sample quality.

From another perspective, $\bm{\mu}_\theta$ can be parameterized as 

$$
\bm{\mu}_\theta = \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \tilde{\x}_\theta(\x_t,t),
$$ 

where $\tilde{\x}_\theta(\x_t,t)$ is the output of the backbone model (U-net) which is used to predict the data $\x_0$ directly.

### $L_T$ and $L_0$ 

$L_T$ is considered a constant and ignored in the training (if $\beta_t$ is fixed). $L_0$ can be regraded as reconstruction error (VAE settings), i.e., $t=1$ in Eq. (4). More details can be found in the [DDPM paper Sec. 3.3](https://arxiv.org/abs/2006.11239).

## Implementation

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/post_img/DDPM-algo.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
      Traning and sampling process (Source: DDPM paper)
</div>

# Connection with DDIM 
DDIM is proposed to accelerate the inference of DDPM. The formulation is slightly different but the training is proved to be the same.  

The forward process is non-Markovian. Consider a family $Q$, indexed by a real vector $\sigma\in\R^T$, the joint is now conditioned on $\x_0$: $q_\sigma(\x_{1:T}\mid\x_0) = q_\sigma(\x_T\mid\x_0)\prod_{i=2}^Tq_\sigma(\x_{t-1}\mid\x_{t},\x_0)$ where 

$$
\begin{equation}
q_\sigma(\x_{t-1}\mid\x_{t},\x_0) = \N\nbr{\sqrt{\bar{\alpha}_{t-1}}\x_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\frac{\x_t-\sqrt{\bar{\alpha}_{t}}\x_0}{\sqrt{1-\bar{\alpha}_{t}}},\sigma_t^2\bm{I}}
\end{equation}
$$

The Eq. (4) is selected to satisfy the joint and $q_\sigma(\x_t\mid\x_0) = \N\nbr{\sqrt{\bar{\alpha}_t}, (1-\bar{\alpha}_t)\bm{I}}$.

The forward process is obtained by $q_\sigma(\x_t\mid\x_{t-1},\x_0) = \frac{q_\sigma(\x_{t-1}\mid\x_{t},\x_0)q_\sigma(\x_t\mid\x_0)}{q_\sigma(\x_{t-1}\mid\x_0)}$. It is also Gaussian and the magnitude controls the randomness in the forward process. If we set $\sigma_t = 0$, the forward process becomes deterministic. 

The reverse process is similar to the DDPM. The joint $p_\theta(\x_{0:T})$ is used to approximate the  $q_\sigma(\x_{t-1}\mid\x_{t},\x_0)$ by minimization of KL-divergence and the training objective is the same as $L_{\text{simple}}$ of DDPM ([Theorem 1 in DDIM paper](https://arxiv.org/abs/2010.02502)). We can let the model (U-net) predict either $\x_0$ directly or the noise $\bm{\epsilon}_t$. For example (noise prediction), 

$$
f_\theta(\x_t) = \frac{\x_t-\sqrt{1-\bar{\alpha}_t}\bm{\epsilon}_{\theta}(\x_t,t)}{\sqrt{\bar{\alpha}_t}} = \tilde{\x}_0
$$

 According to Eq. (5),

$$
\begin{align}
\x_{t-1} &= \sqrt{\bar{\alpha}_{t-1}}\tilde{\x}_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\frac{\x_t-\sqrt{\bar{\alpha}_{t}}\tilde{\x}_0}{\sqrt{1-\bar{\alpha}_{t}}} + \sigma_t\bm{\epsilon}\\
&= \sqrt{\bar{\alpha}_{t-1}}\nbr{\frac{\x_t-\sqrt{1-\bar{\alpha}_t}\bm{\epsilon}_\theta(\x_t,t)}{\sqrt{\bar{\alpha}_{t}}}} + \sqrt{1-\bar{\alpha}_{t}-\sigma_t^2}\bm{\epsilon}_\theta(\x_t,t) + \sigma_t\bm{\epsilon}
\end{align}
$$

$\sigma_t$ is set to $0$ in DDIM and the forward process becomes deterministic and the reverse process becomes implicit probabilistic model.

## Accelerated Inference
In DDPM, we need to go through every forward steps to sample in backward process due to Markovian property. DDIM proposes to use a subset of total time steps to sample in reverse process, i.e., $\{\x_{\tau_1},...,\x_{\tau_S}\}\subset\{\x_1,...,\x_T\}$. From another perspective, the interval between two time steps is increased in the reverse process, e.g., $\tau_{s+1}-\tau_s = \Delta\tau \geq 1$ where $\Delta\tau$ is a hyperparameter. Now, we are sampling from $q_\sigma(\x_{t-\Delta_\tau}\mid\x_{t},\x_0)$ iteratively. Hence, the reverse process is accelerated. The reason we can do this is that the loss function does depend on the "marginal" $q_\sigma(\x_t\mid\x_0)$ but not directly on the joint $q_\sigma(\x_{1:T}\mid\x_0)$. Also, the backward process is non-Markovian, i.e., $q_\sigma(\x_{t-\Delta_\tau}\mid\x_{t},\x_0)$. Hence, we can consider a subset of latent variables instead of all of them.


# Connection with Score-based DM

## Score Matching

Score of a probability density function $p(\x)$ is defined as $\nabla_{\x}\log p$. For example, if $p_{\bm{\theta}}$ is Gaussian distributed, $\nabla_{\x}\log p = -\frac{\x-\bm{\mu}}{\sigma^2} = -\frac{\bm{\epsilon}}{\sigma}$ (standardization). We want to learn a score model $\bm{s}\_\theta(\x)=\nabla_{\x}\log p_\theta$ to approximate $\nabla_{\x}\log p$ because we can draw samples from the approximated score function. To train $\bm{s}_\theta$, we minimize the Fisher Divergence, which is also called score matching,

$$
D_F\sbr{p_{data}\mid p_\theta} = \mathbb{E}_{p_{data}}\sbr{\frac{1}{2}\lVert\nabla_{\x}\log p_{data}(\x) - \bm{s}_\theta(\x)\rVert^2}.
$$

However, $p_{data}$ is unknown. It can be replaced by the derivative of $\bm{s}\_\theta(\x)$ to eliminate the unknown $p_{data}$, 

$$
D_F\sbr{p_{data}\mid p_\theta} = \mathbb{E}_{p_{data}}\sbr{Tr(\nabla_{\x} \bm{s}_\theta(\x)) + \frac{1}{2}\lVert \bm{s}_\theta(\x)\rVert^2} + \text{Constant}.
$$

Note that we use $\bm{s}\_\theta(\x)$ to approximate the $\nabla_{\x}\log p_{data}(\x)$. Hence, $Tr(\nabla_{\x} \bm{s}_\theta(\x))$ is actually the second derivative.  It raises another problem that the computation of the trace, which cannot be scaled to high dimensionality.

To circumvent the computation of $Tr(\nabla_{\x}^2\log p_{\bm{\theta}}(\x))$, we can use either **Denoising Score Matching** or **Sliced score matching**. Here, we focus on the former which is used in [Noise Conditional Score Networks](https://arxiv.org/abs/1907.05600). The idea is that we purturb the data $\x$ with a pre-specified noise distribution $q_{\sigma}(\tilde{\x}\mid\x)$ and minimize

$$
 \mathbb{E}_{q_{\sigma}}\mathbb{E}_{p_{data}}\sbr{\frac{1}{2}\lVert\bm{s}_\theta(\tilde{\x}) - \nabla_{\tilde{\x}} q_{\sigma}(\tilde{\x}\mid\x)\rVert^2}.
$$

Note that $\bm{s}\_\theta(\tilde{\x}) = \nabla_{\tilde{\x}} q_{\sigma}(\tilde{\x}\mid\x) \approx \nabla_{\x}\log p_{data}(\x)$ is true only when the noise level $\sigma$ is small enough, which leads to $\bm{s}\_\theta(\tilde{\x}) \approx p_{data}(\x)$. There are also two problems:

- Inaccurate score estimation in low data density
- Slow mixing of Langevin dynamics

## DDPM to Score Matching (Tweedie's Formula)

From Eq. (1), we have the forward process $\x_t\sim\N(\x_t \mid \sqrt{\bar{\alpha}_t}\x_0, (1-\bar{\alpha}_t)\I)$. By Tweedie's formula, the mean can be approximated as

$$
\begin{align*}
\sqrt{\bar{\alpha}_t}\x_0 &= \x_t + (1-\bar{\alpha}_t)\nabla_{\x_t}\log p(\x_t)\\
\x_0 &= \frac{\x_t}{\sqrt{\bar{\alpha}_t}} + \frac{1-\bar{\alpha}_t}{\sqrt{\bar{\alpha}_t}}\nabla_{\x_t}\log p(\x_t).\\
\end{align*}
$$

In this case, the $\tilde{\bm{\mu}}_t (\x_t, \x_0)$ can be parameterized as

$$
\tilde{\bm{\mu}}_t (\x_t, \x_0) = \frac{1}{\sqrt{\alpha}_t} \x_t + \frac{1-\alpha_t}{\sqrt{\alpha}_t}\nabla_{\x_t}\log p(\x_t).
$$

Then, the backbone model is used to predict the score at time step $t$. The approximation becomes

$$
\bm{\mu}_\theta = \frac{\x_t}{\sqrt{\bar{\alpha}_t}} + \frac{1-\bar{\alpha}_t}{\sqrt{\bar{\alpha}_t}}\bm{s}_\theta(\x_t, t).
$$

Finally, the objective becomes

$$
\begin{align*}
L_t &= \mathbb{E}_{\x_0, \bm{\epsilon}} \sbr{\frac{1}{2\sigma_t^2} \| \tilde{\bm{\mu}}_t(\x_t, \x_0) - \bm{\mu}_\theta(\x_t, t) \|^2 }\\
&= \mathbb{E}_{\x_0, \bm{\epsilon}} \sbr{\frac{1}{2\sigma_t^2}\frac{ (1 - \alpha_t)^2 }{\alpha_t} \|\bm{s}_\theta(\x_t, t) - \nabla_{\x_t}\log p(\x_t)\|^2}.
\end{align*}
$$

Note that the true noise and the score looks very similar in the above equation. If we compare two type of $\x_0$ from score and noise, we have $\nabla_{\x_t}\log p(\x_t)=-\frac{1}{\sqrt{1-\bar{\alpha}}}\bm{\epsilon}$.

# Connection with Guidance

In guided diffusion models, we approximate the conditional data distribution $p(\x\mid y)$ instead of $p(\x)$. Hence, our goal is to learn the gradient $\nabla_{\x_t}\log p(\x_t\mid y)$ which can be written as 

$$
\begin{equation}
\nabla_{\x_t}\log p(\x_t\mid y) = \underbrace{\nabla_{\x_t}\log p(y\mid\x_t)}_{\text{adversarial gradient}} + \underbrace{\nabla_{\x_t}\log p(\x)}_{\text{uncond. gradient}}.
\end{equation}
$$

## Classifier Guidance

In the classifier guidance, we have a classifier $\bm{C}_\phi$ which is used to predict the label $y$ given noised data $\x_t$. The gradient of the classifier is used as guidance, i.e.,

$$
\begin{align*}
\nabla_{\x_t}\log p(\x_t\mid y) &\approx \nabla_{\x_t}\log C_\phi(y\mid\x_t) + \bm{s}_\theta(\x_t, t)\\
&= \nabla_{\x_t}\log C_\phi(y\mid\x_t)  - \frac{1}{\sqrt{1-\bar{\alpha}}}\bm{\epsilon}_\theta(\x_t, t)\\
&= - \frac{1}{\sqrt{1-\bar{\alpha}}} \underbrace{\nbr{\bm{\epsilon}_\theta(\x_t, t) - \sqrt{1-\bar{\alpha}}\nabla_{\x_t}\log C_\phi(y\mid\x_t)}}_{\text{new predictor}}.
\end{align*}
$$

Note that $\nabla_{\x_t}\log p(\x_t)=-\frac{1}{\sqrt{1-\bar{\alpha}}}\bm{\epsilon}$. Then, the new predictor can be combined with a parameter $w$ to control the guidance strength, i.e., 

$$
\begin{equation}
\tilde{\bm{\epsilon}}(\x_t, t) = \bm{\epsilon}_\theta(\x_t, t) - w\sqrt{1-\bar{\alpha}}\nabla_{\x_t}\log C_\phi(y\mid\x_t),
\end{equation}
$$

where we have the form

$$
\nabla_{\x_t}\log p(\x_t) + \gamma \nabla_{\x_t}\log p(y\mid\x_t).
$$

One of the drawbacks of the classifier guidance is that the classifier has to be trained separately. In most of work, $\bm{\epsilon}\_\theta(\x_t, t)$ is replaced by the conditioned version $\bm{\epsilon}\_\theta(\x_t, t, y)$ in Eq. (9).

## Classifier-free Guidance

To avoid the classifier, we need to reconsider the term $\nabla_{\x_t}\log p(y\mid\x_t)$. With Bayes' theorem, we have

$$
\begin{align*}
\nabla_{\x_t}\log p(y\mid\x_t) &= \nabla_{\x_t}\log p(\x_t\mid y) - \nabla_{\x_t}\log p(\x_t)\\
&\approx \bm{s}_\theta(\x_t, t, y) - \bm{s}_\theta(\x_t, t)\\
&= -\frac{1}{\sqrt{1-\bar{\alpha}}}\nbr{\bm{\epsilon}_\theta(\x_t, t, y)-\bm{\epsilon}_\theta(\x_t, t)}
\end{align*}
$$

The idea is that we use a new model $\bm{\epsilon}\_\theta(\x_t, t, y)$ to learn the conditioned score. To obtain the approximated $\nabla_{\x_t}\log p(\x_t\mid y)$, we just add the unconditioned score, i.e., 

$$
\begin{align*}
\nabla_{\x_t}\log p(\x_t\mid y) &\approx \gamma\nabla_{\x_t}\log p(y\mid\x_t) + \nabla_{\x_t}\log p(\x_t)\\
&\approx -\frac{\gamma}{\sqrt{1-\bar{\alpha}}}\nbr{\bm{\epsilon}_\theta(\x_t, t, y)-\bm{\epsilon}_\theta(\x_t, t)} - \frac{1}{\sqrt{1-\bar{\alpha}}}\bm{\epsilon}_\theta(\x_t, t)\\
&=- \frac{1}{\sqrt{1-\bar{\alpha}}}\underbrace{\nbr{\gamma\bm{\epsilon}_\theta(\x_t, t, y) - (\gamma-1)\bm{\epsilon}_\theta(\x_t, t)}}_{\text{new predictor}}.
\end{align*}
$$

If we set $\gamma=w+1$, we have the same form as the one in the classifier-free paper. Hence, the new predictor can be written as

$$
\begin{equation}
\tilde{\bm{\epsilon}}(\x_t, t, y) = (w+1)\bm{\epsilon}_\theta(\x_t, t, y) - w\bm{\epsilon}_\theta(\x_t, t).
\end{equation}
$$

The advantage of the classifier-free guidance is that we do not need to train the classifier separately. The conditioned model $\bm{\epsilon}\_\theta(\x_t, t, y)$ and unconditioned model $\bm{\epsilon}\_\theta(\x_t, t, \emptyset)$ are learned jointly by "turn off" a certain ratio of the labels (looks like dropout normalization). For example, if we set the ratio (threshold) to $p$, we generate a mask `mask = torch.rand(cemb.shape[0])<threshold` where `cemb` is a batch of label embeddings. Then, we "turn off" these labels, i.e., `cemb[np.where(mask)[0]] = 0`. Finally, we add time embeddings and label embeddings, i.e., `emb = cemb + temb` to as the input to the backbone model.

## Noise Conditional Score Networks (NCSN)

To address these two issues, NCSN perturb the data in multiple steps.

## Langevin Dynamics for Sampling

- $\bm{s_\theta}:\mathbb{R}^D \rightarrow \mathbb{R}^D$: network trained to approximate the score of $p_{data}(\x)$

### Langevin Dynamics
$$
\x_{i+1} \leftarrow \x_i + \epsilon\nabla_{\x}\log p(\x) + \sqrt{2\epsilon}\bm{z}_i \quad i=0,1,...,T,
$$
where $\bm{z}_i\sim\N\nbr{\bm{0},\bm{I}}$. 

# Unified Framework by Stochastic Differential Equations (SDE)
## Forward
$$
d\x = \f(\x, t)dt + g(t)d\w
$$

- $\w$: standard Wiener process
- $\f(\cdot,t): \mathbb{R}^D \rightarrow \mathbb{R}^D$ drift coefficient
- $g(\cdot): \mathbb{R} \rightarrow \mathbb{R}$ diffusion coefficient of $\x_t$

## Reverse
$$
d\x = \sbr{\f(\x,t) - g(t)^2\nabla_{\x}\log p_t(\x)} + g(t)d\bar{\w}
$$

### Sampling

- General Numerical SDE Solver
- Predictor-corrector Sampler
- ODE Solver

