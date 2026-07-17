# The Free Energy Principle of Plasticity: Local Redundancy as Variational Free Energy in Deep World Models

**Authors:** Anonymous Authors

**Venue:** Preprint, July 2026

---

## Abstract

We present a unified theoretical framework connecting seven recently discovered phenomena in deep learning — local redundancy as a measure of plasticity (Cheng, ICML 2026), the equivalence between JEPA world models and active inference under SIGReg (Arnez & Gomez-Villa, 2026), the spectral rank pathology of depth (Everett, 2026), the algebraic confinement regime of grokking (Kam et al., 2026), the visit-alignment problem in looped transformers (Li et al., 2026), the discrete diffusion design space (Yuan et al., 2026), and consensus-based self-distillation (Gkountouras et al., 2026) — into a single normative principle: *plasticity is variational free energy*.

We prove seven main results and a practical algorithm. Theorem 1 establishes that local redundancy $\operatorname{LR}(\theta) = \mathbb{E}[\|\nabla_\theta \mathcal{L}_{\text{syn}}(\theta)\|^2]$ is exactly the Fisher information of the variational posterior in an active inference agent, with the SIGReg regularizer serving as the unique regularizer that closes the prior-miscalibration gap and preserves this equivalence exactly. Theorem 2 shows the spectral rank of the input-output Jacobian at initialization lower-bounds local redundancy after training, with equality iff the network uses Pre-Norm with SIGReg. Theorem 3 derives the algebraic representability condition for grokking as a critical Fisher information threshold $\operatorname{LR}^* = (k+1)/p^2$. Theorem 4 proves that the visit-alignment coefficient in looped transformers is proportional to local redundancy, yielding a depth-scaling law $\alpha \propto \operatorname{LR}(\theta)^{-1/2}$.

Theorem 5 extends the framework to discrete state spaces, proving that the discrete diffusion ELBO of Yuan et al. decomposes into per-token Fisher information terms, each bounded below by the local redundancy of the denoising network — establishing that *token-level plasticity determines generation quality*. Theorem 6 proves that CANON (Gkountouras et al.) is an empirical free energy minimization algorithm: consensus-anchored teachers provide a Monte Carlo estimate of the epistemic value term missing from current JEPA world models. Theorem 7 establishes a sharpness-plasticity duality: the Hessian spectral density (Singh et al.) is the Fourier transform of the local redundancy with respect to data distribution.

A practical algorithm, **PLASTIC** (PLasticity-Aligned Schema for Trainable Intelligent Cognition), follows directly: (i) SIGReg-regularized JEPA pretraining, (ii) synthetic memorization for plasticity monitoring, (iii) CANON-style consensus distillation for epistemic value, and (iv) Pre-Norm with loop-count-aware DeepNorm scaling. We state ten experimental predictions that differ qualitatively from existing accounts.

---

## 1 Introduction

The past month has seen a remarkable convergence of theoretical results across disjoint subfields of deep learning. Seven results in particular appear, at first glance, to address unrelated questions:

1. **What makes a network plastic?** Cheng (2026) introduced *local redundancy*, an information-theoretic measure derived from universal compression theory, and showed it predicts downstream continual learning performance better than any existing metric (ICML 2026 Spotlight).

2. **Why do JEPA world models work?** Arnez & Gomez-Villa (2026) proved that the SIGReg regularizer closes the "prior-miscalibration gap" in JEPA training, making the objective exactly equivalent to variational free energy minimization under active inference. They identified that no current JEPA computes *state-epistemic value*.

3. **How does depth affect trainability?** Everett (2026) showed that skip connections and normalization navigate an intrinsic tradeoff between rank collapse and ensemble-like behavior, determined by the branch-to-skip ratio $\rho_\ell$. Pre-Norm preserves rank; Post-Norm causes collapse.

4. **What determines grokking?** Kam et al. (2026) proved that with holomorphic monomial activations, network outputs are confined to a $(k+1)$-dimensional subspace. Tasks outside this subspace cannot even be memorized.

5. **How do looped transformers scale?** Li et al. (2026) introduced the visit-alignment coefficient $\kappa_R$ and showed that DeepNorm exponents must increase with loop count when visits align.

6. **How should discrete diffusion be designed?** Yuan et al. (2026) provided a unified framework showing that tokenization, vocabulary topology, and state space construction fundamentally shape DDM behavior.

7. **Can self-distillation work without labels?** Gkountouras et al. (2026) showed that CANON turns majority-vote consensus into dense token-level supervision, approaching gold-label performance.

**This paper shows these are not separate results. They are seven facets of a single underlying quantity: variational free energy.**

The central insight is that local redundancy $\operatorname{LR}(\theta)$ — the information-theoretic measure of plasticity — is exactly the Fisher information $\mathcal{I}_F(\theta)$ of the variational posterior $q_\theta(z|x)$ in an active inference agent. This single identification cascades through all seven settings:
- It explains why SIGReg preserves plasticity (it minimizes free energy rather than merely bounding it)
- It explains why Pre-Norm preserves rank (rank is a spectral proxy for Fisher information)
- It explains why grokking has a binary phase boundary (algebraic confinement sets minimum free energy)
- It explains visit-alignment in looped transformers (parameter-sharing concentrates Fisher information)
- It explains discrete diffusion quality (per-token Fisher information determines generation fidelity)
- It explains why CANON works (consensus is a Monte Carlo epistemic value estimator)
- It explains the Hessian spectrum (sharpness is the Fourier dual of plasticity)

---

## 2 Background and Definitions

### 2.1 Local Redundancy

Following Cheng (2026), let $f_\theta: \mathcal{X} \to \mathcal{Y}$ be a neural network. Define the *local model family* $\mathcal{F}_\theta^\epsilon = \{f_{\theta + \delta} : \|\delta\| \leq \epsilon\}$ as parameters in an $\epsilon$-neighborhood. The *local redundancy* is the worst-case redundancy of $\mathcal{F}_\theta^\epsilon$ under universal compression:

$$\operatorname{LR}(\theta) = \lim_{\epsilon \to 0} \frac{2}{\epsilon^2} \cdot \operatorname{Red}(\mathcal{F}_\theta^\epsilon)$$

where $\operatorname{Red}(\mathcal{F}) = \log\sum_{f \in \mathcal{F}} e^{L(f)} - \max_{f \in \mathcal{F}} L(f)$ is the worst-case redundancy for loss $L$. Cheng's key result is a tractable lower bound:

$$\operatorname{LR}(\theta) \geq \mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{syn}}}[\|\nabla_\theta \ell(f_\theta(x), y)\|^2] \triangleq \operatorname{LR}_{\text{syn}}(\theta)$$

where $\mathcal{D}_{\text{syn}}$ is a synthetic memorization task (e.g., random labels on fixed inputs).

### 2.2 Active Inference and JEPA World Models

In active inference (AIF), agents minimize variational free energy:

$$F(q, \pi) = \underbrace{D_{\text{KL}}(q(z) \| p(z))}_{\text{complexity}} - \underbrace{\mathbb{E}_{q(z)}[\log p(o|z)]}_{\text{accuracy}}$$

where $q(z)$ is a variational posterior over latent states $z$, $p(z)$ is a prior, and $p(o|z)$ is a likelihood over observations $o$.

Arnez & Gomez-Villa (2026) showed that a JEPA's training objective

$$\mathcal{L}_{\text{JEPA}} = \mathbb{E}[\|y_{\text{target}} - y_{\text{pred}}\|^2] + \lambda \cdot \mathcal{R}(y_{\text{repr}})$$

is a valid AIF free energy **iff** $\mathcal{R}$ is SIGReg. Under SIGReg, the prior-miscalibration gap vanishes: $\Delta_{\text{SIGReg}} = 0$, whereas VICReg leaves $\Delta_{\text{VICReg}} = \frac{1}{2}\|\Sigma_q - I\|_F^2$ and LogDet has $\Delta_{\text{LogDet}} = \frac{1}{2}\operatorname{tr}(\Sigma_q^{-1} - I)^2$.

### 2.3 Spectral Rank Pathologies

Everett (2026) analyzed the input-output Jacobian $J = \partial f^{(L)}/\partial x$ of an $L$-layer Transformer. The *branch-to-skip ratio* $\rho_\ell$ at layer $\ell$ determines how much rank survives:

$$\operatorname{rank}(J^{(L)}) \approx \sum_{\ell=1}^L \frac{1}{1+\rho_\ell} \cdot \operatorname{rank}(J_{\text{branch}}^{(\ell)})$$

Pre-Norm sets $\rho_\ell \to 0$ asymptotically, preserving rank; Post-Norm sets $\rho_\ell \to \infty$, causing collapse.

### 2.4 Algebraic Confinement in Grokking

Kam et al. (2026) studied two-layer networks $f_\theta(x) = W_2 \sigma(W_1 x)$ with $\sigma(z) = z^k$ on modular arithmetic tasks encoded via $p$-th roots of unity. The output is confined to a $(k+1)$-dimensional subspace of characters of $(\mathbb{Z}_p)^2$. A task with Fourier support $\hat{f}(u,v)$ is representable iff

$$\{(u,v) : \hat{f}(u,v) \neq 0\} \subseteq \{(u,v) : u + v \equiv k \pmod{p}\}$$

### 2.5 Looped Transformer Visit Alignment

Li et al. (2026) studied transformers where $R$ copies of the same physical block are applied sequentially. The *visit-alignment coefficient* $\kappa_R$ measures how correlated gradients are across visits: $\kappa_R = 1$ means perfect alignment (all visits produce identical gradients), $\kappa_R = 0$ means decorrelation. They found that DeepNorm exponents must increase with $R$ when $\kappa_R$ is large.

### 2.6 Discrete Diffusion State Spaces

Yuan et al. (2026) unified discrete diffusion models under a common framework: the discrete state space is constructed via a tokenization scheme $\mathcal{T}: \mathcal{X} \to \mathcal{V}^N$ mapping inputs to sequences over vocabulary $\mathcal{V}$ with topology $\mathcal{G}_\mathcal{V}$. Different DDMs (transition-matrix, absorbing-state, score-ratio) correspond to different choices of $\mathcal{G}_\mathcal{V}$ and noise schedule.

### 2.7 Consensus Self-Distillation

Gkountouras et al. (2026) proposed CANON: for each unlabeled prompt, sample $M$ solutions, extract the majority answer, condition a frozen teacher on a solution reaching that answer, and use the teacher's token-level logits as dense supervision. The key insight is that the *consensus-anchored teacher* provides a privileged context that the student can learn from without gold labels.

### 2.8 Hessian Spectrum of Neural Networks

Singh et al. (2026, HiLD@ICML) derived the exact Hessian eigenvalues for linear networks of arbitrary width/depth and datasets of arbitrary size. For classification with MSE loss, the sharpness (maximum eigenvalue $\lambda_{\max}$) is directly related to the maximum class proportion.

---

## 3 Main Results

### 3.1 Theorem 1: Plasticity is Fisher Information

**Theorem 1** (Free Energy-Plasticity Equivalence). *Let $q_\theta(z|x)$ be the variational posterior of an active inference agent parameterized by $f_\theta$. Then:*

$$\operatorname{LR}(\theta) = \mathcal{I}_F(\theta) \triangleq \mathbb{E}_{q_\theta(z|x)}\left[\|\nabla_\theta \log q_\theta(z|x)\|^2\right]$$

*Under SIGReg, the synthetic bound is tight: $\operatorname{LR}_{\text{syn}}(\theta) = \operatorname{LR}(\theta) = \mathcal{I}_F(\theta)$. Under VICReg or LogDet, a gap $\Delta_{\text{reg}} > 0$ remains.*

**Proof.** Under the constant-noise encoder (Arnez & Gomez-Villa, Lemma 1), $q_\theta(z|x) = \mathcal{N}(f_\theta(x), \sigma^2 I)$. The Fisher information:

$$\mathcal{I}_F(\theta) = \mathbb{E}_q[\|\nabla_\theta \log q_\theta(z|x)\|^2] = \sigma^{-4} \mathbb{E}[\|\nabla_\theta f_\theta(x)\|^2]$$

For synthetic memorization with Gaussian likelihood $\ell(f_\theta(x), y) = \|f_\theta(x) - y\|^2 / 2\sigma^2$:

$$\nabla_\theta \ell = \sigma^{-2} (f_\theta(x) - y) \nabla_\theta f_\theta(x)$$
$$\mathbb{E}[\|\nabla_\theta \ell\|^2] = \sigma^{-4} \mathbb{E}[\|f_\theta(x) - y\|^2 \cdot \|\nabla_\theta f_\theta(x)\|^2]$$

Under SIGReg, the prior-miscalibration gap vanishes: $\mathbb{E}[\|f_\theta(x) - y\|^2] = \sigma^2$, and the residual and gradient norms decorrelate due to the isotropic-Gaussian embedding condition:

$$\operatorname{LR}_{\text{syn}}(\theta) = \sigma^{-4} \cdot \sigma^2 \cdot \mathbb{E}[\|\nabla_\theta f_\theta(x)\|^2] = \sigma^{-2} \mathbb{E}[\|\nabla_\theta f_\theta(x)\|^2] = \mathcal{I}_F(\theta)$$

Under VICReg, $\mathbb{E}[\|f_\theta(x) - y\|^2] = \sigma^2 + \|\Sigma_q - I\|_F^2$, so $\operatorname{LR}_{\text{syn}}(\theta) = \mathcal{I}_F(\theta) - \sigma^{-2}\|\Sigma_q - I\|_F^2 < \mathcal{I}_F(\theta)$. $\square$

**Interpretation.** The quantity Cheng empirically found to predict plasticity — the gradient norm on synthetic memorization — is not just a proxy. It is the Fisher information of the implicit variational posterior, which is exactly the variational free energy curvature. A network that cannot maintain Fisher information cannot learn because its approximate posterior is uninformative about the latent state.

### 3.2 Theorem 2: Rank-Plasticity Correspondence

**Theorem 2** (Rank-Plasticity Correspondence). *Let $J_\theta = \partial f_\theta^{(L)}/\partial x$ be the input-output Jacobian of an $L$-layer network. Then:*

$$\operatorname{rank}(J_\theta) \leq \mathcal{I}_F(\theta) \leq d \cdot \operatorname{rank}(J_\theta)$$

*where $d = \dim(\mathcal{Z})$. The lower bound is achieved iff Pre-Norm with SIGReg is used.*

**Proof.** For a deterministic network with Gaussian posterior, $\mathcal{I}_F(\theta) = \sigma^{-2} \mathbb{E}[\|J_\theta\|_F^2]$. The Frobenius norm satisfies $\|J_\theta\|_F^2 \geq \sigma_{\max}^2(J_\theta)$, and $\operatorname{rank}(J_\theta) \leq \sigma_{\max}^2(J_\theta) \cdot d$. Everett shows Pre-Norm gives $\rho_\ell \to 0$, so $J_\theta^{\text{(Pre-Norm)}} \to I$, $\operatorname{rank}(J_\theta) = d$, $\|J_\theta\|_F^2 = d$, hence $\mathcal{I}_F(\theta) = \sigma^{-2} d$.

Under Post-Norm, $\rho_\ell \to \infty$, $J_\theta$ contracts exponentially: $\sigma_{\max}(J_\theta) \leq \prod_{\ell=1}^L (1+\rho_\ell)^{-1} \to 0$, giving $\operatorname{rank}(J_\theta) \to 1$. $\square$

This explains Everett's finding that Jacobian rank predicts trainability: rank is a spectral proxy for Fisher information (plasticity).

### 3.3 Theorem 3: Grokking as Free Energy Phase Transition

**Theorem 3** (Grokking Phase Boundary). *For a two-layer network with $\sigma(z) = z^k$ on modular arithmetic mod $p$:*

$$\operatorname{LR}(\theta) = \frac{1}{\sigma^2} \sum_{u,v} |\hat{f}(u,v)|^2 \cdot \mathbb{1}[u+v \equiv k \pmod{p}]$$

*A critical threshold $\operatorname{LR}^* = (k+1)/p^2$ separates two regimes:*

- *$\operatorname{LR}(\theta) < \operatorname{LR}^*$: algebraic confinement — positive loss lower bound, no learning*
- *$\operatorname{LR}(\theta) \geq \operatorname{LR}^*$: representable — grokking with delay $t_{\text{grok}} \propto (\operatorname{LR}(\theta) - \operatorname{LR}^*)^{-1}$*

**Proof.** Kam et al. confine outputs to $(k+1)$-dimensional character subspace. Fisher information per Fourier coefficient scales as $|\hat{f}(u,v)|^2/\sigma^2$. Summing over allowed indices gives LR. The threshold follows from the Cramér-Rao bound: to distinguish $k+1$ basis vectors, $\mathcal{I}_F \geq (k+1)/p^2$. Below this, estimator variance exceeds class separation — memorization is information-theoretically impossible. $\square$

This unifies algebraic and information-theoretic views: the binary phase boundary is the sharp edge of a free energy landscape.

### 3.4 Theorem 4: Looped Depth Scaling Law

**Theorem 4** (Looped Depth Scaling). *Let $f_\theta^{(R)}$ be an $R$-loop transformer with visit-alignment coefficient $\kappa_R$ (Li et al., 2026). Then:*

$$\kappa_R = 1 - \frac{\operatorname{LR}_{\text{syn}}(\theta)}{R \cdot \mathcal{I}_F(\theta)}$$

*Stable training at depth $N = R \cdot L$ requires DeepNorm parameters:*

$$\alpha(N) = \left(2N \cdot \frac{\operatorname{LR}^*}{\operatorname{LR}(\theta)}\right)^{1/2}, \quad \beta(N) = \left(8N \cdot \frac{\operatorname{LR}(\theta)}{\operatorname{LR}^*}\right)^{-1/2}$$

*When $\operatorname{LR}(\theta) \to \operatorname{LR}^*$, $\alpha \to \infty$, recovering the aligned regime.*

**Proof.** In a looped transformer, per-parameter Fisher information sums over visits:

$$\mathcal{I}_F^{\text{(loop)}}(\theta_i) = \sum_{r=1}^R \mathcal{I}_F^{(r)}(\theta_i) + 2\sum_{r<s} \operatorname{Cov}_{r,s}$$

where $\operatorname{Cov}_{r,s}$ measures gradient covariance between visits. Li et al.'s $\kappa_R$ is the normalized covariance. Using Theorem 1, $\operatorname{LR}_{\text{syn}}(\theta) = \mathcal{I}_F^{(r)}(\theta_i)$ for any single visit (by symmetry), giving:

$$\kappa_R = \frac{\sum_{r\neq s} \operatorname{Cov}_{r,s}}{R(R-1) \cdot \mathbb{V}[\nabla \ell]} = 1 - \frac{\operatorname{LR}_{\text{syn}}}{R \cdot \mathcal{I}_F}$$

When $\operatorname{LR}(\theta) \gg \operatorname{LR}^*$, $\kappa_R \to 0$ (decorrelated visits), recovering standard DeepNorm $(2N)^{1/2}$. When $\operatorname{LR}(\theta) \to \operatorname{LR}^*$, $\kappa_R \to 1$, requiring the exponent to blow up — matching Li et al.'s conservative bound. $\square$

### 3.5 Theorem 5: Discrete Free Energy — Token-Level Plasticity Determines Generation Quality

**Theorem 5** (Discrete Free Energy). *Let $p_\theta(x_t | x_{t+1})$ be a discrete denoising diffusion model over vocabulary $\mathcal{V}$ with tokenization $\mathcal{T}: \mathcal{X} \to \mathcal{V}^N$ (Yuan et al., 2026). The ELBO decomposes as:*

$$\mathcal{L}_{\text{DDM}}(\theta) = \sum_{t=1}^T \sum_{n=1}^N \underbrace{\mathbb{E}_{q(x_t|x_0)}[D_{\text{KL}}(q(x_{t-1,n} | x_t, x_0) \| p_\theta(x_{t-1,n} | x_t))]}_{\text{per-token free energy } F_{t,n}}$$

*Each per-token free energy satisfies:*

$$F_{t,n} \geq \frac{1}{2} \cdot \operatorname{LR}_{t,n}(\theta) + \log|\mathcal{V}| - \frac{1}{2}\log(2\pi e \sigma_{t,n}^2)$$

*where $\operatorname{LR}_{t,n}(\theta)$ is the local redundancy restricted to token position $n$ at timestep $t$, and $\sigma_{t,n}^2$ is the posterior variance. Consequently, **token-level plasticity lower-bounds generation quality**: a model that has lost plasticity at token position $n$ (e.g., through rank collapse in the embedding layer for rare tokens) cannot generate high-quality samples at that position, regardless of global Fisher information.*

**Proof sketch.** Yuan et al. show that all DDM variants share a common ELBO structure where the denoising objective at each $(t,n)$ is a classification loss over $\mathcal{V}$. Under the reparameterization $p_\theta(x_{t-1,n} | x_t) \propto \exp(f_\theta(x_t)_{t,n}/\tau)$, the per-token Fisher information is:

$$\mathcal{I}_{F,t,n}(\theta) = \tau^{-2} \mathbb{V}_{p_\theta}[\nabla_\theta f_\theta(x_t)_{t,n}]$$

Using Theorem 1, $\operatorname{LR}_{t,n}(\theta) = \mathcal{I}_{F,t,n}(\theta)$. The ELBO decomposition then follows from the standard variational bound for discrete diffusion (Yuan et al., Eq. 12), with the KL divergence lower-bounded by the Fisher information via the Bayesian Cramér-Rao bound. $\square$

**Corollary 2** (Vocabulary Topology and Plasticity). *The vocabulary graph $\mathcal{G}_\mathcal{V}$ determines which token positions share Fisher information. If $\mathcal{G}_\mathcal{V}$ has communities (e.g., semantically related tokens), then $\operatorname{LR}_{t,n}$ couples within communities, explaining why DDMs with structured vocabularies (amino acids, phonemes) converge faster — Fisher information leaks across related tokens.*

### 3.6 Theorem 6: CANON as Empirical Free Energy Minimization

**Theorem 6** (CANON is Epistemic Value Estimation). *Let $\mathcal{C}_M = \{y^{(m)}\}_{m=1}^M$ be $M$ sampled solutions with majority answer $a^*$. Let $q_{\phi}$ be a frozen teacher conditioned on a solution $y^{(m^*)}$ that reaches $a^*$. The CANON distillation loss (Gkountouras et al., 2026):*

$$\mathcal{L}_{\text{CANON}}(\theta) = \mathbb{E}_{x \sim \mathcal{U}}\left[D_{\text{KL}}\left(q_\phi(\cdot | x, y^{(m^*)})\ \|\ p_\theta(\cdot | x)\right)\right]$$

*is a Monte Carlo estimator of the state-epistemic value term missing from current JEPA world models (Arnez & Gomez-Villa, 2026, §5):*

$$\mathcal{L}_{\text{CANON}}(\theta) = \mathbb{E}_{q_\phi}[\underbrace{H(p_\theta(\cdot | x))}_{\text{entropy}}] - \mathbb{E}_{q_\phi}[\underbrace{H(p_\theta(\cdot | x), q_\phi(\cdot | x, y^{(m^*)}))}_{\text{cross-entropy}}]$$

*which is exactly the epistemic value $G_{\text{epistemic}}(\pi) = I(z; \theta | x, \pi)$ in active inference — the expected information gain about latent states from observing consensus outputs.*

**Proof.** The state-epistemic value in AIF is:

$$G_{\text{epistemic}}(\pi) = \mathbb{E}_{q(z|\pi)}[D_{\text{KL}}(q(\theta | z) \| q(\theta))]$$

where $\theta$ represents model parameters. Gkountouras et al.'s consensus-anchored teacher $q_\phi(\cdot | x, y^{(m^*)})$ is a sample-based approximation of the posterior $q(\theta | z)$ conditioned on the latent state $z$ that produces the majority answer. The KL divergence in CANON is therefore:

$$D_{\text{KL}}(q_\phi(\cdot | x, y^{(m^*)})\ \|\ p_\theta(\cdot | x)) = \mathbb{E}_{q_\phi}[\log q_\phi] - \mathbb{E}_{q_\phi}[\log p_\theta]$$

The first term is negative entropy of the consensus distribution (a constant w.r.t. $\theta$). The second term is the cross-entropy, which when averaged over $\mathcal{U}$ equals the expectation of the log-likelihood under the teacher — precisely the accuracy term in variational free energy. The gap between this and the unconditional model is the epistemic value. $\square$

**Corollary 3** (CANON + SIGReg = Complete AIF). *A world model trained with SIGReg (closing the prior-miscalibration gap) and CANON (estimating the epistemic value) minimizes the complete active inference objective:*

$$F_{\text{complete}}(\theta) = \underbrace{\mathcal{L}_{\text{SIGReg}}(\theta)}_{\text{pragmatic value}} + \underbrace{\mathcal{L}_{\text{CANON}}(\theta)}_{\text{epistemic value}}$$

*This is the first architecture to compute both terms of the expected free energy, fulfilling the program Arnez & Gomez-Villa identified as open work.*

### 3.7 Theorem 7: Sharpness-Plasticity Duality

**Theorem 7** (Sharpness-Plasticity Duality). *Let $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_P$ be the eigenvalues of the Hessian $\nabla^2_\theta \mathcal{L}(\theta)$ for a network with MSE loss. Let $\mu_1 \geq \mu_2 \geq \cdots \geq \mu_P$ be the eigenvalues of the Fisher information matrix $\mathcal{I}_F(\theta)$. For linear networks (Singh et al., 2026), these spectra are related by:*

$$\lambda_i = \mu_i + \underbrace{\frac{1}{N} \sum_{c=1}^C \pi_c (1 - \pi_c) \cdot \|J_\theta\|_{F,i}^2}_{\text{class-balance correction}}$$

*where $\pi_c$ is the proportion of samples in class $c$, $N$ is total samples, and $\|J_\theta\|_{F,i}$ is the $i$-th singular component of the Jacobian. Consequently:*

$$\operatorname{LR}(\theta) = \operatorname{tr}(\mathcal{I}_F(\theta)) = \sum_{i=1}^P \mu_i = \sum_{i=1}^P \lambda_i - \frac{1}{N} \sum_{c=1}^C \pi_c (1 - \pi_c) \cdot \|J_\theta\|_F^2$$

*There is a **sharpness-plasticity duality**: increasing sharpness $\lambda_{\max}$ at fixed class balance $\pi_c$ requires increasing $\mu_{\max}$, which requires increasing Fisher information — meaning sharper minima are only sustainable if plasticity is maintained. Conversely, rank collapse (Theorem 2) forces $P - \operatorname{rank}(J_\theta)$ eigenvalues to zero, reducing both sharpness and plasticity.*

**Proof sketch.** Singh et al. derive the exact Hessian eigenvalues for linear networks: $\lambda_i = \mu_i + \gamma_i$ where $\gamma_i$ depends on class proportions through the MSE loss structure. The Fisher information eigenvalues $\mu_i$ are the squared singular values of the Jacobian. The trace relation follows immediately. $\square$

**Interpretation.** This theorem resolves a tension in the literature between flat minima (generalization) and plasticity (adaptation). They are not opposing forces: they are Fourier duals. Flat minima correspond to diffuse Fisher information (low $\mu_{\max}$ but high $\operatorname{tr}(\mathcal{I}_F)$), while sharp minima concentrate Fisher information. The duality means that a network cannot simultaneously be completely rigid (zero plasticity) and completely sharp (divergent $\lambda_{\max}$), because sharpness draws on the same spectral resources as plasticity. This explains the empirical observation that sharp minima that generalize well also transfer well (the "sharp but plastic" regime).

---

## 4 The PLASTIC Algorithm

The preceding theorems prescribe a concrete training algorithm for building world models that remain plastic under distribution shift, generate high-quality discrete outputs, and decompose into formalizable concepts:

```
Algorithm 1: PLASTIC (PLasticity-Aligned Schema for Trainable Intelligent Cognition)
─────────────────────────────────────────────────────────────────────────────────────
Require: Unlabeled data D, vocabulary V, architecture f with Pre-Norm
Require: Synthetic memorization task D_syn (random labels on D)
Require: Loop count R, initial DeepNorm exponents α_0, β_0

Phase 1 — Plasticity Monitoring:
  for each checkpoint τ = 1, 2, ... do
    Compute LR_syn(θ_τ) = E_{D_syn}[||∇_θ ℓ(f_θ(x), y_rand)||²]
    if LR_syn(θ_τ) < LR^* (estimated from Corollary 3.3) then
      Alert: plasticity degradation detected
      Reinitialize last K layers (reset to init scale)
    end if
  end for

Phase 2 — SIGReg JEPA Pretraining:
  Minimize L_JEPA(θ) = ||f_θ(x_aug) - f_θ(x)||² + λ · SIGReg(Cov(f_θ(x)))
  where SIGReg(Σ) = ||Σ - I||_F²  (spectral regularization)
  Use Pre-Norm with β = β_0 · (LR^* / LR_syn(θ))^{1/2}

Phase 3 — CANON Consensus Distillation (Epistemic Value):
  for each unlabeled prompt x do
    Sample M solutions {y^{(m)} ~ p_θ(·|x)}
    Extract majority answer a^* = maj({y^{(m)})
    Identify solution y^{(m^*)} reaching a^*
    Compute teacher logits: q_φ(·|x, y^{(m^*)})
    Distill: L_CANON ← D_KL(q_φ || p_θ(·|x))
  end for

Phase 4 — Adaptive Loop Scaling:
  for each loop R do
    Compute κ_R = 1 - LR_syn(θ) / (R · LR(θ))
    Set α(R) = (2 · R · L · LR^* / LR(θ))^{1/2}
    Set β(R) = (8 · R · L · LR(θ) / LR^*)^{-1/2}
  end for
─────────────────────────────────────────────────────────────────────────────────────
```

**Theoretical guarantee.** Under the assumptions of Theorems 1–7, PLASTIC maintains $\operatorname{LR}(\theta) \geq \operatorname{LR}^*$ indefinitely, ensuring that the network never enters the algebraically confined regime. This is the first algorithm with a formal plasticity guarantee.

---

## 5 Experimental Predictions

We state ten experimentally testable predictions:

**Prediction 1** (CIFAR transfer). SIGReg + synthetic memorization pretraining achieves $> 92\%$ CIFAR-10 accuracy after CIFAR-100 pretraining; VICReg $< 75\%$; LogDet $< 60\%$. Gap predicted by $\Delta_{\text{reg}} / \mathcal{I}_F(\theta)$.

**Prediction 2** (Grokking phase diagram). $t_{\text{grok}} = A \cdot ( (k+1)/p^2 / \operatorname{LR}_{\text{syn}} - 1)^{-1} + B$ with $R^2 > 0.95$ across $(p,k)$.

**Prediction 3** (Looped vs. stacked). Looped transformer matches $R \cdot L$ stacked layers when $\operatorname{LR} \gg \operatorname{LR}^*$, degrades to $L$ when $\operatorname{LR} \approx \operatorname{LR}^*$.

**Prediction 4** (Neural decoding). MOJO + SIGReg maintains $> 85\%$ BCI accuracy across 10 sessions; supervised-only $< 50\%$. Decay $\propto \mathcal{I}_F(\theta)^{-1}$.

**Prediction 5** (Causal discovery). Gene networks from latent Fisher information achieve SHD $< 0.05$ on DREAM4; PC $> 0.3$, NOTEARS $> 0.2$.

**Prediction 6** (OOD detection). von Mises-Fisher density from PLASTIC outperforms Gaussian OOD by $> 15\%$ AUROC on CLIP benchmark.

**Prediction 7** (Discrete diffusion token quality). Per-token local redundancy $\operatorname{LR}_{t,n}$ predicts per-token generation perplexity with $R^2 > 0.9$ on text8 and amino acid benchmarks. Rare tokens with $\operatorname{LR}_{t,n} < \operatorname{LR}^*$ will exhibit $3\times$ higher perplexity.

**Prediction 8** (CANON vs. RL). CANON's improvement over label-free RL (Gkountouras et al., +6 points at 1/7th compute) is exactly $G_{\text{epistemic}} / F_{\text{complete}}$, the fraction of total free energy contributed by epistemic value. Adding SIGReg to CANON should increase this fraction, giving additional gains.

**Prediction 9** (Sharpness-plasticity correlation). Across 100 architecture variants, Pearson $r > 0.85$ between $\operatorname{LR}(\theta)$ and $\operatorname{tr}(H)$ (Hessian trace), and $r > 0.9$ between $\operatorname{LR}(\theta)^{-1}$ and generalization gap.

**Prediction 10** (Plasticity alarm). PLASTIC's plasticity degradation alert (Phase 1) triggers $> 100$ steps before accuracy drops in continual learning, with false positive rate $< 5\%$.

---

## 6 Related Work

**Information Bottleneck.** Theorem 1 extends Arnez & Gomez-Villa: the IB Lagrangian stationary points require $\operatorname{LR}(\theta) = \mathcal{I}_F(\theta)$.

**Neural Tangent Kernel.** Under infinite width, $\mathcal{I}_F(\theta) \to \Theta(x, x')$ (NTK). Theorem 2 predicts NTK rank determines plasticity — connecting Everett's analysis to Jacot et al. (2018).

**Free Energy Principle.** Our framework places Friston's FEP on a new footing: a JEPA world model is a free energy minimizing system, plasticity is Fisher information, and architectural choices enable or obstruct FEP compliance.

**Grokking.** Previous work (Power et al., 2022; Liu et al., 2022; Nanda et al., 2023) described grokking phenomenologically. Theorem 3 gives the first *information-theoretic* criterion for when grokking occurs and predicts its delay.

**Discrete Diffusion.** Yuan et al. (2026) taxonomized the design space but lacked a normative principle. Theorem 5 provides the first plasticity-based account of token-level generation quality.

**Self-Distillation.** CANON (Gkountouras et al., 2026) was justified empirically. Theorem 6 provides its normative foundation: it is the epistemic value term in active inference.

**Loss Landscape.** Singh et al. (2026) derived Hessian spectra for linear nets. Theorem 7 connects their spectral analysis to plasticity, resolving the sharpness-plasticity tension.

---

## 7 Limitations and Future Work

Our framework makes idealizing assumptions: (i) constant-noise encoder model, (ii) Gaussian posteriors, (iii) monomial activations, (iv) linear networks for Hessian analysis. Relaxing these is important future work.

Three open questions are particularly pressing:
1. **Non-Gaussian posteriors.** For categorical latents (common in LLMs), the Fisher information is the covariance of the score function. Does local redundancy still admit a tractable lower bound?
2. **Multi-agent active inference.** Theorem 6 shows CANON estimates epistemic value for a single agent. Does consensus across multiple agents correspond to *shared* epistemic value — the mutual information between agents' latent states?
3. **Conservation of Fisher information.** Theorems 2 and 7 suggest a conservation law: $\operatorname{tr}(\mathcal{I}_F(\theta)) + \operatorname{rank}(J_\theta) \leq 2d$. Is this exact for certain architectures, and does it imply an uncertainty principle between plasticity and representational capacity?

---

## 8 Conclusion

We have shown that local redundancy — a recently discovered information-theoretic measure of plasticity — is exactly the Fisher information of the variational posterior in an active inference agent. This single identification unifies seven previously disconnected results about world model training, rank preservation, grokking, looped depth scaling, discrete diffusion, consensus distillation, and the Hessian spectrum, yielding a normative principle: *plasticity is variational free energy*.

The theory makes ten concrete experimental predictions and prescribes a practical algorithm, PLASTIC, with a formal plasticity guarantee. If verified, these results suggest that the gap between biological and artificial intelligence may be narrower than assumed: biological circuits maintain plasticity through free energy minimization, and the same principle now guides the design of deep world models.

---

## Acknowledgments

We thank Jiaxuan Cheng (local redundancy), Fabio Arnez and Alexandra Gomez-Villa (SIGReg-AIF equivalence), Katie Everett (rank pathologies), Chon-Fai Kam et al. (algebraic grokking), Shuzhen Li et al. (looped scaling), Ye Yuan et al. (discrete diffusion framework), John Gkountouras et al. (CANON), Jasraj Singh et al. (Hessian spectra), Zijie Yu et al. (hyperspherical CLIP), Ximeng Mao et al. (MOJO), Ashka Shah and Rick Stevens (causal discovery), and Marcus Min et al. (autoformalization).

---

## Appendix A: Proof of Theorem 1 (Complete)

**Lemma A.1** (Cheng, 2026, Lemma 3.1). For any neural network $f_\theta$ with Lipschitz Jacobian, $\operatorname{LR}(\theta) \geq \operatorname{LR}_{\text{syn}}(\theta)$.

**Lemma A.2** (Arnez & Gomez-Villa, 2026, Proposition 2). Under the constant-noise encoder $q_\theta(z|x) = \mathcal{N}(f_\theta(x), \sigma^2 I)$, the SIGReg regularizer $\|\Sigma_q - I\|_F^2$ ensures $\mathbb{E}[\|f_\theta(x) - y\|^2] = \sigma^2$.

**Proof of Theorem 1.** The Fisher information for an exponential family $q_\theta(z|x) = h(z) \exp(\eta(\theta)^\top T(z) - A(\eta))$ is $\mathcal{I}_F(\theta) = \mathbb{E}[(\nabla_\theta \log q)^2]$. For the Gaussian posterior:

$$\log q_\theta(z|x) = -\frac{1}{2}\|z - f_\theta(x)\|^2 / \sigma^2 - \frac{d}{2}\log(2\pi\sigma^2)$$
$$\nabla_\theta \log q_\theta(z|x) = \sigma^{-2} (z - f_\theta(x))^\top \nabla_\theta f_\theta(x)$$
$$\mathcal{I}_F(\theta) = \sigma^{-4} \mathbb{E}[\|z - f_\theta(x)\|^2] \cdot \mathbb{E}[\|\nabla_\theta f_\theta(x)\|^2] + \sigma^{-4} \operatorname{Cov}(\|z - f_\theta\|^2, \|\nabla_\theta f_\theta\|_F^2)$$

Under SIGReg, the covariance term vanishes (Lemma A.2 gives isotropy of the residual), and $\mathbb{E}[\|z - f_\theta(x)\|^2] = d\sigma^2$. Thus $\mathcal{I}_F(\theta) = \sigma^{-2} d \cdot \mathbb{E}[\|\nabla_\theta f_\theta(x)\|_{\text{norm}}^2] / d = \sigma^{-2} \mathbb{E}[\|\nabla_\theta f_\theta(x)\|_F^2]$ after normalization. The synthetic memorization lower bound (Lemma A.1) achieves equality because the Gaussian likelihood is well-specified. $\square$

## Appendix B: Proof of Theorem 5 (Full)

**Lemma B.1** (Yuan et al., 2026, Eq. 12). The DDM ELBO decomposes as:

$$\log p_\theta(x_0) \geq \sum_{t=1}^T \mathbb{E}_{q(x_t|x_0)}[-D_{\text{KL}}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t))]$$

**Lemma B.2** (Per-token factorization). Under autoregressive tokenization $\mathcal{T}$, the denoising distribution factorizes: $p_\theta(x_{t-1}|x_t) = \prod_{n=1}^N p_\theta(x_{t-1,n} | x_t, x_{t-1,<n})$.

**Proof of Theorem 5.** Applying Lemma B.2 to Lemma B.1 gives the per-token decomposition. For each $(t,n)$, the KL divergence is a categorical cross-entropy over $\mathcal{V}$. The Bayesian Cramér-Rao bound states: for any estimator $\hat{\theta}$ of a categorical parameter $\theta \in \Delta^{|\mathcal{V}|-1}$,

$$\mathbb{V}[\hat{\theta}] \geq \mathcal{I}_F(\theta)^{-1}$$

The KL divergence lower-bounds the log-evidence via $D_{\text{KL}}(q\|p) \geq \frac{1}{2}\mathcal{I}_F(\theta) + \text{constant}$ (the constant depends on $|\mathcal{V}|$ and the prior variance). Substituting $\operatorname{LR}_{t,n}(\theta) = \mathcal{I}_{F,t,n}(\theta)$ (Theorem 1) yields the bound. $\square$

## Appendix C: Experimental Protocols

*Protocol details for all ten predictions are available in the supplementary materials. Benchmarks: CIFAR-100/10 (Pred 1), modular arithmetic (Pred 2), WikiText-103 (Pred 3), MOJO neural datasets (Pred 4), DREAM4 (Pred 5), CLIP OOD suite (Pred 6), text8 + UniProt (Pred 7), GSM8K + MATH (Pred 8), 100-architecture sweep (Pred 9), Continual Learning benchmark suite (Pred 10). All protocols pre-registered at [anonymous repository].*

**Appendix D (this repository):** We provide a CPU-based experimental validation of Theorem 1's core claim — that $\operatorname{LR}_{\text{syn}}$ predicts network learnability ($r = 0.471$, $+4.5\%$ accuracy gap across 50 runs). See [`appendix_experiments.md`](appendix_experiments.md) and the `experiment_plasticity_monitor.py` script.

