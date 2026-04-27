# Related Work / SOTA (comparability‑aware)

This note tracks papers that are most relevant to **SAFE‑TTA** (certificate‑guided, risk‑controlled test‑time action selection) for **cross‑subject MI‑EEG decoding** under **strict LOSO**:

- **Strict LOSO**: the test subject is never used for training or calibration.
- **No target labels at test time**: any “certificate” or selection rule must use only **unlabeled** target data at selection time.
- **Frozen classifier** (for the CSP+LDA line): we do **not** update classifier parameters at test time; we only select an action (alignment / feature / classifier family) and possibly gate/fallback.

This file is **not** a full literature survey; it is a “design map” for upgrading the **certificate** when standard proxies (entropy, marginal‑uniformity, evidence‑NLL, etc.) show weak correlation with true accuracy.

---

## A) EEG alignment / distribution matching (candidate action families)

These methods mainly expand the **action set** \(\mathcal A\) (what we can select at test time). They do **not** solve certificate validity by themselves, but they determine the available *headroom*.

### A.1 Euclidean alignment (EA) and variants
- EA is widely used as a strong, stable “default” alignment for cross‑subject/session EEG.
- Practical note repeatedly emphasized in the EA follow‑up literature: *placement matters* (often after temporal filtering), and CAR can interact with alignment.

Repo mapping:
- `ea-csp-lda` is our anchor action in many SAFE‑TTA runs.

### A.2 Riemannian alignment on SPD covariances (RPA / TSA)
- **RPA** (Riemannian Procrustes Analysis) matches source/target covariance distributions on the SPD manifold via *translation + scaling + rotation*.
- **TSA** (Tangent Space Alignment) performs alignment in the tangent space (vectorized log‑mapped covariances), reporting broad gains across many BCI datasets in the original paper.

Repo mapping:
- We currently treat these as *candidate families* (not paper‑faithful reimplementations in all details), and we explicitly label their naming in our assets.

---

## B) Test‑time adaptation (TTA) stability: why “proxy ≠ accuracy” is expected

These works motivate why naive unlabeled objectives can **hurt** some subjects, especially under low‑SNR and strong shift.

### B.1 Tent (entropy minimization; “proxy can mislead”)
- Core idea: adapt a model at test time by minimizing prediction entropy on unlabeled target data.
- Known issue: entropy minimization can cause **over‑confidence** and negative transfer when the proxy is misaligned with target risk.

### B.2 EATA (sample selection + anti‑forgetting)
- Key message: **sample selection** and conservative regularization are crucial to stabilize test‑time adaptation and avoid catastrophic drift.

SAFE‑TTA takeaway:
- We should treat “certificate validity” as the central scientific problem: the proxy/certificate must be **risk‑aligned** and **rejectable** (i.e., it should have a reliable “don’t adapt / fall back” region).

---

## C) Unsupervised model selection / validators (closest to our “certificate” problem)

We can interpret each candidate action \(a\in\mathcal A\) as a “model”. The certificate is an **unsupervised validator** \(C(a;X^{(t)})\) used to choose the best model without target labels.

### C.1 DEV (ICML 2019): Deep Embedded Validation
**Problem:** pick DA hyper‑parameters/models without target labels.  
**Idea:** construct an estimator of target risk using only:
- labeled source validation,
- a density‑ratio / domain‑classifier estimate between source and target feature distributions.

SAFE‑TTA port:
- Use **feature‑space ratio estimation** to build an **importance‑weighted CV** score for each candidate action (a “risk‑aligned” certificate).
- This is conceptually aligned with our existing `iwcv` selector family.

### C.2 IWCV (JMLR 2007): Importance‑Weighted Cross‑Validation under covariate shift
**Problem:** choose models under covariate shift.  
**Idea:** under **covariate shift** \(p_S(x)\neq p_T(x)\) but \(p_S(y\mid x)=p_T(y\mid x)\), the target risk can be written as
\[
R_T(f)=\mathbb{E}_{(x,y)\sim p_T}\big[\ell(f(x),y)\big]
=\mathbb{E}_{(x,y)\sim p_S}\big[w(x)\,\ell(f(x),y)\big],
\quad
w(x)=\frac{p_T(x)}{p_S(x)}.
\]
Estimate \(w(x)\) (e.g., via a domain classifier), and use importance‑weighted CV to select the model/hyper‑parameter.

Why this matters for SAFE‑TTA:
- Our current “prediction‑distribution proxies” (entropy / marginal uniformity / evidence) are **not** tied to a target‑risk estimator of CSP+LDA.
- IWCV/DEV‑style certificates are closer to “what accuracy measures”: a risk estimate.

### C.3 SND (ICCV 2021): Soft Neighborhood Density (validator)
**Problem:** unsupervised validation for DA without target labels.
**Idea:** a good model should embed target samples of the same class nearby, forming *dense neighborhoods* in feature space. SND builds a validation criterion from the **entropy of similarity distributions** between target samples in representation space (a “soft neighborhood density” score), and shows it is effective for tuning UDA hyper‑parameters and training iterations.

SAFE‑TTA port (CSP/LDA feature space is enough):
- Compute an SND‑like score on target LDA feature vectors \(z_i\), using soft pseudo‑labels \(q_i\) as class‑conditional structure hints.
- This can produce a certificate less sensitive to class‑prior assumptions than “KL‑to‑uniform”.

### C.4 MixVal (NeurIPS 2023): Mixed Samples as Probes
**Problem:** unsupervised model selection for DA.  
**Idea:** use *mixed target samples with pseudo labels* as **probes** to directly probe target structure, via two probe types:
- **intra‑cluster** mixed probes: evaluate neighborhood density (SND‑like behavior),
- **inter‑cluster** mixed probes: probe the classification boundary.
MixVal is positioned as combining strengths of Entropy‑based and SND‑based validators, improving stability of model selection across UDA methods.

SAFE‑TTA port:
- We already implement MixVal‑style probes in the **LDA feature space** (mixing LDA features and reading out probabilities).
- Next design question is: which probe statistics correlate best with true \(\Delta\)acc across candidate actions (e.g., “hard‑major” mode, intra/inter ratio, margin/entropy on probes).

### C.5 Transfer Score (ICLR 2024): “can we evaluate DA models without target labels?”
**Problem:** evaluate/choose DA models without target labels.  
**Idea:** propose a scoring framework that empirically tracks transfer performance better than naive heuristics.

SAFE‑TTA takeaway:
- Treat validators as a first‑class research object: define a score, validate it by oracle‑gap reduction, and explicitly report failures.

---

## D) “What should we try next” (certificate redesign menu)

Given the evidence from our runs (oracle headroom exists but oracle gap remains large), the most productive certificate upgrades are:

1) **Risk‑aligned validators** (DEV/IWCV style)
   - Build a target‑risk estimator for each candidate using:
     - source labeled loss (e.g., LDA NLL on labeled source),
     - a domain ratio estimate \(w(x)\) between candidate‑transformed source and target feature distributions.
   - Expected benefit: stronger monotonicity with accuracy; clearer theoretical assumptions.

2) **Structure‑based validators** (SND style)
   - Use neighborhood density / cluster consistency on target features \(z_i\), with soft pseudo‑labels.
   - Expected benefit: detect “looks confident but wrong” collapse modes.

3) **Probe‑based validators** (MixVal style)
   - Expand probe features beyond a single mean CE:
     - intra vs inter probe score gap,
     - probe margin distribution (quantiles),
     - instability under multiple \(\lambda\) schedules.
   - Expected benefit: better sensitivity to decision boundary geometry.

4) **Label‑shift aware components** (optional)
   - If strong label shift exists, explicitly estimate target priors (e.g., BBSE‑style) and avoid forcing uniform marginals.

---

## References (download list)

To keep this repo self‑contained, we prefer storing papers (or extracted notes) in `papers/` and linking them here.

Validator / model selection:
- You et al., **“Towards Accurate Model Selection in Deep Unsupervised Domain Adaptation”**, ICML 2019 (DEV).  
- Sugiyama et al., **“Covariate Shift Adaptation by Importance Weighted Cross Validation”**, JMLR 2007 (IWCV).  
- Li et al., **“Tune it the Right Way: Unsupervised Validation of Domain Adaptation via Soft Neighborhood Density”**, ICCV 2021 (SND).  
- Zeng et al., **“Mixed Samples as Probes for Unsupervised Model Selection in Domain Adaptation”**, NeurIPS 2023 (MixVal).  
- Jain et al., **“Can We Evaluate Domain Adaptation Models Without Target-Domain Labels?”**, ICLR 2024 (Transfer Score).

TTA stability:
- Wang et al., **“Tent: Fully Test-Time Adaptation by Entropy Minimization”**, ICLR 2021.  
- Niu et al., **“Efficient Test-Time Model Adaptation without Forgetting (EATA)”**, ICML 2022.

EEG alignment baselines:
- Rodrigues et al., **“Riemannian Procrustes Analysis: Transfer Learning for Brain–Computer Interfaces”**, IEEE TBME 2018.  
- Bleuzé et al., **“Tangent space alignment: Transfer learning for Brain‑Computer Interface”**, Frontiers in Human Neuroscience 2022.  
- Wu, **“Revisiting Euclidean Alignment for Transfer Learning in EEG‑Based Brain‑Computer Interfaces”**, (survey / tutorial style).
