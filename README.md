# CV2 Final Project — Cross-Signer Generalization on WLASL

**Domain C — Gesture & Sign Language Recognition.** Master in Artificial Intelligence · Universidade de Vigo · ESEI · 2025–26.

Deliverable: a Kaggle notebook ([rqe_transformer_wlasl.ipynb](rqe_transformer_wlasl.ipynb)) that evaluates whether a Transformer encoder fed with **Relative Quantization Encoding (RQE)** of 3D skeletal landmarks reduces the cross-signer generalization gap on WLASL100, compared to a baseline that ignores temporal structure.

> The notebook contains the full project content (problem, architecture, experiments, failure analysis, thesis direction). This README mirrors the justifications so they are visible without opening the notebook.

---

## 1. Problem statement & hypothesis (Steps 1–2)

**Open challenge (from the survey).** Zhou et al. (2023, arXiv:2310.13039) list **cross-subject / cross-signer generalization** as an unsolved problem in pose-based action recognition: models trained on one set of signers tend to overfit to body proportions, signing style, and camera placement, and drop significantly on unseen signers. In WLASL this is the dominant generalization gap because the dataset has few examples per gloss and signers are not uniformly distributed across classes.

**Why it matters.** A sign-language recognizer that only works for the people it was trained on is not useful in practice (accessibility, HCI). The gap between same-signer and unseen-signer accuracy is what determines deployability.

**Research question.** Can a Transformer encoder operating on RQE-normalized 3D landmarks reduce the cross-signer gap on WLASL100, compared to a non-temporal baseline on the same input?

**Hypothesis.** Partially yes. Removing signer-specific translation and torso-scale at the input level should make the representation more signer-invariant by construction, and the Transformer's global attention should weight informative frames (hand-shape transitions) over body posture — where most signer-style bias lives. We expect the gap to shrink, not vanish: handedness, signing speed, and finger anatomy survive RQE.

## 2. Architecture selection & justification (Step 3)

**Chosen: RQE-Transformer.** Linear projection of per-frame RQE-normalized landmarks to `d_model=256`, learned positional embedding, 4-layer / 8-head pre-norm Transformer encoder, masked temporal global average pool, linear head.

**Why this addresses cross-signer generalization (mechanistically).**
- **RQE at the input** removes the two clearest signer-identity signals (absolute body position and absolute body size) *before* the network sees them, so the model cannot use them as shortcuts. Representational prior, not learned — generalizes to unseen signers by construction.
- **Self-attention over the whole sequence** lets the model weight informative sub-intervals (hand-shape transitions, contact moments) independently of when they happen. Different signers sign at different speeds; attention is more tolerant of this than a fixed convolutional receptive field.
- **Per-frame token = whole-body pose flattened.** Cross-joint structure is learned by attention rather than baked into a fixed graph, avoiding hardcoding a single skeletal topology.

**Alternative considered and rejected: ST-GCN** (Yan et al. 2018; 2022–2024 variants). ST-GCN encodes the skeleton topology as a **fixed adjacency matrix** with learned edge weights. This graph prior is useful when the body topology is stable, but in WLASL the signing-relevant joints are dominated by fingers, and the finger graph is exactly where signer anatomy varies most (finger length ratios, joint mobility, handedness). Hard-wiring this graph forces the model to learn signer-specific corrections inside the GCN layers — the kind of memorization we want to avoid for cross-signer generalization. Benchmark accuracy alone is not used as justification.

**Second alternative (briefly): I3D / Video-Swin on raw RGB.** Rejected because raw RGB carries the strongest signer-identity signal of all (face, clothing, skin tone, background) and would widen the cross-signer gap, not close it.

## 3. Experimental setup (Step 4)

- **Dataset:** WLASL100 (top-100 most frequent glosses) with MediaPipe Holistic landmarks.
- **Split:** signer-disjoint — a fixed subset of signer IDs is held out from training and is the only test set. Never seen during training.
- **Models compared:** RQE-Transformer and a Pooled-MLP baseline (same input, same training budget, no temporal modeling). The baseline isolates how much of the cross-signer gap RQE alone closes.
- **Metrics:** Top-1 accuracy and macro F1 on the held-out test set (per §6.2).
- **Fine-tune config (declared per §6.1):** AdamW (lr 3e-4, weight decay 1e-4), cosine schedule, 15 epochs, batch 32, max 64 frames, cross-entropy with label smoothing 0.1, grad-clip 1.0, seed 0.

## 4. Failure analysis (Step 5)

The notebook produces (i) per-signer accuracy on the held-out test set and (ii) the top-confused class pairs from the confusion matrix.

**Case 1 — Worst held-out signer.** Typically a left-handed signer or one with atypical signing speed. **Mechanism:** RQE normalizes translation and torso scale but **not** handedness or temporal speed. Left-handed signing flips which hand carries the lexical content, so the same gloss lives in a different region of `input_proj`'s output and the linear projection cannot un-flip it. Speed differences shift salient frames; the Transformer's learned **absolute** positional embedding misaligns attention patterns calibrated on slow signers when tested on fast ones. Both failures point to specific design properties: the invariance set in RQE is incomplete, and the positional encoding is absolute rather than relative.

**Case 2 — Top confused gloss pair.** Pairs sharing the same gross arm trajectory but differing in hand-shape. **Mechanism:** the per-frame token is the *flattened* RQE-normalized landmark vector. Pose-only landmarks at MediaPipe's resolution have low fidelity on the fingers (small Cartesian deltas between finger configurations after torso-scale normalization), so the input itself underrepresents the discriminative signal. This is a representational limit of pose-only input, not a learning failure — ST-GCN would have the same problem.

## 5. Thesis direction (Step 7)

**Did the results support the hypothesis?** *Partially.* RQE+attention closes part of the gap relative to a baseline that ignores temporal structure (direction supported). But per-signer and confusion analyses show residual errors concentrate on (a) signers with handedness/speed outside the training distribution and (b) gloss pairs distinguished by fine hand-shape — neither addressed by RQE.

**Concrete next step — "RQE+".** Two preprocessing additions motivated directly by the two failure cases:
1. **Handedness canonicalization** — mirror all landmarks to a single dominant hand whenever the inferred dominant hand is left.
2. **Signing-speed renormalization** — resample each clip to a fixed number of *signing-active* frames using hand-motion energy.

Plus an architectural addition: a **finger-resolution branch** feeding the 21-point MediaPipe Hand landmarks of the dominant hand as a second token stream into the Transformer, so finger geometry is not averaged out with the rest of the body.

**Feasibility.** All three are preprocessing-only or token-concat changes (no new training data, no new model family), and can be ablated on the same signer-disjoint split. Expected effect tied to the failure cases: case 1 should close substantially; case 2 should improve in proportion to how much the hand-only stream resolves finger-shape ambiguity.

---

## Files

- [rqe_transformer_wlasl.ipynb](rqe_transformer_wlasl.ipynb) — end-to-end Kaggle notebook (data loading, RQE, model, training, evaluation, failure analysis).

## LLM disclosure (§10)

Parts of the code scaffolding and some markdown phrasing in the notebook and this README were drafted with the assistance of a large language model (GitHub Copilot Chat). All design choices, the architecture justification, the failure analysis, and the thesis direction reflect my own reasoning and were reviewed manually.

## Key references

1. Zhou L. et al. (2023). *Human pose-based estimation, tracking and action recognition with deep learning: a survey.* arXiv:2310.13039
2. Yan S., Xiong Y., Lin D. (2018). *Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition.* AAAI.
3. Li D. et al. (2020). *Word-level Deep Sign Language Recognition from Video (WLASL).* WACV.
4. Bohacek M., Hruz M. (2022). *Sign Pose-based Transformer for Word-level Sign Language Recognition.* WACV Workshops.
5. Selvaraj P. et al. (2022). *OpenHands: Making Sign Language Recognition Accessible.* ACL.
6. Lugaresi C. et al. (2019). *MediaPipe: A Framework for Building Perception Pipelines.* arXiv:1906.08172
7. Vaswani A. et al. (2017). *Attention Is All You Need.* NeurIPS.
8. Loshchilov I., Hutter F. (2019). *Decoupled Weight Decay Regularization (AdamW).* ICLR.
9. Liu Z. et al. (2022). *Video Swin Transformer.* CVPR.
10. De Coster M. et al. (2023). *Machine Translation from Signed to Spoken Languages.* UAIS.