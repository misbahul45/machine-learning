# Deep Learning → Transformer Research: Build the Model From Scratch

> **Learning contract:** do not treat a model as understood because it runs. Derive its equations, implement a minimal version, verify its gradients, measure its resource use, explain its failure modes, and reproduce at least one controlled experiment.
>
> **Scope of this README:** the mathematical and engineering foundations required to progress from deep learning to a research-level understanding of decoder-only Transformers. This is the first foundation-model README—not the complete roadmap for GPT-3-scale distributed pretraining, post-training, or agent systems.

## Table of contents

- [0. Mission, constraints, and what “from scratch” means](#0-mission-constraints-and-what-from-scratch-means)
- [1. Prerequisite map](#1-prerequisite-map-connect-the-existing-math-and-ml-roadmap)
- [2. Research workflow](#2-research-workflow-and-definition-of-done)
- [3. Roadmap to Transformer research](#3-roadmap-from-deep-learning-foundations-to-transformer-research)
- [4. Mathematics inside a decoder-only Transformer](#4-mathematics-inside-a-decoder-only-transformer)
- [5. Model architecture and variants](#5-model-architecture-exact-baseline-and-deliberate-variants)
- [6. Training system design](#6-training-system-design)
- [7. Hardware, memory, and compute](#7-hardware-memory-and-compute-budget)
- [8. Core experiments](#8-core-experiments-do-not-skip-them)
- [9. Testing standards](#9-testing-standards-for-a-research-quality-implementation)
- [10. Research-level Transformer literacy](#10-moving-from-implementation-to-research-level-transformer-literacy)
- [11. Deliverables and mastery gates](#11-deliverables-and-mastery-gates)
- [12. Recommended references](#12-recommended-references)
- [13. Boundary to a GPT-3-class goal](#13-important-boundary-from-this-transformer-phase-to-a-gpt-3-class-goal)
- [14. Modern LLM system map](#14-the-modern-llm-is-a-system-not-only-a-neural-network)
- [15. Model-family literacy](#15-model-family-literacy-what-to-compare-in-public-research)
- [16. Advanced architecture and inference](#16-advanced-architecture-moe-long-context-multimodality-and-efficient-inference)
- [17. Post-training](#17-post-training-turning-a-base-model-into-a-useful-assistant)
- [18. Self-improving LLMs and parameters](#18-self-improving-llms-define-the-term-precisely)
- [19. AI psychology and AI anxiety](#19-ai-psychology-ai-anxiety-and-human-factors)
- [20. AI policy and safety engineering](#20-ai-policy-governance-and-safety-engineering)
- [21. Interpretability](#21-interpretability-and-mechanistic-understanding)
- [22. Evaluation science](#22-evaluation-science-measure-capability-without-fooling-yourself)
- [23. Professional AI engineering](#23-professional-ai-engineering-production-architecture-and-reliability)
- [24. Research operations](#24-research-operations-and-experimental-discipline)
- [25. Hardware and distributed scaling](#25-hardware-and-scaling-from-a-small-lab-to-distributed-training)
- [26. Advanced capstones](#26-capstone-sequence-from-a-tiny-model-to-a-governed-assistant)
- [27. Mastery gates](#27-mastery-gates-for-advanced-llm-engineering)
- [28. Advanced reading map](#28-suggested-advanced-reading-map)
- [29. Professional readiness](#29-definition-of-professional-readiness)

---

## 0. Mission, constraints, and what “from scratch” means

The long-term target is to build and train your own language model with an understanding of the mathematics, code, data pipeline, hardware, and system design. The learning path deliberately separates **understanding a foundation model** from **having the compute budget to train a frontier-scale model**.

### Target outcomes for this phase

By the end, you should be able to:

- Derive and implement a neural network's forward pass, loss, and backward pass for a small model.
- Explain automatic differentiation as reverse-mode differentiation over a computational graph, not as magic.
- Explain why normalization, residual paths, initialization, learning-rate schedules, and AdamW influence optimization.
- Implement tokenization for a small corpus, create language-model training examples, and explain the limitations of the tokenizer.
- Derive scaled dot-product attention, multi-head causal self-attention, and the decoder-only Transformer block.
- Build a small GPT-style model with randomly initialized parameters and train it on a permitted corpus **from random initialization**.
- Verify a manual implementation against PyTorch using gradient checks and controlled tests.
- Profile the model and account for parameters, activations, optimizer state, FLOPs, memory traffic, throughput, and inference cache.
- Read Transformer papers by translating claims into hypotheses, equations, ablations, implementation changes, and measurable results.
- Produce a reproducible experiment with a configuration, fixed data split, seed policy, checkpoints, metrics, and an honest limitations section.

### Non-goals for this first README

This phase does not promise that free hardware can train a GPT-3-class foundation model. It also does not cover the full post-training stack (instruction fine-tuning, preference optimization, RLHF), multimodal models, tool-using agents, or production distributed training in full depth. Those topics come **after** the base model and training loop are understood.

A model can be trained from scratch and still be tiny. The meaningful first goal is to own the full pipeline for a small model, establish sound measurements, and learn which parts scale—and which assumptions break—before increasing the budget.

### Training-from-scratch definition

For the core project in this README, “from scratch” means:

1. Start with randomly initialized model parameters.
2. Build or explicitly understand the tokenizer and vocabulary.
3. Prepare the pretraining corpus yourself, subject to its license and terms.
4. Create next-token training targets.
5. Train using a next-token objective from the initial parameters.
6. Save checkpoints and training metadata.
7. Evaluate on held-out text that was not used to train or select the model.

Using PyTorch tensor kernels, an optimizer implementation, a data loader, or a public research paper is not the same as fine-tuning a pretrained model. You are still training your own model from random initialization. However, using a pretrained tokenizer means the tokenizer is borrowed rather than built from scratch; document that distinction clearly.

---

## 1. Prerequisite map: connect the existing math and ML roadmap

The existing roadmap already covers the key mathematical foundations—linear algebra, calculus, probability, statistics, information theory, optimization—and classical ML. Do not restart every subject as an isolated course. Revisit a concept when it appears in an algorithm, derive it for the concrete model, then implement and test it.

```text
Linear Algebra ───────┐
Differential Calculus ─┤
Probability & Statistics ─┤
Information Theory ────┤
Numerical Analysis ────┤
Optimization ──────────┘
          │
          ▼
Tensor programming and computational graphs
          │
          ▼
Perceptron → MLP → backpropagation → optimization diagnostics
          │
          ▼
Representation learning → sequence modeling
          │
          ▼
Attention → causal self-attention → Transformer blocks
          │
          ▼
Tokenizer + corpus + objective + training loop
          │
          ▼
A small decoder-only language model trained from random initialization
          │
          ▼
Research reproduction, profiling, scaling experiments, system optimization
```

### Required foundations: competency, not just vocabulary

| Foundation | You must understand and derive | It becomes useful in |
|---|---|---|
| Linear algebra | Matrix multiplication; transpose; dot products; vector norms; tensor shapes; rank; projections; eigenvalues/SVD at a conceptual and practical level | Dense layers, embeddings, attention, gradient transformations, low-rank structure |
| Multivariable calculus | Partial derivatives, gradient, Jacobian, chain rule, directional derivatives, Taylor expansion | Backpropagation, optimization, gradient checking |
| Probability | Random variables, conditional probability, expectation, variance, categorical distributions, likelihood | Next-token prediction, sampling, interpreting output probabilities |
| Statistics | Train/validation/test split, estimation, uncertainty, data leakage, overfitting, sampling bias | Evaluation, ablations, claims about model improvements |
| Information theory | Entropy, cross-entropy, KL divergence, negative log-likelihood, perplexity | Language-model loss and evaluation |
| Optimization | Empirical risk, SGD, momentum, Adam/AdamW, regularization, learning-rate schedules, gradient clipping | Training stability and convergence |
| Numerical analysis | Floating-point formats, rounding, overflow/underflow, stable softmax/log-sum-exp, conditioning | Mixed precision and debugging silent numerical errors |
| Algorithms and systems | Complexity, memory layout, profiling, caching, files/streams, reproducibility | Data loading, attention performance, checkpointing |

### Entry diagnostic (do this before the main implementation)

- [ ] Multiply two matrices without calling a high-level matrix-multiplication function.
- [ ] Derive the gradient of mean squared error for linear regression.
- [ ] Derive logistic-regression binary cross-entropy at a high level.
- [ ] Explain the difference between a probability, a likelihood, and a loss.
- [ ] Implement a numerically stable softmax using the log-sum-exp principle.
- [ ] Explain why a train score is not an unbiased estimate of generalization.
- [ ] Explain why correlated input features can cause numerical problems.

If a box is unchecked, do not abandon the Transformer path. Add a focused prerequisite exercise and connect it to the current implementation.

---

## 2. Research workflow and definition of “done”

Use the same investigation template for every component and paper.

1. **Problem:** What limitation or failure is the method addressing?
2. **Claim:** What exactly is asserted to improve, under which conditions?
3. **Assumptions:** Which assumptions are necessary? Which are empirical conveniences?
4. **Mathematics:** Can the central equation be derived and interpreted symbol by symbol?
5. **Shapes:** What is the shape, dtype, device, and memory layout of every important tensor?
6. **Implementation:** Can the mechanism be expressed in a small reference implementation?
7. **Verification:** Does it pass unit tests, finite-difference gradient checks, and a framework comparison?
8. **Experiment:** Is there a baseline, a controlled change, a metric, and a repeatable protocol?
9. **Failure analysis:** What changes when dimensions, data, precision, or hyperparameters are altered?
10. **Cost:** What are the time, memory, compute, and data costs?
11. **Evidence:** Which results are reproduced, which are inferred, and which remain unknown?
12. **Communication:** Can you explain the mechanism without hiding behind library names?

A checklist is not complete because you watched a lecture, copied code, or obtained a decreasing loss. Mark a component “understood” only after you can **derive → implement → verify → measure → explain** it.

### Research notebook template

Use one file per investigation, for example `research/attention/causal-mask-ablation.md`.

```text
Question:
Paper / source:
Hypothesis:
Baseline:
Single variable changed:
Dataset and split:
Model/configuration:
Random seeds:
Metric and measurement protocol:
Expected result:
Observed result:
Negative or unexpected results:
Runtime / memory / hardware:
Interpretation:
Threats to validity:
Next experiment:
```

Do not present an uncontrolled experiment as proof of causation. When comparing model variants, hold the data, token budget, evaluation split, and other meaningful conditions constant wherever possible.

---

## 3. Roadmap: from deep learning foundations to Transformer research

This is a mastery-based sequence, not a promise that each phase takes a fixed number of days. The total duration depends on existing Python/PyTorch fluency, available computing hardware, and how deeply you complete the experiments.

### Phase 0 — Engineering and tensor fluency

**Goal:** remove implementation friction before neural-network complexity increases.

Study and practice:

- Python functions, classes, iterators, generators, context managers, exceptions, packaging, virtual environments.
- NumPy indexing, broadcasting, matrix operations, random generators, dtype behavior, and memory copies versus views.
- PyTorch tensors, devices, dtypes, modules, parameter registration, optimizers, `train()`/`eval()`, `no_grad()`, checkpoint serialization.
- Basic command-line tooling, version control, logging, tests, and reproducible configuration.
- Tensor shape notation and an explicit shape ledger for each forward pass.

Build:

- [ ] A deterministic data-batch iterator.
- [ ] A tiny training loop with training/validation metric logging.
- [ ] A checkpoint containing model state, optimizer state, step, configuration, and random-generator states when practical.
- [ ] Tests for tensor shapes, expected dtypes, device placement, and deterministic behavior under a fixed seed.

**Exit gate:** given a small tensor expression, you can predict its output shape and identify likely device/dtype errors without trial-and-error coding.

### Phase 1 — Neural networks from first principles

**Goal:** understand the computational graph and the origin of gradients.

Topics:

- Linear model and perceptron.
- Affine transformations: `z = xW + b` (state the convention for row/column vectors).
- Activation functions: sigmoid, tanh, ReLU, GELU; saturation and dead activations.
- MLPs and the universal-approximation idea, with its limitations.
- Loss functions: MSE, binary cross-entropy, categorical cross-entropy.
- Computational graphs and reverse-mode automatic differentiation.
- Backpropagation through a chain of layers.
- Initialization, gradient scale, saturation, vanishing/exploding gradients.
- Finite-difference gradient checking.

Build in order:

1. Scalar-value autodiff engine for expressions such as `a*b + c`.
2. Manual NumPy linear regression and a one-hidden-layer MLP.
3. Manual backward pass for an affine layer and an activation function.
4. The same small network in PyTorch.
5. A comparison harness that checks analytical gradients against central finite differences and PyTorch gradients.

Core equations to derive:

For a layer `H = XW + b`, where `X ∈ R^(B×Din)`, `W ∈ R^(Din×Dout)`, `b ∈ R^Dout` and upstream gradient `G = ∂L/∂H`:

```text
dL/dW = Xᵀ G
dL/db = sum over batch dimension of G
dL/dX = G Wᵀ
```

Verify these shapes against your own vectorization convention. Many backpropagation bugs are shape or transpose errors rather than deep mathematical errors.

**Exit gate:** you can explain why the backward pass uses the chain rule, implement one affine layer's gradients, and detect a deliberately injected gradient bug.

### Phase 2 — Optimization and training dynamics

**Goal:** understand why an architecture that is mathematically valid may still fail to train.

Topics:

- Empirical risk minimization and mini-batch stochastic gradients.
- SGD, momentum, Adam, and AdamW; what moving averages estimate.
- Learning rate versus batch size; warmup and decay schedules.
- Weight decay versus loss penalties; do not assume these are identical for every optimizer.
- Gradient clipping and the difference between clipping gradients by value and by norm.
- Initialization, activation statistics, residual connections.
- Batch normalization versus layer normalization; why sequence models typically need normalization that does not depend on other examples in the batch.
- Train/validation curves, underfitting, overfitting, loss spikes, and seed variance.
- Gradient norms, parameter update norms, activation distributions, and optimizer-state diagnostics.

Experiments:

- [ ] Overfit a tiny dataset intentionally; if the model cannot, debug the implementation before scaling data.
- [ ] Compare learning rates logarithmically across a small range.
- [ ] Plot gradient norms and loss versus step.
- [ ] Compare plain SGD and AdamW under a documented, fair setup.
- [ ] Remove normalization or residual connections from a deeper MLP and observe the optimization effect.
- [ ] Run at least three seeds for a result you intend to call a research finding.

**Exit gate:** you can diagnose whether a failure likely comes from data, the objective, gradients, initialization, optimizer settings, or numerical instability.

### Phase 3 — Representation learning and sequence modeling

**Goal:** understand what a language model is predicting before introducing attention.

Topics:

- One-hot representations, learned embeddings, embedding geometry, and tied embeddings.
- Discrete sequences, vocabulary, token IDs, special tokens, and padding.
- N-gram language models and conditional probabilities.
- Autoregressive factorization of a joint sequence distribution.
- Recurrent neural networks, state transitions, and the difficulty of long-range credit assignment.
- Truncated backpropagation through time; vanishing/exploding gradients.
- Teacher forcing for next-token learning and the distinction between training and generation.
- Sequence shifts: inputs contain earlier tokens; labels contain the next token.

Core language-model factorization:

```text
p(x₁, …, xT) = ∏ₜ p(xₜ | x₁, …, xₜ₋₁)
```

This factorization defines an autoregressive objective; it does not imply that the model reasons correctly or that its training data are trustworthy.

Build:

- [ ] A bigram or trigram baseline.
- [ ] A character-level RNN language model.
- [ ] A next-token data loader that makes input/target alignment explicit.
- [ ] An autoregressive sampler that appends one token at a time.
- [ ] A comparison of train and held-out negative log-likelihood.

**Exit gate:** you can explain the training objective, construct correct target shifts, and explain the difference between parallel next-token training and sequential autoregressive generation.

### Phase 4 — Attention before the Transformer

**Goal:** derive attention as a content-dependent weighted combination, not as a library call.

Topics:

- Query, key, and value as learned projections.
- Dot-product similarity and its limitations.
- Scaled dot-product attention.
- Row-wise softmax, numerical stability, and probability simplex interpretation.
- Why scaling by `sqrt(d_k)` helps control score variance under a simple initialization assumption.
- Causal masks and why future tokens must not influence an earlier token during autoregressive training.
- Multi-head attention and the different subspaces represented by each head.
- Shape transformations and contiguous/non-contiguous tensor behavior.

Scaled dot-product attention:

```text
Q = X W_Q          # [B, T, d_k]
K = X W_K          # [B, T, d_k]
V = X W_V          # [B, T, d_v]
S = Q Kᵀ / sqrt(d_k)
A = softmax(S + causal_mask, dim=-1)
O = A V
```

For a multi-head implementation, include a head axis such as `[B, n_heads, T, head_dim]`. The mask must broadcast over batch/head dimensions correctly. Use `-∞` or a numerically appropriate mask value before softmax; test rows that have only legal past positions and confirm that a forbidden future position receives zero probability within numerical tolerance.

Required experiments:

- [ ] Prove causality with a unit test: changing a future token must not change outputs at prior positions in evaluation mode.
- [ ] Compare scaled versus unscaled dot-product scores.
- [ ] Inspect attention matrices on a tiny hand-constructed sequence.
- [ ] Compare single-head and multi-head attention with controlled parameter counts where feasible.
- [ ] Validate attention output and gradients against a simple reference implementation.

**Exit gate:** you can derive the attention equations, explain every tensor shape, and test that no future-token information leaks into earlier predictions.

### Phase 5 — Transformer block internals

**Goal:** understand why the architecture is arranged as it is and where variants differ.

Topics:

- Residual paths and residual stream interpretation.
- Pre-LayerNorm versus Post-LayerNorm arrangements.
- LayerNorm: normalization axes, learned gain/bias, and numerical epsilon.
- RMSNorm as a related alternative; do not treat it as identical to LayerNorm.
- Position information: learned positional embeddings, sinusoidal encoding, and rotary positional embeddings (RoPE).
- Feed-forward/MLP sublayer, expansion ratio, GELU and SwiGLU.
- Dropout: where it is used and how `train()` and `eval()` differ.
- Decoder-only causal blocks versus encoder and encoder-decoder designs.
- Final normalization, vocabulary projection, logit scaling, and weight tying.
- Parameter initialization and scaling behavior in deep residual networks.

A common pre-normalization decoder block is:

```text
u = x + CausalSelfAttention(LayerNorm(x))
y = u + FeedForward(LayerNorm(u))
```

The exact architecture is a design choice: a paper may change normalization, activation, positional representation, attention layout, bias terms, or initialization. Document the exact variant you implement; “Transformer” alone is not a sufficiently precise model specification.

Build the block in independently tested parts:

1. `Embedding` for tokens.
2. Positional strategy (start simple, then compare variants).
3. Stable `LayerNorm`.
4. Causal self-attention.
5. Feed-forward sublayer.
6. Residual connection and dropout placement.
7. Stack of blocks.
8. Final norm and language-model head.

**Exit gate:** you can draw the complete block, explain the role and computational cost of each part, and reproduce its forward pass from your own shape ledger.

### Phase 6 — Tokenization, data, objective, and evaluation

**Goal:** understand that a language model includes more than neural-network layers.

#### Tokenization

Study:

- Character, byte-level, word-level, and subword tokenization.
- Byte Pair Encoding (BPE) and merge operations.
- Vocabulary-size trade-offs: sequence length, embedding/head parameters, rare words, and multilingual coverage.
- Special tokens, end-of-document boundaries, whitespace, Unicode normalization, invalid byte sequences, and reversible decoding.
- Train tokenizer on the training corpus only when doing so is part of the experiment; record tokenizer version and vocabulary hashes.
- Do not silently compare models with different tokenizers as if token counts or perplexity were directly interchangeable.

Build:

- [ ] Character tokenizer and decoder.
- [ ] A small BPE tokenizer, or a verified byte-level tokenizer implementation that you fully understand.
- [ ] Round-trip tests and tests for Unicode, whitespace, special tokens, and unseen text.
- [ ] A script that reports tokens per document, vocabulary coverage, and sample tokenizations.

#### Corpus preparation

Study:

- Licensing, permitted use, provenance, source mixture, language balance, and documentation.
- Deduplication at exact and near-duplicate levels.
- Removing corrupted/boilerplate records and documenting filters.
- PII and sensitive data risks; do not collect private data without a valid basis and safeguards.
- Document-level train/validation/test splits to reduce leakage from overlapping chunks.
- Evaluation contamination, repeated boilerplate, and source-specific artifacts.
- Data quality and mixture as experimental variables, not a one-time preprocessing chore.

Start with a small, permitted educational corpus or your own appropriately licensed text. Public availability does not automatically mean unrestricted training permission. Record source URLs, licenses/terms, dates accessed, filters, and hashes. Do not begin by crawling the entire web.

#### Objective and evaluation

For logits `zₜ` over vocabulary `V` and gold next token `xₜ`, categorical negative log-likelihood is:

```text
pₜ = softmax(zₜ)
Lₜ = -log pₜ[xₜ]
L = mean over valid target positions of Lₜ
```

Equivalently, it is categorical cross-entropy when targets are represented as one-hot labels. Compute it with a numerically stable fused framework loss in the optimized implementation, but know what that loss computes and which positions are included.

Perplexity:

```text
PPL = exp(mean negative log-likelihood)
```

Perplexity depends on tokenization and evaluation protocol. Use it to compare compatible models on the same held-out tokenization and corpus; do not claim it measures every aspect of language quality.

Required safeguards:

- [ ] Keep a held-out validation set that does not drive gradient updates.
- [ ] Keep a separate test set for final reporting, or clearly state when data scarcity forces a different protocol.
- [ ] Report tokenization, data split, model size, context length, training tokens, compute, and training duration alongside loss/perplexity.
- [ ] Inspect actual generations; a single scalar metric is not sufficient.
- [ ] Label synthetic/demo results as such and do not imply broad language competence from a tiny corpus.

**Exit gate:** you can trace one example from raw text through token IDs, input/target shifting, logits, loss, decoding, and evaluation without ambiguity.

### Phase 7 — Build a small decoder-only GPT from random initialization

**Goal:** integrate the complete base-model training pipeline.

Minimum end-to-end features:

- Configuration object for vocabulary size, context length, layers, attention heads, embedding width, dropout, and model variant.
- Tokenizer and data-loader versions recorded in the run metadata.
- Training and validation loops with correct gradient clearing, backpropagation, optimizer step, and learning-rate schedule.
- Gradient clipping when justified by the selected configuration.
- Periodic validation and checkpoint saving.
- Resumable training: model weights alone are insufficient for faithful continuation; save optimizer and scheduler state and step metadata.
- Generation with a clear stop condition and configurable sampling.
- Training logs, validation logs, throughput, memory, and failure records.
- Tests for shapes, causality, loss reduction on a deliberately tiny dataset, checkpoint round-trip, and resumed-versus-uninterrupted training where practical.

Suggested first target:

- A small decoder-only Transformer with roughly hundreds of thousands to a few million parameters, then increase only after the baseline is correct.
- A short context window and a small corpus.
- A single-device run; use CPU when no appropriate GPU is available.
- A character-level tokenizer for the first architectural correctness pass, followed by a subword tokenizer experiment.

The parameter range is an educational starting point, not a guarantee of a particular speed or output quality. Tune dimensions to fit actual RAM/VRAM and measure the result.

**Exit gate:** a clean run from random initialization produces checkpoints, decreasing training loss, held-out validation metrics, generated samples, and enough metadata for another person to reproduce the experiment.

### Phase 8 — Numerical correctness, profiling, and hardware efficiency

**Goal:** know whether a speedup is real and where the time/memory goes.

Study:

- CPU/GPU execution model; host/device transfers.
- Memory hierarchy, registers, caches, GPU shared memory/SRAM, and high-bandwidth memory.
- Compute-bound versus memory-bandwidth-bound operations and arithmetic intensity.
- FLOP estimates, parameter count, token throughput, step latency, and memory peaks.
- Mixed precision (FP32, FP16, BF16 where supported), dynamic loss scaling for FP16, and precision-specific failure modes.
- Activation memory; gradient accumulation; activation checkpointing and its recomputation cost.
- Fused operations, kernel launch overhead, tensor layouts, and padding waste.
- Standard attention versus memory-efficient exact attention (for example FlashAttention).
- Data-loader bottlenecks, tokenizer throughput, checkpoint I/O, and compile warmup.

Never optimize before you have a correct reference. A fast kernel that changes outputs or gradients unexpectedly is not a successful optimization.

Profiling protocol:

1. Record hardware, driver, framework, compiler, dtype, batch size, sequence length, model config, and random seed.
2. Run warmup iterations and separate compilation/startup from steady state.
3. Synchronize device execution where needed before timing GPU work.
4. Measure a meaningful number of steady-state steps.
5. Report median/percentiles and variability instead of a single lucky result.
6. Record peak allocated memory and reserved memory distinctly when the framework exposes them.
7. Compare outputs and gradients to the reference within dtype-appropriate tolerances.
8. Repeat the baseline after every meaningful environment or code change.

**Exit gate:** you can answer: “What is the bottleneck, what evidence supports that, how much faster is the change, what accuracy/numerical trade-off exists, and is it reproducible?”

### Phase 9 — Transformer variants and research reproduction

**Goal:** move from implementing the canonical architecture to evaluating research claims.

Progress in this order:

1. Reproduce a minimal causal decoder and verify mathematical correctness.
2. Run a controlled comparison of learned position embeddings versus a second positional method.
3. Compare pre-normalization choices or a normalization variant with fixed budgets.
4. Evaluate activation/FFN alternatives such as GELU and SwiGLU.
5. Compare ordinary attention with a memory-efficient exact attention implementation, checking output/gradient tolerance and wall-clock performance.
6. Investigate grouped-query attention (GQA) or multi-query attention (MQA) to understand the quality, memory, and inference trade-offs.
7. Fit a small empirical scaling curve over several model sizes or token budgets; report uncertainty and avoid overclaiming from a tiny range.
8. Reproduce one paper result as closely as available compute and data allow, then list all deviations.

Candidate research themes:

- Positional representations: learned embeddings, sinusoidal, RoPE, and context-length extrapolation.
- Normalization and residual scaling.
- Attention compute and memory: exact attention kernels and IO-aware algorithms.
- Multi-head, multi-query, and grouped-query attention.
- MLP/FFN design, gated activations, and parameter allocation.
- Initialization, optimizer choices, and learning-rate schedules.
- Tokenizer/data mixtures, deduplication, and data quality.
- Model/data/compute scaling laws and their uncertainty.
- Training stability and numerical precision.
- Inference caching, throughput, and latency.

**Exit gate:** you can turn a research paper into an experiment plan and provide evidence that supports or fails to support the claim under your particular setup.

---

## 4. Mathematics inside a decoder-only Transformer

This section is the minimum derivation set. Work each derivation on paper, then validate it in code. Use a consistent matrix convention: below, batches are rows and hidden features are columns.

### 4.1 Tensor shape conventions

| Symbol | Meaning | Typical shape |
|---|---|---|
| `B` | Batch size | scalar dimension |
| `T` | Sequence length / context length | scalar dimension |
| `V` | Vocabulary size | scalar dimension |
| `C` | Model/embedding width | scalar dimension |
| `L` | Number of Transformer blocks | scalar hyperparameter |
| `H` | Number of attention heads | scalar hyperparameter |
| `D_h` | Width per attention head, commonly `C/H` | scalar dimension |
| `X` | Token IDs | `[B, T]` integer tensor |
| `E` | Token embeddings | `[B, T, C]` floating tensor |
| `Q, K` | Query and key tensors | `[B, H, T, D_h]` |
| `V_h` | Per-head values (avoid confusing with vocab size `V`) | `[B, H, T, D_h]` in common designs |
| `A` | Attention probabilities | `[B, H, T, T]` for dense attention |
| `Z` | Vocabulary logits | `[B, T, V]` |

Before coding any module, write input and output shapes, dtype, device, and axes reduced by every operation.

### 4.2 Embedding lookup

A token embedding table is a trainable matrix `W_E ∈ R^(V×C)`. For token ID `i`, the representation is row `W_E[i]`. It is a learned lookup, not a one-hot vector that must be materialized in memory.

Questions to derive and test:

- How many parameters exist in `W_E`? (`V × C`.)
- What shape is produced by looking up a batch of token IDs `[B,T]`? (`[B,T,C]`.)
- How do gradients accumulate when a token ID occurs multiple times?
- What does weight tying change when the same embedding table is used for the output projection?

### 4.3 Linear projection

For `X ∈ R^(B×T×C)` and `W ∈ R^(C×D)`, a batched projection applies the same matrix at each sequence position:

```text
Y[b,t,:] = X[b,t,:] W + b
Y ∈ R^(B×T×D)
```

This is the central operation in attention projections and the MLP. Understand matrix multiplication first, then understand how a tensor library maps the operation to optimized kernels.

### 4.4 Softmax and categorical cross-entropy

For logits `z ∈ R^V`:

```text
softmax(z)_i = exp(z_i) / Σⱼ exp(z_j)
```

Numerically stable implementation subtracts `m = max(z)` before exponentiation:

```text
softmax(z)_i = exp(z_i - m) / Σⱼ exp(z_j - m)
```

For a target class `y`, the loss is `L = -log softmax(z)_y`. For a single example, the gradient with respect to logits is:

```text
∂L/∂z_i = p_i - 1[i = y]
```

For a batch and sequence, apply the gradient at valid target positions according to the reduction convention (mean, sum, or masked mean) used by the loss implementation. Derive the Jacobian of softmax and show how cross-entropy simplifies the gradient. Test stable loss for extreme logits.

### 4.5 Scaled dot-product attention

For one head and sequence representations `X`:

```text
Q = X W_Q
K = X W_K
V = X W_V
S = Q Kᵀ / √D_h
P = softmax(S + M)
O = P V
```

`M` is the causal mask: locations in which the query position would attend to a future key are blocked. For tokens indexed from zero, a legal attention matrix is lower triangular including its diagonal. Verify whether a library uses a Boolean mask, an additive mask, or a different convention; mask conventions vary across APIs.

Why divide by `√D_h`? Under a simplified model where query/key components are independent with zero mean and unit variance, the dot-product variance grows roughly with `D_h`. Scaling controls logit magnitude as head width increases, helping avoid an overly saturated softmax at initialization. This is an intuition under assumptions, not a universal guarantee about trained activations.

Multi-head attention splits or projects representations into multiple heads, applies attention separately, concatenates the head outputs, then applies an output projection. The exact shape order may change across implementations; the math should not.

### 4.6 Causal masking and information flow

Autoregressive language modeling requires that position `t` only use tokens at positions `≤ t` in its context. The mask enforces this constraint during training, when the full target sequence is present in parallel.

Required unit tests:

- Attention probability at a forbidden future position is zero within tolerance.
- Modifying input positions after `t` does not change logits at positions `≤ t` when using the same model state and evaluation behavior.
- The diagonal convention is correct and each non-padding query has at least one legal key.
- The target shift is correct: output position `t` is trained to predict token `t+1` under the chosen indexing convention.

### 4.7 Feed-forward network

A common two-layer position-wise MLP is:

```text
FFN(x) = W₂ φ(W₁ x + b₁) + b₂
```

`φ` may be GELU or another activation. The hidden expansion width often exceeds `C`, creating substantial parameter and compute cost. In a gated variant such as SwiGLU, two projections create a gate and content pathway; define the exact formula from the chosen reference rather than assuming all gated MLPs are equivalent.

### 4.8 Residual path and normalization

A common pre-norm block has the form:

```text
x₁ = x + Attention(Norm₁(x))
x₂ = x₁ + FFN(Norm₂(x₁))
```

The residual connection provides a direct additive path through depth. Normalization controls activation scale but also changes the optimization geometry. Pre-LN, Post-LN, LayerNorm, and RMSNorm are distinct design variants. Record the selected variant and its hyperparameters.

### 4.9 Autoregressive training objective

Given tokens `x₁…xT`, the model predicts each token from its available prefix. Under next-token shifting, a training example might use:

```text
input  = [x₁, x₂, …, xT-1]
target = [x₂, x₃, …, xT]
```

Then minimize the mean negative log-likelihood over target positions:

```text
L(θ) = - (1 / N_targets) Σₙ log pθ(targetₙ | allowed contextₙ)
```

Use the exact indexing convention in your code, document it, and test it on a tiny sequence whose expected targets are obvious.

### 4.10 Backpropagation through the block

The loss gradient flows from vocabulary logits into the output projection, final normalization, every block, attention and FFN parameters, residual streams, positional representation, and embeddings. PyTorch builds a graph of operations and computes vector-Jacobian products during the backward pass.

You need not hand-code every optimized kernel to understand training. You must, however, be able to:

- Derive gradients for affine transformations, common activations, softmax-cross-entropy, and normalization at a conceptual/math level.
- Explain how the chain rule composes local derivatives.
- Use finite differences to test parameter gradients on a tiny model.
- Explain why the gradient is accumulated in parameter storage before the optimizer update.
- Distinguish gradients from parameter updates and optimizer states.

---

## 5. Model architecture: exact baseline and deliberate variants

### Baseline architecture to implement first

Start with a **decoder-only causal Transformer** because it directly supports next-token language modeling and provides a compact path from attention math to a generative model.

```text
raw text
  ↓
tokenizer → token IDs [B, T]
  ↓
token embedding (+ chosen position mechanism)
  ↓
repeat L times:
    pre-normalization
    causal multi-head self-attention
    residual addition
    pre-normalization
    position-wise FFN
    residual addition
  ↓
final normalization
  ↓
language-model projection → logits [B, T, vocab_size]
  ↓
next-token cross-entropy loss (training) / sampling (generation)
```

This is a conceptual baseline, not one unique canonical configuration. Different papers use different norms, activations, biases, position methods, attention variants, and initialization rules.

### Architecture decisions to make explicit

| Decision | First implementation | Later comparison |
|---|---|---|
| Position information | Learned position embeddings or a simple well-tested baseline | Sinusoidal, RoPE, extrapolation behavior |
| Normalization | LayerNorm in a pre-norm block | RMSNorm, placement and epsilon |
| FFN activation | GELU | SwiGLU or another gated activation |
| Attention | Dense causal multi-head attention | GQA/MQA, FlashAttention, context optimizations |
| Biases | Choose one documented configuration | Compare with a bias-free variant if justified |
| Embedding/output weights | Start untied for clarity or tied if the reference requires it | Parameter count and quality impact |
| Precision | FP32 for correctness on a small model | BF16/FP16 after baseline verification |
| Optimizer | AdamW with a documented schedule | Controlled optimizer/schedule ablations |
| Context length | Small enough for local tests | Longer contexts only with resource accounting |

Do not change many of these simultaneously. You will not know which change caused the effect.

### Shape and parameter accounting

Record a shape ledger for every layer. For a standard dense attention block with model width `C` and `H` heads, each head has width `D_h = C/H` in the common equal-split arrangement. Combined Q/K/V projection widths are commonly each `C`; the concatenated head output returns to width `C`. The FFN hidden width is `D_ff`.

A rough parameter accounting for a typical block (biases and norm parameters omitted for clarity) is:

- Q, K, V and output projections: approximately `4 C²` parameters.
- Two FFN matrices: approximately `2 C D_ff` parameters.
- If `D_ff ≈ 4C`, FFN matrices are approximately `8 C²` parameters, so the FFN is often a larger parameter share than attention projections.

A decoder with `L` blocks therefore has a block-parameter order of `L × (4C² + 2 C D_ff)` plus embeddings, output head (unless tied), biases, and normalization. Treat this as an estimate for the stated architecture, not an exact count for every Transformer variant.

### Concrete baseline configurations for constrained hardware

Treat these as **starting experiments**, not hardware guarantees. Begin with the smallest configuration that passes the tests, then record peak RAM/VRAM and step time before moving up. Vocabulary size and whether embeddings/output weights are tied materially change the parameter count.

| Preset | Layers `L` | Width `C` | Heads `H` | Context `T` | FFN width | Data/tokenizer | Use |
|---|---:|---:|---:|---:|---:|---|---|
| CPU correctness | 2 | 64 | 2 | 128 | 256 | Character-level, small corpus | Unit/integration tests, first end-to-end training run |
| Small accelerator | 4 | 128 | 4 | 256 | 512 | Small subword vocabulary, permitted corpus | Learn training curves and profiling |
| Research sandbox | 6 | 256 | 8 | 512 | 1,024 | Versioned subword corpus | Small ablations only if memory and runtime measurements support it |

For the CPU preset, start with batch size `4` or `8`, then reduce it if memory or latency is problematic. On a GPU, batch size should be selected by measurements, not by copying a config from another machine. For the very first model, a character vocabulary often keeps the output projection small; later, experiment with a subword tokenizer to understand the costs of larger vocabularies.

A practical token-budget ladder is:

- **Debug budget:** a few hundred to a few thousand sequences/tokens, just enough to test data alignment and loss.
- **Pipeline budget:** a small corpus and a fixed, documented token budget sufficient to test checkpointing, validation, and generation end to end.
- **Research budget:** several controlled runs across configurations or variants, only after measuring expected runtime and memory.

Do not confuse “the training loop runs” with “the model is useful.” Tiny token budgets validate code and teach mechanics; they do not establish broad language competence.

### Cost scaling to derive before each run

For dense causal attention, the simple score matrix has dimensions `[B, H, T, T]`. Its memory is quadratic in `T` in a straightforward implementation. A rough per-layer compute view is:

- Projections and feed-forward work scale approximately linearly in sequence length `T` and quadratically with width `C` (for a fixed FFN expansion ratio).
- Dense attention score/value operations scale approximately with `B × T² × C`.
- Increasing context length can therefore change the bottleneck even when parameter count remains constant.

Before a run, estimate model parameters, token budget, parameter/optimizer memory, batch/context activation needs, and expected wall time. After the run, compare estimates with measurements and explain large discrepancies.

### Position representations: research questions

- What information is available to the model if there is no explicit positional mechanism?
- Why does causal masking alone not guarantee that token positions are represented correctly?
- How do learned embeddings, sinusoidal encodings, and RoPE differ mathematically?
- What does “context-length extrapolation” measure, and how is it distinct from merely running at a longer sequence length?
- Which training and evaluation settings make a comparison fair?

### Normalization and residual paths: research questions

- Which activations are normalized and along which dimensions?
- How do pre-norm and post-norm alter the gradient path?
- How are gain, bias, epsilon, and variance conventions implemented?
- Can a training instability be connected to activation or gradient statistics?
- Does the proposed change improve quality, stability, speed, memory, or some combination—and how are those measured separately?

### Attention alternatives: research questions

- Dense self-attention forms a `[T,T]` score/probability structure per head in a straightforward implementation. How does this affect memory and compute as `T` grows?
- What does FlashAttention change about IO and tiling, and what does it **not** change about the exact attention function?
- Why can GQA/MQA reduce key/value cache size during decoding?
- Which comparisons need equal model sizes, equal token budgets, or equal wall-clock budgets?

---

## 6. Training system design

The training program should be designed as a small reproducible system, not a single notebook cell.

### Suggested project layout

```text
transformer-from-scratch/
├── README.md
├── pyproject.toml
├── configs/
│   ├── tiny_cpu.yaml
│   └── tiny_gpu.yaml
├── data/
│   ├── raw/                 # do not commit large/private corpora
│   ├── processed/           # ignored or versioned by manifest
│   └── manifests/
├── research/
│   ├── notes/
│   ├── derivations/
│   ├── experiments/
│   └── paper-reproductions/
├── src/
│   ├── tokenizer/
│   ├── data/
│   ├── nn/
│   │   ├── embeddings.py
│   │   ├── normalization.py
│   │   ├── attention.py
│   │   ├── feed_forward.py
│   │   ├── transformer_block.py
│   │   └── language_model.py
│   ├── training/
│   │   ├── losses.py
│   │   ├── optimizer.py
│   │   ├── scheduler.py
│   │   ├── trainer.py
│   │   └── checkpoint.py
│   ├── evaluation/
│   ├── generation/
│   └── utils/
├── tests/
│   ├── test_tokenizer.py
│   ├── test_shapes.py
│   ├── test_attention_causality.py
│   ├── test_gradients.py
│   ├── test_loss.py
│   └── test_checkpoint_resume.py
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── generate.py
│   └── profile.py
└── runs/                    # ignored; checkpoints and logs
```

This is a proposed organization, not a requirement to create every file on day one. Start with the smallest coherent version and split modules as tests and experiments become clearer.

### Training-step contract

A typical step should make each operation visible:

1. Fetch token batches and move them to the target device.
2. Construct inputs and next-token targets.
3. Clear gradients (`zero_grad`, with `set_to_none=True` if suitable).
4. Run forward pass and compute loss.
5. Check loss for non-finite values.
6. Backpropagate.
7. Optionally clip gradients using the documented rule.
8. Update parameters with optimizer.
9. Advance the scheduler at its intended cadence.
10. Record loss, learning rate, gradient norm, time, and relevant memory metrics.
11. Periodically evaluate and checkpoint.

The exact scheduler cadence (per batch, optimizer step, or epoch) must match the schedule definition. If using gradient accumulation, distinguish micro-batches from optimizer steps and scale the loss correctly.

### Checkpoint contract

For faithful resume, save as many of the following as your experiment requires:

- model `state_dict`;
- optimizer state;
- scheduler state;
- global step and epoch/data position;
- model and training configuration;
- tokenizer vocabulary/merge files and version/hash;
- data manifest and preprocessing version;
- random-number-generator state for Python, NumPy, CPU tensors, and accelerator where practical;
- software versions, hardware metadata, git commit, and metrics history.

A checkpoint is not a substitute for a reproducible corpus. Large datasets may be represented by immutable manifests and hashes rather than copied into every run.

### Software stack

Start with a boring, reproducible stack:

- Python and a virtual environment/project lockfile.
- NumPy for reference math and small from-scratch exercises.
- PyTorch for tensor kernels, autograd, modules, and optimizers.
- A test runner such as `pytest`.
- A configuration format that can be serialized alongside runs.
- Plain JSONL/CSV logging at first; add a tracking tool only if it solves a clear problem.
- Optional `einops` for explicit tensor rearrangements, after you understand the underlying shape operations.
- Optional `torch.profiler` and accelerator tooling when hardware is available.
- Optional Triton only after profiling shows a kernel bottleneck and the reference implementation is tested.

Using frameworks is compatible with learning from scratch. The critical distinction is whether you understand and own the model specification, initialization, corpus, objective, and training process—not whether every matrix multiplication is handwritten in Python.

---

## 7. Hardware, memory, and compute budget

### First principle: measure the actual machine

Do not assume a GPU is available, that VRAM equals system RAM, or that a free hosted runtime will remain available. Record your own CPU, RAM, accelerator model, VRAM, driver, framework version, and storage. Free-cloud availability and usage restrictions change; treat hosted accelerators as optional opportunities rather than a guaranteed foundation.

For a CPU-only machine with limited RAM, begin with a character-level or small-subword model, short context, small batches, and a compact corpus. Keep the implementation correct before increasing model width, layer count, or context length. CPU training can teach the full pipeline even when it is not fast enough for large experiments.

### Parameter memory is only one part of training memory

If the model has `N` parameters and each parameter occupies `b` bytes, parameter storage alone is roughly `N × b`. During training, memory also holds gradients, optimizer states, temporary workspaces, and activations saved for backward.

For a conventional Adam/AdamW-style run, a rough planning estimate for weights + gradients + two FP32 optimizer moments + possibly a master-weight copy can be on the order of **16–20 bytes per parameter**, depending on parameter/gradient precision and implementation. This is only a baseline estimate; it excludes activations and framework overhead and is not a substitute for measuring peak memory.

| Quantity | Approximation | Why it matters |
|---|---|---|
| Parameter storage | `N × bytes_per_parameter` | Persistent weights |
| Gradients | often near one gradient tensor per parameter | Needed for backward/update |
| Optimizer state | depends on optimizer and state precision; Adam-style methods keep moments | Can be larger than the weights themselves |
| Activations | depends on layers, batch, sequence length, width, and checkpointing | Often a major training-memory cost |
| Attention intermediates | straightforward implementations can materialize per-head `[T,T]` tensors | Sequence length can sharply increase memory use |
| KV cache at inference | approximately `2 × layers × batch × sequence × kv_heads × head_dim × bytes` | Grows with generated/context tokens; formula depends on cache layout |

### Rough training-compute accounting

A widely used rough estimate for dense Transformer pretraining is:

```text
training FLOPs ≈ 6 × N_parameters × N_training_tokens
```

This is a planning heuristic, not an exact profiler result. It omits or simplifies effects such as attention's sequence-length-dependent cost, vocabulary projection, recomputation, padding, data movement, kernel efficiency, and architecture-specific operations. Use it to estimate orders of magnitude, then benchmark your actual code.

Example: a hypothetical 10-million-parameter model trained for 10-million tokens gives a rough order of `6 × 10^7 × 10^7 = 6 × 10^14` training FLOPs under that heuristic. Convert this to wall-clock estimates only if you have a credible sustained throughput measurement; peak hardware FLOPs are not the same as delivered training throughput.

### Memory/compute debugging checklist

- [ ] Measure peak memory at the actual batch and context length.
- [ ] Reduce batch size before changing architecture if the immediate goal is to fit the model.
- [ ] Use gradient accumulation only when you understand effective batch size and optimizer-step cadence.
- [ ] Reduce context length for early correctness experiments.
- [ ] Watch activation memory, not just parameter count.
- [ ] Consider activation checkpointing only after a correct baseline; it trades additional compute for less stored activation memory.
- [ ] Try mixed precision only after FP32 reference behavior is reliable, and verify loss/gradient stability.
- [ ] Benchmark dataloading, tokenization, and checkpoint writes as well as model kernels.
- [ ] Do not assume FlashAttention fixes every bottleneck; it primarily changes the memory-traffic behavior of attention.

### Why GPT-3-scale training is a separate systems problem

Training a model with hundreds of billions of parameters from random initialization requires a vastly larger compute, data, memory, interconnect, and engineering budget than a small personal experiment. Free or low-cost local resources are suitable for learning the mechanics, validating code, and conducting tightly scoped experiments—not for reproducing that scale of pretraining. The correct bridge is to learn distributed-training principles and scaling methodology after the single-device pipeline is sound.

---

## 8. Core experiments: do not skip them

Use the same protocol: establish a baseline, change one meaningful variable, preserve the data split, track compute, and write what the results do **not** prove.

### Experiment A — Gradient correctness

**Question:** do manually derived gradients match the numerical estimate and PyTorch autograd?

- Use a tiny network and deterministic, small-valued parameters.
- Compare analytical gradients with central finite differences on selected entries.
- Test multiple epsilon values and explain floating-point/finite-difference error.
- Include a deliberately incorrect transpose and confirm the test catches it.

### Experiment B — Overfit a tiny batch

**Question:** can the training code learn a tiny dataset at all?

- Train repeatedly on a tiny fixed batch.
- Expect training loss to fall strongly for a sufficiently expressive model and suitable optimization setup.
- If it cannot, inspect target shifts, loss reduction, parameter registration, gradients, learning rate, and optimizer-step execution.
- Do not use tiny-batch overfitting as evidence that generalization is good.

### Experiment C — Causal-mask integrity

**Question:** can future tokens leak into previous predictions?

- Compute logits on a sequence.
- Change token positions strictly after index `t`.
- Verify logits through index `t` remain unchanged within an appropriate numerical tolerance, with dropout disabled and all other conditions fixed.
- Assert that masked attention probabilities are effectively zero.

### Experiment D — Context length cost

**Question:** how do latency and memory scale as `T` changes?

- Benchmark several sequence lengths while holding model width, batch, dtype, and hardware constant.
- Record forward and forward+backward separately.
- Separate startup/compilation from steady-state timing.
- Explain the observed curve and deviations from ideal asymptotic complexity.

### Experiment E — Tokenizer trade-offs

**Question:** how does vocabulary size affect sequence length and model costs?

- Train or construct two tokenizers with different vocabulary sizes on a small permitted corpus.
- Compare tokens per document, sequence truncation, embedding/output parameter counts, loss, and speed.
- Do not compare raw perplexity across incompatible tokenizers without discussing the evaluation convention.

### Experiment F — Positional representation

**Question:** what changes when the position mechanism changes?

- Hold architecture, data, token budget, context, optimizer, and evaluation fixed as far as possible.
- Compare two position mechanisms.
- Repeat seeds if feasible.
- Report quality, speed, memory, and behavior at lengths not seen during training separately.

### Experiment G — Precision and kernel validation

**Question:** does an optimization improve throughput without unacceptable numerical deviation?

- Preserve a tested FP32 reference.
- Compare an optimized/mixed-precision implementation against it on fixed inputs.
- Check output and gradient errors, training stability, peak memory, and steady-state throughput.
- Document acceptable tolerances and the rationale for them.

### Experiment H — Small scaling study

**Question:** what trend can be observed as model size or training tokens increase?

- Choose several small configurations that fit the available resources.
- Record parameter count, data tokens, total training steps, compute/throughput estimates, validation loss, and wall time.
- Change one scaling axis at a time where possible.
- Fit a simple curve only after inspecting the raw points and uncertainty.
- Do not extrapolate a reliable frontier-scale prediction from a tiny, noisy range.

---

## 9. Testing standards for a research-quality implementation

### Unit tests

- Shapes: every module returns exactly the expected dimensions.
- Dtypes/devices: no unintended dtype conversion or host/device transfer.
- Numerical stability: no NaN/Inf from extreme but valid logits.
- Causality: future-token perturbation does not alter earlier outputs.
- Padding/ignored targets: loss excludes precisely the intended positions.
- Gradient checks: important small modules have numerical-vs-analytic checks.
- Checkpoint round-trip: saving and loading preserves outputs within expected tolerance.
- Generation: termination conditions work; token IDs decode correctly.
- Reproducibility: determinism claims are scoped to the operations/hardware where they actually hold.

### Integration tests

- A tiny model can complete one training step and update parameters.
- Training loss falls on a deliberately tiny training sample under a validated setup.
- Validation runs without accidentally updating weights or optimizer state.
- Checkpoint resume restores model and optimizer states.
- The tokenizer version used by the checkpoint matches the tokenizer at generation time.
- The end-to-end pipeline works from raw-text manifest to sample generation.

### Common failure modes

| Symptom | Plausible causes to inspect first |
|---|---|
| Loss is NaN/Inf | Learning rate too high, unstable softmax, invalid mask values, mixed-precision overflow, corrupt data, bad initialization |
| Loss does not move | Incorrect target shift, detached graph, missing `zero_grad`/optimizer step issue, parameters not registered, too low/high learning rate, constant or corrupted inputs |
| Training loss decreases but validation loss rises | Overfitting, data leakage/split mismatch, train/validation distribution shift, too much training for the data budget |
| Earlier logits change when future tokens change | Broken causal mask, wrong mask convention, unintended bidirectional attention, accidental preprocessing leakage |
| Model emits repeated tokens | Weak training signal, sampling setup, decoding bug, data repetition, insufficient data diversity, model too small or undertrained |
| OOM appears suddenly | Longer sequence, larger batch, retained computation graph, saved activations, optimizer states, fragmentation, unexpected tensor copies |
| GPU appears idle | Data loading, tokenization, CPU preprocessing, synchronization, small kernels, transfers, or insufficient batching |
| Faster kernel changes results | Different mask/dropout semantics, dtype tolerance, reduction order, bug in kernel, different implementation contract |

Never “fix” a failure by silently ignoring it. Capture the smallest failing case and turn it into a regression test.

---

## 10. Moving from implementation to research-level Transformer literacy

### Paper-reading procedure

Read a paper in four passes:

1. **Map:** title, abstract, main figure, core claim, baseline, and evaluation tasks.
2. **Mechanism:** write the core equations and draw the modified architecture; locate what changed relative to the baseline.
3. **Evidence:** inspect ablations, model/data budgets, training steps/tokens, hyperparameters, variance, and evaluation details.
4. **Reproduction:** implement the smallest meaningful variant, write tests, run a controlled comparison, and document deviations.

Questions to answer before accepting a claim:

- Is the comparison compute-matched, token-matched, parameter-matched, or only wall-clock-matched?
- Did the baseline receive comparable tuning and training duration?
- Is the result statistically/reproducibly distinguishable from seed noise?
- Are evaluation data contaminated by training data or preprocessing overlap?
- Is the proposed speedup due to fewer FLOPs, better IO, lower memory use, a different hardware fit, or measurement artifacts?
- Does the paper report the conditions where the method fails?
- Does a small-scale result support only the small-scale claim, or is the paper extrapolating further?

### Suggested paper order

1. **Attention Is All You Need** — foundational attention-based Transformer architecture. Re-derive scaled dot-product attention and multi-head attention.
2. **Deep Residual Learning for Image Recognition (ResNet)** — residual connections and optimization of deep networks; useful outside language modeling too.
3. **Layer Normalization** — normalization for recurrent/sequence-style models; derive its axes and parameterization.
4. **Language Models are Unsupervised Multitask Learners (GPT-2)** — decoder-only language modeling and zero-shot evaluation framing.
5. **Language Models are Few-Shot Learners (GPT-3)** — understand a large-scale study and its resource implications; do not use it as a personal compute target.
6. **Decoupled Weight Decay Regularization (AdamW)** — optimizer and weight-decay distinction.
7. **Training Compute-Optimal Large Language Models (Chinchilla)** — model/data allocation and compute-optimal training arguments.
8. **RoFormer: Enhanced Transformer with Rotary Position Embedding** — RoPE mechanism and relative-position behavior.
9. **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** — GPU memory hierarchy and IO complexity; attention function remains exact under the stated implementation.
10. **PaLM / SwiGLU discussion** — gated feed-forward variants and model design trade-offs.
11. **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints** — query-head versus key/value-head trade-offs and inference cache.
12. **A current end-to-end language-modeling course such as Stanford CS336** — tokenizer, PyTorch, compute/memory accounting, GPU kernels, parallelism, scaling, data, evaluation, and alignment. Treat course offerings and materials as versioned sources.

For each paper, produce a short record with the claim, the exact mechanism, an equation, an implementation plan, the necessary compute/data, and the key limitation.

### From “reading papers” to a research question

Use bounded questions. Good examples for a small setup:

- “At this model size and token budget, how do two position mechanisms change held-out loss and extrapolation behavior?”
- “Does the memory-efficient attention implementation preserve outputs and gradients within a stated tolerance, and what is its speedup at the sequence lengths my hardware can run?”
- “How does vocabulary size change sequence length, embedding/head parameters, and throughput on this fixed corpus?”
- “At a fixed total training-token budget, which of these small model configurations yields lower held-out loss, and how noisy is the observation across seeds?”

Avoid a research question such as “Which Transformer is best?” It lacks a defined data distribution, resource budget, metric, and comparison protocol.

---

## 11. Deliverables and mastery gates

Treat the following as repository artifacts, not as vague intentions.

### Gate A — Mathematics and backpropagation

- [ ] Written derivations for affine-layer gradients and softmax-cross-entropy gradient.
- [ ] Tiny scalar autodiff or equivalent reverse-mode explanation.
- [ ] NumPy MLP implementation.
- [ ] Finite-difference gradient-check tests.
- [ ] PyTorch-vs-reference gradient comparison.

### Gate B — Sequence modeling

- [ ] Tokenizer with tests.
- [ ] Bigram baseline.
- [ ] Small RNN language model.
- [ ] Correct next-token input/target shift.
- [ ] Held-out loss and generation script.

### Gate C — Transformer architecture

- [ ] Embedding and positional mechanism.
- [ ] Stable normalization.
- [ ] Causal attention reference implementation.
- [ ] Multi-head attention shape tests.
- [ ] Feed-forward sublayer.
- [ ] Residual Transformer block.
- [ ] Complete decoder-only language model.
- [ ] Causal-masking regression test.

### Gate D — From-scratch pretraining

- [ ] Documented, permitted training corpus and immutable data manifest.
- [ ] Randomly initialized parameters (no loaded pretrained weights).
- [ ] Training objective and target alignment documented.
- [ ] Validation, checkpointing, and checkpoint-resume test.
- [ ] Logged tokens/step, wall time, model configuration, optimizer schedule, loss, and memory.
- [ ] Generated samples and limitations report.

### Gate E — Systems and numerical analysis

- [ ] Parameter/memory budget worksheet.
- [ ] Training throughput benchmark with warmup and repeat measurements.
- [ ] Numerical comparison against a reference implementation.
- [ ] At least one profiling result identifying the actual bottleneck.
- [ ] One optimization with measured benefit and a regression test.

### Gate F — Research reproduction

- [ ] One precisely stated research question.
- [ ] Written baseline and ablation plan.
- [ ] Reproducible experiment configuration and data split.
- [ ] Results with resource accounting and uncertainty/limitations.
- [ ] Failed attempts and deviations recorded, not deleted.
- [ ] A short report separating observation from interpretation.

### Definition of complete

The Transformer phase is complete when all of the following are true:

```text
I can derive the main equations.
I can implement the core model from random initialization.
I understand the tensors and shapes in the forward pass.
I can explain how gradients reach every trainable component.
I can verify gradients and causality with tests.
I can train, evaluate, checkpoint, resume, and generate text.
I can measure speed and memory instead of guessing.
I know the data and evaluation limitations.
I can reproduce one controlled model comparison.
I can explain what a research paper changes and what its evidence supports.
```

---

## 12. Recommended references

Use original papers and maintained documentation for exact implementation details. Blog posts and videos can provide intuition, but they should not replace the paper, equation, tests, and measurements.

### Courses and implementation-oriented material

- [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) — end-to-end language-model construction, tokenization, PyTorch, compute/memory accounting, GPU kernels, parallelism, scaling, data, evaluation, and related topics.
- [Andrej Karpathy: Zero to Hero](https://karpathy.ai/zero-to-hero.html) — a practical derivation-first route through neural networks and language-model implementation.
- [nanoGPT repository](https://github.com/karpathy/nanoGPT) — a compact GPT implementation to inspect after attempting your own version; use it as a comparison target rather than as a substitute for deriving the model.
- [PyTorch autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html) — computational graph and reverse-mode automatic differentiation.
- [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html) — profiling runtime behavior and operator costs.
- [Triton documentation](https://triton-lang.org/main/index.html) — GPU kernels once the reference model and profiling workflow are sound.

### Foundational and systems papers

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Layer Normalization](https://arxiv.org/abs/1607.06450)
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165)
- [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101)
- [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)

### Data and tokenizer references

- [Neural Machine Translation of Rare Words with Subword Units (BPE in NMT)](https://arxiv.org/abs/1508.07909)
- [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759) — a controlled small-language-model setting; check dataset and paper terms before use.
- [The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale](https://arxiv.org/abs/2406.17557) — large-scale data curation study; useful for learning data-quality methodology, not a requirement to download a web-scale corpus.

When a source is updated, save the paper/version/commit used for an experiment. The most current documentation may evolve; the README's equations and principles are the starting point, not a guarantee that all APIs remain unchanged.

---

## 13. Important boundary: from this Transformer phase to a GPT-3-class goal

After the gates above, the next learning layer is **large-scale training systems**: data parallelism, FSDP/ZeRO-style optimizer-state sharding, tensor parallelism, pipeline parallelism, activation recomputation, distributed checkpointing, communication overlap, failure recovery, cluster scheduling, and end-to-end resource planning. Only after those foundations should you study the economics and engineering of pretraining at very large scale.

A GPT-3-class model is not simply the same small model with a larger `n_embd`. Large-scale pretraining combines model design, tokenizer and data mixture, optimization, distributed systems, failure recovery, evaluation, and significant compute budgets. You can build the architecture from scratch and reproduce its core objective at small scale now; reproducing the original large-scale training run is a separate resource-constrained project.

**Rule for the entire journey:** no fine-tuning shortcut is needed for this learning phase. Train the small model from random initialization, prove the implementation is correct, document its limits, and use research papers to make the next scaling decision.

# PART II — ADVANCED LLM ENGINEERING, POST-TRAINING, SELF-IMPROVEMENT, SAFETY & RESEARCH

> This part extends the first README from a small Transformer trained from random initialization to the broader engineering discipline behind modern assistant models. It covers public, reproducible concepts—not private or undocumented internals of any company. “Claude-engineer level” is treated as a bar for rigorous systems thinking, research literacy, evaluation, and production engineering, not as a claim that a public curriculum can reveal Anthropic's proprietary methods.

## 14. The modern LLM is a system, not only a neural network

A high-capability assistant is usually a stack of systems and training stages. Exact pipelines vary by lab and model family; the diagram below is a conceptual map, not a claim that every organization uses every stage.

```text
Data governance + corpus construction
        ↓
Tokenizer / multimodal encoders / data mixture
        ↓
Pretraining from random initialization
        ↓
Base model evaluation + checkpoint selection
        ↓
Supervised post-training (instruction / demonstrations)
        ↓
Preference optimization (e.g. DPO) and/or reward modeling + RL
        ↓
Tool-use, coding, reasoning, multimodal and domain-specific training
        ↓
Safety training + adversarial red-teaming
        ↓
System evaluations, calibration, release gates
        ↓
Inference runtime + routing + tools + retrieval + memory
        ↓
Monitoring, incident response, regression tests, controlled updates
```

### 14.1 Separate the objects you are building

| Object | What it is | Main question |
|---|---|---|
| Base model | Next-token predictor trained on a large corpus | What distribution has it learned? |
| Instruction model | Model trained to respond to requests and demonstrations | Does it follow instructions reliably? |
| Preference-aligned model | Model optimized using preference signals or reward objectives | Which responses does it prefer, and what are the proxy failures? |
| Tool-using assistant | Model plus tool schemas, runtime, permissions, and state | Can it take correct, authorized actions? |
| Product assistant | Complete user-facing system with UX, policy, logging, reliability, and operations | Is it useful, safe, dependable, and maintainable in real contexts? |
| Continually improved system | A governed process that evaluates candidate changes before promotion | Can it improve without silently regressing? |

Do not collapse these into the word “LLM.” A better base model does not automatically imply a better assistant, and a better benchmark score does not automatically imply a safer product.

## 15. Model-family literacy: what to compare in public research

Names such as GPT, Claude, Qwen, DeepSeek, and Kimi refer to evolving model families, not single fixed architectures. Some model releases are open-weight; others are API-only; available technical detail varies. Study public papers, model cards, technical reports, and released code. Never infer confidential implementation details from a product name or a benchmark table.

Compare model families along these dimensions:

- **Dense vs. Mixture-of-Experts (MoE):** all parameters participate per token versus routing tokens to a subset of experts.
- **Attention design:** MHA, MQA, GQA, sliding/local attention, latent or compressed KV representations, and context extension methods.
- **Position representation:** learned positions, sinusoidal encodings, RoPE variants, and their extrapolation limits.
- **Normalization and block layout:** Pre-LN/Post-LN, RMSNorm, residual scaling, and stability behavior.
- **Feed-forward design:** GELU, SwiGLU/GEGLU, expert FFNs, and activation/compute trade-offs.
- **Tokenizer and data:** vocabulary size, language coverage, code representation, deduplication, data mixture, and contamination controls.
- **Pretraining recipe:** token budget, batch construction, optimizer, learning-rate schedule, precision, and stability safeguards.
- **Post-training:** demonstrations, preference data, reward models, policy optimization, synthetic data, and verifier-based learning.
- **Inference:** KV-cache design, quantization, speculative decoding, batching, parallelism, and latency/throughput trade-offs.
- **Capabilities and limits:** coding, multilingual performance, long-context retrieval, math, multimodality, tool use, refusal behavior, calibration, and robustness.
- **Governance:** evaluation coverage, release policy, known limitations, model access, privacy, and incident response.

### 15.1 Architecture comparison notebook

For each public model report, record:

```text
Model / release / report date:
Public technical report:
Weight availability and license:
Dense or MoE (evidence / unknown):
Parameter count and active parameters (if disclosed):
Context window (exact version):
Tokenizer / vocabulary (if disclosed):
Attention / positional method (if disclosed):
Pretraining data and token budget (if disclosed):
Post-training methods (if disclosed):
Evaluation setup and contamination caveats:
Inference requirements / quantization options:
Safety and limitations:
What is known vs. what is speculation:
```

**Research rule:** explicitly label each statement as `reported`, `independently reproduced`, `inferred`, or `unknown`. This prevents marketing claims and community speculation from turning into false technical certainty.

## 16. Advanced architecture: MoE, long context, multimodality, and efficient inference

### 16.1 Mixture-of-Experts

A simplified sparse MoE layer uses a router to choose expert networks for each token. For token representation `h`, a router produces probabilities `p = softmax(W_r h)`. A top-k selection activates a small subset of experts, whose outputs are combined using routing weights.

Research questions:

- How do routing decisions affect expert specialization and token-level compute?
- How are load imbalance, expert collapse, and routing instability detected?
- What auxiliary load-balancing losses are used, and how can they conflict with the main objective?
- How do expert capacity, token dropping, all-to-all communication, and expert parallelism affect throughput?
- How do you distinguish total parameters from active parameters per token?

Experiments: implement a toy top-1 and top-2 router; log tokens per expert; plot routing entropy; compare dense FFN and MoE under matched training tokens and carefully reported compute. Do not claim an MoE is cheaper without including communication, memory, and serving costs.

### 16.2 Long-context systems

A large context window is not equivalent to reliable long-context reasoning. Separate:

- positional extrapolation from retrieval accuracy;
- maximum accepted input length from useful context length;
- needle retrieval from multi-document synthesis;
- attention complexity from KV-cache memory;
- context length from evidence attribution and resistance to distraction.

Test at several sequence lengths and positions, with distractors, repeated facts, conflicting evidence, and answerable/unanswerable cases. Report accuracy by token position and total inference cost.

### 16.3 Multimodal model foundations

A multimodal model must map different modalities into compatible representations while preserving their distinct structure. Study:

- image patches, vision encoders, and image-text alignment;
- audio tokenization, spectrograms, speech encoders/decoders, and streaming latency;
- video frame sampling, temporal alignment, and audio-video synchronization;
- modality-specific encoders versus a shared token space;
- cross-attention, modality adapters, and joint sequence modeling;
- data licensing, privacy, representation imbalance, and modality-specific safety evaluation.

Implement a small image-text contrastive experiment before attempting a general multimodal assistant. Measure retrieval, calibration, and failures—not just training loss.

### 16.4 Inference engineering

Understand the difference between prefill (processing the prompt) and decode (generating tokens). Study:

- KV cache memory and cache reuse;
- continuous batching and scheduling;
- tensor parallelism and communication overhead;
- quantization: weight-only, activation-aware, and KV-cache quantization;
- speculative decoding and acceptance rate;
- prefix caching, constrained decoding, and structured output;
- time-to-first-token, inter-token latency, tokens/second, throughput, and cost per request.

Benchmark quality and latency together. Quantization can change output quality; speculative decoding depends on acceptance behavior; batching can improve throughput while worsening individual latency.

## 17. Post-training: turning a base model into a useful assistant

This is a conceptual and experimental curriculum. Begin with a tiny model and small, appropriately licensed datasets; do not jump directly into expensive RL runs.

### 17.1 Supervised fine-tuning (SFT) as one stage—not the whole project

Although the overall goal includes training a base model from scratch, post-training is still required to understand how assistant behavior is shaped. SFT trains on prompt/response demonstrations, usually applying loss to the response tokens rather than indiscriminately to all prompt tokens.

Study:

- conversation serialization and role boundaries;
- target masking and loss masking;
- packing multiple examples without cross-example leakage;
- truncation and long-response handling;
- data deduplication, provenance, and train/evaluation separation;
- overfitting, response diversity, style imitation, and capability regression.

Exercise: train a small base model from random initialization, then run a clearly separated SFT experiment. Keep the base checkpoint and compare both on held-out tasks.

### 17.2 Preference learning

Preference data contains comparisons such as `(prompt, chosen response, rejected response)`. It is not an objective truth label; it represents judgments under a particular rubric, annotator population, and context.

Study:

- preference collection and annotation disagreement;
- reward models and pairwise ranking losses;
- reward hacking and Goodhart's law;
- KL constraints and reference policies;
- DPO-style direct preference optimization;
- reward-model calibration and out-of-distribution failures;
- disagreement between helpfulness, truthfulness, harmlessness, and user satisfaction.

Implement a toy pairwise preference model, inspect label noise, then run a small DPO experiment only after you can explain the objective and failure modes. Compare it against SFT on frozen evaluation sets.

### 17.3 Reinforcement learning for language models

Know the distinctions between policy, action/token, state/context, reward, return, advantage, value function, reference policy, and rollout. Study REINFORCE, PPO-style clipping, generalized advantage estimation, KL regularization, reward normalization, and off-policy versus on-policy data.

Language-model RL has practical hazards: expensive rollouts, reward-model exploitation, unstable updates, length bias, reward variance, and loss of useful capabilities. Before an RL experiment, write down:

1. the exact policy and objective;
2. the source and semantics of reward;
3. the reference policy and KL treatment;
4. rollout budget and compute estimate;
5. a frozen, held-out evaluation set;
6. stop conditions and rollback criteria;
7. regressions to monitor beyond reward.

Do not interpret an increasing scalar reward as proof of better reasoning or safer behavior.

### 17.4 Verifiable rewards and synthetic data

For domains with checkable answers—unit tests, exact arithmetic, formal proofs, schema validation, or constrained tasks—automated verifiers can provide a stronger signal than a subjective reward model. But verifiers can also be incomplete or gamed.

Study process supervision versus outcome supervision, rejection sampling, self-consistency, teacher-generated synthetic data, verifier design, and data contamination. Keep synthetic data provenance and filter rules; test for model collapse, diversity loss, and repeated teacher errors.

## 18. Self-improving LLMs: define the term precisely

“Self-improving” can describe several very different mechanisms. They must not be treated as interchangeable.

| Level | Mechanism | What changes? | Typical risk |
|---|---|---|---|
| 0 | Prompt iteration | Instructions or context | Prompt overfitting; hidden behavior changes |
| 1 | Retrieval/memory improvement | External knowledge store | Poisoning, stale facts, privacy leakage |
| 2 | Tool and workflow improvement | Tools, plans, orchestration | Unauthorized actions, brittle tool chains |
| 3 | Data curation loop | Training/evaluation examples | Contamination, feedback loops, synthetic-data collapse |
| 4 | Adapter or parameter update | A subset of learned weights | Forgetting, reward hacking, capability regression |
| 5 | Full model training update | Model parameters broadly | Expensive failure, drift, hard-to-debug regressions |
| 6 | Research automation loop | Hypotheses, experiments, code, model candidates | Self-confirming evaluations, unreviewed dangerous capability changes |

A system can improve at levels 0–3 without changing its core model weights. This can be safer and cheaper than continually updating parameters. First prove a measurable need for parameter updates rather than assuming that weight updates are inherently superior.

### 18.1 A governed improvement loop

```text
Observe failure or unmet requirement
        ↓
Create a reproducible failure case
        ↓
Classify: knowledge / reasoning / tool / UX / policy / model limitation
        ↓
Choose lowest-risk intervention
        ↓
Propose change (data, prompt, tool, adapter, or weights)
        ↓
Run unit + regression + safety + capability evaluations
        ↓
Compare with current champion on frozen and fresh tests
        ↓
Independent review / authorization
        ↓
Canary deployment with monitoring and rollback
        ↓
Retain audit record; promote only if predefined gates pass
```

### 18.2 Self-improving parameters: research agenda

Potential mechanisms include periodic retraining on curated feedback, adapters, online learning, continual learning, meta-learning, learned optimizers, and automated experiment search. Each changes different parts of the system and carries different risks.

Core questions:

- What exactly is the update rule, and which parameters can it modify?
- What is the provenance and consent status of every training example?
- How are user data, secrets, and untrusted tool outputs excluded or protected?
- How do you prevent evaluation data from entering training data?
- What is the forgetting profile across old and new capabilities?
- Is the improvement causal, or caused by randomness, data leakage, or benchmark overfitting?
- What is the rollback mechanism and the immutable baseline?
- Can the system modify its own evaluator, permissions, safety policy, or logging? It should not be allowed to silently weaken these controls.

Required experiments: replay buffer versus newest-only data; regularization versus no regularization; frozen backbone versus adapter updates; old-task/new-task performance matrix; privacy leakage probes; and ablation of feedback quality. Report uncertainty across multiple seeds when feasible.

### 18.3 Continual learning and catastrophic forgetting

Study rehearsal/replay, regularization-based approaches, parameter isolation, adapters, retrieval augmentation, and task/domain shift. Measure backward transfer, forward transfer, and forgetting—not only the score on the newest task. A “self-improving” system that improves a recent benchmark while forgetting previously reliable behavior is not an unqualified improvement.

### 18.4 Recursive self-improvement claims

Claims that a model can recursively improve itself should be decomposed into specific, testable capabilities: proposing experiments, writing code, predicting results, executing isolated tests, interpreting failures, and selecting candidates. Measure each stage separately. Do not assume autonomous research competence from strong coding benchmarks, and do not grant a self-modifying system unrestricted production access.

## 19. AI psychology, AI anxiety, and human factors

“AI psychology” is an interdisciplinary area, not proof that a model has human-like subjective experience. Separate three objects of study:

1. **Human psychology around AI:** trust, over-reliance, automation bias, anthropomorphism, AI anxiety, job-related concern, perceived agency, and emotional attachment.
2. **Observable model behavior:** linguistic expressions, apparent preferences, refusal patterns, role consistency, persuasion, sycophancy, and behavior under stress-test prompts.
3. **Claims about machine consciousness or subjective experience:** philosophically and scientifically contested; fluent first-person text alone does not establish experience.

### 19.1 AI anxiety research questions

AI anxiety can mean worry about job displacement, loss of control, privacy, misinformation, social change, or personal dependence. Study it with careful human-subject methods rather than treating it as a property of the model.

Research design:

- define the construct and target population;
- distinguish state anxiety from trait anxiety and general technology anxiety;
- use validated instruments where appropriate and cite their validation;
- pre-register hypotheses and primary outcomes;
- obtain informed consent and institutional ethics review where required;
- protect identity and sensitive responses;
- avoid deceptive or fear-inducing interventions without approved justification;
- report effect sizes, uncertainty, limitations, and demographic coverage;
- avoid making clinical diagnoses from model interactions or a single survey score.

Possible study: compare how different AI disclosure styles affect trust calibration and perceived control, with a pre-specified hypothesis and debriefing. Measure appropriate reliance—not simply whether participants trust the system more.

### 19.2 Anthropomorphism and over-reliance

Build tests for whether users mistake confidence, fluent language, or emotional phrasing for factual reliability. Evaluate calibrated reliance: do users accept correct answers and question incorrect ones? Useful interventions may include evidence display, uncertainty cues, source inspection, friction before consequential actions, and clear boundaries around what the system has actually done.

### 19.3 Model behavior evaluations

A model may produce emotional language because it learned language patterns and conversational roles. Describe behavior operationally: “the model generated an anxiety-related self-description under prompt X,” not “the model experienced anxiety.” For claims about consciousness, report uncertainty and avoid anthropomorphic certainty in either direction when evidence does not support it.

## 20. AI policy, governance, and safety engineering

Safety is not a single refusal prompt. It is a lifecycle of threat modeling, data governance, training interventions, system controls, evaluation, monitoring, and response.

### 20.1 Safety taxonomy

Evaluate at least these categories:

- harmful instructions and facilitation;
- self-harm and crisis-related interactions;
- privacy leakage and memorization;
- prompt injection and indirect prompt injection;
- jailbreak robustness and instruction hierarchy;
- tool misuse and unauthorized actions;
- cyber abuse and dual-use requests;
- biological/chemical dual-use risk;
- fraud, impersonation, and manipulation;
- misinformation, hallucinations, and false citations;
- bias, disparate performance, and language coverage;
- high-stakes domain limitations;
- data poisoning, model theft, and supply-chain compromise;
- autonomy, persistence, and uncontrolled external effects.

A risk assessment should consider likelihood, impact, exposure, detectability, reversibility, and mitigation strength. Do not treat one aggregate “safety score” as adequate.

### 20.2 Threat modeling for an assistant with tools

For each tool, specify:

```text
Tool name and purpose:
Allowed caller / identity:
Input schema and validation:
Data classification:
Permission scope:
Read-only or side-effecting:
External systems affected:
Confirmation requirement:
Rate / budget limit:
Timeout and retry policy:
Audit event fields:
Failure behavior:
Rollback / compensation:
Abuse cases and adversarial tests:
Owner and review date:
```

Use least privilege, explicit allowlists, sandboxing, secret isolation, human approval for high-impact actions, and independent authorization checks. A prompt saying “be safe” is not a security boundary.

### 20.3 Defense in depth

```text
Policy + data governance
   ↓
Training-time behavior shaping
   ↓
System prompt and instruction hierarchy
   ↓
Input / retrieval / tool-output screening
   ↓
Constrained tool permissions and sandbox
   ↓
Output checks where appropriate
   ↓
Human confirmation for high-impact operations
   ↓
Logging, monitoring, incident response, rollback
```

Every layer can fail. Design so a single model error does not immediately become an irreversible external action. Treat retrieved documents, browser pages, emails, and tool outputs as untrusted input; do not let them override system-level permissions.

### 20.4 Safety policy as a versioned artifact

A production policy should define scope, prohibited assistance, allowed safe alternatives, escalation behavior, high-stakes boundaries, privacy requirements, tool permissions, and evaluation examples. Version policy changes, map each rule to tests, and require review before release. Avoid vague rules that cannot be translated into observable tests.

### 20.5 Safety evaluation and red teaming

Use a matrix of policy category × attack strategy × language × context length × modality × tool access × severity. Include direct requests, role-play, multi-turn escalation, encoded or indirect requests, prompt injection in retrieved content, and benign near-neighbor requests to measure over-refusal.

Track at least:

- attack success rate with a documented scoring rubric;
- false refusal rate on benign cases;
- severity-weighted harm outcomes;
- privacy extraction success;
- tool authorization violations;
- robustness across languages and paraphrases;
- evaluator agreement and known blind spots;
- regression against the previous release.

Use independent reviewers for high-severity cases. Do not publish operationally dangerous exploit details unnecessarily; report enough methodology to enable responsible verification.

### 20.6 AI policy and standards literacy

Study the NIST AI Risk Management Framework, NIST Generative AI Profile, ISO/IEC 42001 concepts, relevant privacy and data protection law, copyright/licensing, sector-specific rules, and the EU AI Act where applicable. Standards and law change; confirm current editions, jurisdiction, and applicability with primary sources and qualified counsel. A framework checklist is not a legal opinion or a guarantee of compliance.

## 21. Interpretability and mechanistic understanding

Interpretability asks what internal representations and computations contribute to behavior. It does not provide a complete explanation of a model by default.

Study:

- probing and representation analysis;
- activation patching and causal interventions;
- attention analysis and its limitations;
- feature visualization and sparse autoencoders;
- circuits, induction heads, and algorithmic toy tasks;
- logit lens and intermediate predictions;
- causal mediation and counterfactual activation tests;
- mechanistic interpretability reproducibility and scaling limits.

For each interpretability claim, distinguish correlation from causal evidence. Attention weights alone are not a complete explanation. A feature that appears interpretable may be polysemantic, context-dependent, or an artifact of the method.

Project: train a small Transformer on a synthetic algorithmic task, form a hypothesis about an attention head or MLP feature, intervene on activations, and test whether the predicted behavior changes. Publish the negative results too.

## 22. Evaluation science: measure capability without fooling yourself

### 22.1 Build a layered evaluation suite

1. **Unit tests:** shapes, masks, loss, gradients, tokenizer round trips.
2. **Behavioral tests:** deterministic cases with expected properties.
3. **Capability tests:** coding, math, retrieval, instruction following, multilingual tasks, and tool use.
4. **Robustness tests:** paraphrase, distribution shift, adversarial context, noisy inputs.
5. **Safety tests:** policy compliance, jailbreak resistance, privacy, and tool authorization.
6. **Human evaluations:** blind pairwise comparisons with rubrics and adjudication.
7. **Production tests:** latency, cost, availability, timeouts, failure recovery, and observability.

### 22.2 Benchmark hygiene

- Freeze evaluation examples and track their hashes.
- Keep test sets separate from training and prompt optimization.
- Use fresh private tests for release decisions where feasible.
- Detect contamination and memorized answers.
- Report exact model version, decoding settings, prompt, tools, and scoring code.
- Use multiple seeds or bootstrap intervals when feasible.
- Report task-level results, not only one average.
- Track performance by language, task type, difficulty, and relevant population.
- Separate model capability from scaffold/tool advantage.
- Keep a record of evaluator changes and disagreement.

### 22.3 Evaluator models are fallible

LLM-as-judge evaluation can be useful but may have position bias, verbosity bias, style preference, correlated errors, and sensitivity to prompt framing. Calibrate judges against human-reviewed examples; use pairwise randomization, rubric-specific criteria, adversarial tests, and an uncertainty bucket for ambiguous cases. Never let a model be the only judge of its own improvement without independent checks.

## 23. Professional AI engineering: production architecture and reliability

The “professional” bar is not just advanced model architecture. It is also the ability to deliver a reliable, observable, secure, and maintainable system.

### 23.1 Reference architecture

```text
Client / API gateway
  ├─ identity, authN/authZ, quotas
  ├─ request validation and policy routing
  └─ trace / correlation ID
       ↓
Orchestrator
  ├─ model router and fallback
  ├─ prompt/version registry
  ├─ retrieval pipeline (if needed)
  ├─ tool registry + permission broker
  ├─ bounded planner / workflow state
  └─ timeout, retry, budget, cancellation
       ↓
Inference backend(s)
  ├─ base / instruct / specialized model
  ├─ quantization and batching policy
  └─ token streaming / structured output
       ↓
Post-process + validate
  ├─ schema / citation / policy checks
  ├─ output provenance and uncertainty
  └─ human approval when required
       ↓
Audit log + metrics + traces + feedback store
       ↓
Offline evaluation and governed improvement pipeline
```

### 23.2 Reliability patterns

Implement deadlines, bounded retries with backoff, idempotency keys, circuit breakers, cancellation, queue limits, load shedding, graceful degradation, and explicit fallback behavior. Retries can duplicate side effects unless operations are idempotent. A fallback model can change policy behavior and must be evaluated too.

### 23.3 Observability

Log privacy-safe metadata: model and prompt versions, latency, token counts, tool call names, status, error class, budget consumption, and evaluation labels. Avoid logging secrets or full sensitive conversations by default. Define retention, access control, deletion workflows, and incident access procedures.

Measure:

- request success and timeout rate;
- time-to-first-token and inter-token latency;
- input/output tokens and cost per successful task;
- tool success, retries, and authorization denials;
- schema validity and citation correctness;
- refusal / over-refusal rate;
- regression and incident rate;
- user correction rate and task completion (with privacy-respecting measurement).

### 23.4 Release engineering

Use immutable model artifacts, checksums, versioned datasets, configuration manifests, migration plans, canary releases, rollback, and post-release regression suites. Never automatically promote a candidate solely because its training loss or one benchmark improved.

## 24. Research operations and experimental discipline

Every experiment should have a one-page protocol before code changes:

```text
Question:
Hypothesis:
Baseline:
Independent variable:
Controlled variables:
Dataset / version / license:
Model / tokenizer / code commit:
Metric(s) and primary endpoint:
Compute budget:
Number of seeds:
Expected outcomes:
Failure criteria:
Safety and data risks:
Reproduction command:
Result location:
Decision rule:
```

Use configuration files, deterministic seeds where possible, environment lockfiles, artifact hashes, and experiment tracking. Determinism is not guaranteed across all hardware and kernels; record that limitation. Save logs, optimizer state, RNG state, data sampler position, and scheduler state if you need true training resumption.

### 24.1 Negative results are valuable

Record failed runs, unstable hyperparameters, out-of-memory events, dataset problems, and regressions. Do not silently discard runs that contradict the preferred hypothesis. A rigorous research engineer can explain why an experiment failed and what the result rules out.

## 25. Hardware and scaling from a small lab to distributed training

Scale in stages. Learn the failure modes of a single machine before adding distributed complexity.

### Stage A — CPU / low-memory machine

- implement tensor operations and a tiny model;
- use short sequences, small batches, and a small vocabulary;
- validate correctness and profile bottlenecks;
- expect training to be slow; focus on experiments rather than parameter count.

### Stage B — one GPU or temporary accelerator

- estimate memory before launching;
- profile data loading, kernels, attention, and optimizer step;
- use gradient accumulation when appropriate;
- consider mixed precision only after numerical tests;
- save checkpoints frequently enough for the interruption risk.

### Stage C — multi-GPU

Study data parallelism, DistributedDataParallel, Fully Sharded Data Parallel, ZeRO concepts, tensor parallelism, pipeline parallelism, sequence/context parallelism, communication overlap, and distributed checkpointing. Understand network bandwidth, all-reduce/all-gather costs, stragglers, and topology.

### Stage D — cluster-scale training

Add job scheduling, node failure recovery, health checks, checkpoint integrity, data streaming, storage bandwidth, quota management, fault injection, and cost accounting. Distributed training is a systems project as much as a modeling project.

Before every run estimate:

```text
parameter memory
+ gradients
+ optimizer states
+ activations
+ temporary workspaces
+ KV cache (inference)
+ communication buffers
+ framework/runtime overhead
```

Measure actual peak memory; estimates are not guarantees. Do not assume free cloud accelerators are persistent, available, or suitable for long uninterrupted training.

## 26. Capstone sequence: from a tiny model to a governed assistant

Complete these in order. Each capstone has a research artifact and a reproducible test suite.

### Capstone A — Base model from random initialization

- train a small decoder-only Transformer;
- implement tokenizer and data pipeline;
- validate causal masking and gradients;
- report validation NLL/perplexity and samples;
- include a compute and data statement.

### Capstone B — Instruction-following model

- build a licensed, curated demonstration dataset;
- implement correct response-only loss masking;
- compare base and SFT checkpoints;
- evaluate instruction following and regressions.

### Capstone C — Preference optimization

- collect or use appropriately licensed preference pairs;
- document rubric and disagreement;
- implement a toy reward model or DPO objective;
- evaluate helpfulness, truthfulness, refusal, and style separately.

### Capstone D — Tool-using assistant

- define a narrow tool set with schemas;
- implement authorization outside the model;
- sandbox untrusted code and retrieved content;
- add confirmation for side effects;
- test prompt injection and malformed tool calls.

### Capstone E — Governed self-improvement loop

- create reproducible failure cases;
- generate candidate fixes;
- run frozen capability, safety, and regression suites;
- require human approval for promotion;
- canary, monitor, and roll back;
- compare prompt/retrieval fixes against parameter updates.

### Capstone F — Safety and human-factors study

- select a narrow question about AI anxiety, trust, anthropomorphism, or calibrated reliance;
- obtain ethical approval where needed;
- use consent and privacy-preserving data handling;
- pre-register measures and analyze uncertainty;
- avoid claims about machine subjective experience unsupported by evidence.

### Capstone G — Research reproduction

- select a public paper with a feasible experiment;
- reproduce a small central claim;
- document deviations, compute, and negative results;
- compare your result with the paper and discuss limitations.

## 27. Mastery gates for advanced LLM engineering

You are ready to move forward only when you can demonstrate—not merely describe—the following:

- [ ] Derive and implement the core Transformer forward and backward passes.
- [ ] Explain optimizer behavior and diagnose unstable training.
- [ ] Reproduce a tiny base-model training run from random initialization.
- [ ] Explain why SFT, preference optimization, and RL solve different problems.
- [ ] Build an evaluation suite with contamination controls and confidence estimates.
- [ ] Demonstrate that a proposed improvement survives regression and safety tests.
- [ ] Explain catastrophic forgetting and measure old/new task performance.
- [ ] Threat-model tool use and enforce authorization outside the LLM.
- [ ] Test prompt injection, data leakage, and unsafe side effects.
- [ ] Benchmark inference latency, throughput, memory, and cost.
- [ ] Produce a reproducible paper replication with explicit limitations.
- [ ] Distinguish empirical behavior from claims about model psychology or consciousness.
- [ ] Write a versioned safety policy linked to observable test cases.
- [ ] Operate a canary deployment with monitoring and rollback.
- [ ] Maintain a clear provenance trail for data, weights, prompts, policies, and evaluation.

## 28. Suggested advanced reading map

Read original papers and technical reports in a question-driven order. The list is not a claim that all methods are used by every model family.

### Pretraining and scaling

- [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556)
- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311)

### Efficiency and architecture

- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- [RoFormer / Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Switch Transformers](https://arxiv.org/abs/2101.03961)

### Instruction and preference alignment

- [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- [Training a Helpful and Harmless Assistant with RLHF](https://arxiv.org/abs/2204.05862)

### Evaluation and behavior

- [HELM: Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110)
- [TruthfulQA](https://arxiv.org/abs/2109.07958)
- [SWE-bench](https://arxiv.org/abs/2310.06770)
- [BIG-bench](https://arxiv.org/abs/2206.04615)

### Interpretability and mechanistic analysis

- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
- [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/index.html)

### Safety and governance

- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
- [NIST AI RMF Generative AI Profile](https://www.nist.gov/itl/ai-risk-management-framework)
- [Anthropic research and publications](https://www.anthropic.com/research)
- [OpenAI research](https://openai.com/research/)
- [DeepSeek publications](https://github.com/deepseek-ai)
- [Qwen technical reports and releases](https://github.com/QwenLM)
- [Moonshot AI / Kimi public materials](https://www.moonshot.ai/)

Always verify paper version, date, released code, model license, and evaluation protocol. A public technical report is evidence for what it actually discloses—not proof of undocumented details.

## 29. Definition of professional readiness

Professional readiness is not “I know all the newest model names.” It means you can take a requirement from question to reliable evidence:

1. formulate the problem and risk boundary;
2. choose a baseline and justify the design;
3. derive the relevant objective and understand its assumptions;
4. implement a minimal reference version;
5. test correctness and reproduce results;
6. profile the system and explain the resource budget;
7. evaluate capability, robustness, and safety;
8. document limitations and uncertainty;
9. deploy with least privilege, observability, and rollback;
10. improve only through controlled, auditable experiments.

The standard is not perfection or pretending to know proprietary systems. It is being able to distinguish evidence from speculation, explain trade-offs, build a reproducible implementation, and operate it responsibly.
