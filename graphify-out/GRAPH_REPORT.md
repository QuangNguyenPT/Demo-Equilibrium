# Graph Report - Demo  (2026-09-27)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 1522 nodes · 3829 edges · 104 communities (72 shown, 32 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 216 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Community 0
- Community 1
- Community 2
- Community 3
- Community 4
- Community 5
- Community 6
- Community 7
- Community 8
- Community 9
- Community 10
- Community 11
- Community 12
- Community 13
- Community 14
- Community 15
- Community 16
- Community 17
- Community 18
- Community 19
- Community 20
- Community 21
- Community 22
- Community 23
- Community 24
- Community 25
- Community 26
- Community 27
- Community 28
- Community 29
- Community 30
- Community 31
- Community 32
- Community 33
- Community 34
- Community 35
- Community 36
- Community 37
- Community 38
- Community 39
- Community 40
- Community 41
- Community 42
- Community 43
- Community 44
- Community 45
- Community 46
- Community 47
- Community 48
- Community 49
- Community 50
- Community 51
- Community 52
- Community 53
- Community 54
- Community 55
- Community 56
- Community 57
- Community 58
- Community 59
- Community 60
- Community 61
- Community 62
- Community 63
- Community 64
- Community 65
- Community 66
- Community 67
- Community 68
- Community 69
- Community 70
- Community 71
- Community 72
- Community 73
- Community 74
- Community 75
- Community 76
- Community 77
- Community 78
- Community 79
- Community 80
- Community 81
- Community 82
- Community 83
- Community 84
- Community 85
- Community 86
- Community 87
- Community 88
- Community 89
- Community 90
- Community 91
- Community 92
- Community 93
- Community 94
- Community 95
- Community 96
- Community 97
- Community 98
- Community 99
- Community 100
- Community 102
- Community 103

## God Nodes (most connected - your core abstractions)
1. `Program` - 83 edges
2. `Model` - 65 edges
3. `KnowledgeStore` - 58 edges
4. `Domain` - 57 edges
5. `ModelConfig` - 33 edges
6. `PhasorVector` - 33 edges
7. `DomainRegistry` - 33 edges
8. `SubstrateConfig` - 30 edges
9. `Translate` - 27 edges
10. `ProgramError` - 27 edges

## Surprising Connections (you probably didn't know these)
- `CertifiedLinear` --implements--> `T5 Exact spectral certificate (Cayley x sigmoid-diag)`  [EXTRACTED]
  hrm_eqb/certified.py → vault/01 - Architecture Blueprint/HRM-EQB Specification v0.2 - Theorems and Proofs.md
- `ContractionBounds` --implements--> `C1 Construction certificate Γ < 1`  [EXTRACTED]
  hrm_eqb/equilibrium.py → vault/01 - Architecture Blueprint/HRM-EQB Specification v0.2 - Theorems and Proofs.md
- `Neuromodulators` --implements--> `H1 Hormones are stable and cannot break convergence`  [EXTRACTED]
  hrm_eqb/neuromodulation.py → vault/01 - Architecture Blueprint/HRM-EQB Specification v0.2 - Theorems and Proofs.md
- `EquilibriumCore` --implements--> `C1 Construction certificate Γ < 1`  [EXTRACTED]
  hrm_eqb/equilibrium.py → vault/01 - Architecture Blueprint/HRM-EQB Specification v0.2 - Theorems and Proofs.md
- `cayley()` --implements--> `T5 Exact spectral certificate (Cayley x sigmoid-diag)`  [EXTRACTED]
  hrm_eqb/certified.py → vault/01 - Architecture Blueprint/HRM-EQB Specification v0.2 - Theorems and Proofs.md

## Import Cycles
- 3-file cycle: `hrm/perception/grid/__init__.py -> hrm/perception/grid/domain.py -> hrm/perception/grid/programs.py -> hrm/perception/grid/__init__.py`

## Hyperedges (group relationships)
- **Certified equilibrium: operators -> Γ -> convergence -> halting** — vault_01___architecture_blueprint_hrm_eqb_specification_v0_2___theorems_and_proofs_t5, vault_01___architecture_blueprint_hrm_eqb_specification_v0_2___theorems_and_proofs_s1, vault_01___architecture_blueprint_hrm_eqb_specification_v0_2___theorems_and_proofs_n1, vault_01___architecture_blueprint_hrm_eqb_specification_v0_2___theorems_and_proofs_c1, vault_01___architecture_blueprint_hrm_eqb_specification_v0_2___theorems_and_proofs_e2, vault_01___architecture_blueprint_hrm_eqb_specification_v0_2___theorems_and_proofs_e3, vault_01___architecture_blueprint_hrm_eqb_specification_v0_2___theorems_and_proofs_e6 [EXTRACTED 1.00]
- **Cognition: hormones, curiosity, keep gate** — vault_01___architecture_blueprint_hrm_eqb_specification_v0_2___theorems_and_proofs_h1, vault_01___architecture_blueprint_hrm_eqb_specification_v0_2___theorems_and_proofs_q1, vault_01___architecture_blueprint_hrm_eqb_specification_v0_2___theorems_and_proofs_g1, vault_01___architecture_blueprint_hrm_eqb_specification_v0_2___theorems_and_proofs_hormone_system, vault_01___architecture_blueprint_hrm_eqb_specification_v0_2___theorems_and_proofs_curiosity_protocol [EXTRACTED 1.00]
- **4-D knowledge: quaternion relations in SSD product-key memory** — vault_01___architecture_blueprint_hrm_eqb_specification_v0_2___theorems_and_proofs_t3, vault_01___architecture_blueprint_hrm_eqb_specification_v0_2___theorems_and_proofs_t4, vault_01___architecture_blueprint_hrm_eqb_specification_v0_2___theorems_and_proofs_k1, vault_01___architecture_blueprint_hrm_eqb_specification_v0_2___theorems_and_proofs_m1, vault_01___architecture_blueprint_hrm_eqb_specification_v0_2___theorems_and_proofs_f2 [EXTRACTED 1.00]

## Communities (104 total, 32 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (38): HashingLanguage, OllamaLanguage, LanguageSection, Language backends. HashingLanguage dependency-free: hashed bag-of-words…, TransformersLanguage, LanguageError, LanguageSection, ABC (+30 more)

### Community 1 - "Community 1"
Cohesion: 0.07
Nodes (19): MetaConfig, Metacognition: intent inference, the competence map, curiosity., Curiosity, Curiosity: the model's own choice of what to study next. Study where learning…, Next region to study, or None when nothing learnable is left., IntentHead, Message, ndarray (+11 more)

### Community 2 - "Community 2"
Cohesion: 0.09
Nodes (29): dataclasses, datetime, hashlib, GeneratedTask, ABC, Task generators: practice material with ground truth, per domain and family. A…, Steps in the shortest program that explains the family (0 = not in the…, SequenceTasks (+21 more)

### Community 3 - "Community 3"
Cohesion: 0.08
Nodes (24): FeedbackResult, _ask(), _explain(), _feedback(), _promote(), Command line for a model bundle. python -m hrm create models/main --benchmark…, _report(), _resign() (+16 more)

### Community 4 - "Community 4"
Cohesion: 0.09
Nodes (14): collections, Affine, CumSum, Diff, ABC, Sequence domain: integer sequences in, integer sequences out. Encoding:…, Reverse, Rotate (+6 more)

### Community 5 - "Community 5"
Cohesion: 0.09
Nodes (17): KnowledgeStore, KnowledgeStoreError, Any, Path, RuntimeError, Thread-safe; one instance per model bundle directory., Persist buffered usage counters (retrieval does not write per call)., Atomic group of changes; nests via savepoints. Raise ``Rollback`` to discard. (+9 more)

### Community 6 - "Community 6"
Cohesion: 0.11
Nodes (39): dtype, as_complex(), as_real(), cleanup(), phasor_attention(), phasor_codebook(), Tensor, qconj() (+31 more)

### Community 7 - "Community 7"
Cohesion: 0.11
Nodes (26): ast, The core's weights: everything learned that lives in RAM. Small by design: the…, Intent: what does the sender want from this input? answer solve the task and…, int_kwargs(), ProgramError, ValueError, Perception: the Domain protocol. A domain turns raw examples into phasor scenes…, True if `args` is a non-empty dict of allowed keys with bounded int values. (+18 more)

### Community 8 - "Community 8"
Cohesion: 0.13
Nodes (31): radial_relu(), Per-quaternion max(0, |x| - shrink) x/|x| on (..., 4k): 1-Lipschitz, rotation-…, Context, EquilibriumCore, Tensor, Two-level equilibrium core with a convergence certificate computed from its…, w_h ||Δh|| + w_l ||Δl|| over the last two dims (tokens, features)., Fast and slow levels on (tokens, 4k) states, contractive for every weight value… (+23 more)

### Community 9 - "Community 9"
Cohesion: 0.13
Nodes (14): New bundle. `benchmark` is copied in and locked; without one, the sleep cycle…, LanguageSection, TestPerceive, TestBundle, benchmark(), composite_task(), config(), learn_composite() (+6 more)

### Community 10 - "Community 10"
Cohesion: 0.19
Nodes (31): subprocess, cfg(), key(), fixture, PhasorVector, SSD knowledge store: persistence, write gate, copy-on-write, rollback, crash…, record(), Report (+23 more)

### Community 11 - "Community 11"
Cohesion: 0.12
Nodes (10): as_vector3d(), _is_number(), Any, Example, The single-op vocabulary, drawn from the one shared definition. `Project` is…, True only when the composition really does collapse to a single op. The base…, Read a stored 3-vector as float. The base implementation casts to int64, which…, A 3-vector of finite numbers, else None. Accepts the dense output an embedding… (+2 more)

### Community 12 - "Community 12"
Cohesion: 0.16
Nodes (21): _build(), _coerce(), ConfigError, ModelConfig, Any, Path, ValueError, Model configuration: the single source of truth for every tunable. Each model… (+13 more)

### Community 13 - "Community 13"
Cohesion: 0.13
Nodes (11): Two-level reasoning: fast L-level resonator ticks inside deep H-level cycles., ReasoningConfig, BudgetController, Track energy improvement; returns ticks since the last improvement., MixingHead, M_H: learnable base logits + inter-stream coherence, Sinkhorn-projected., features(), Regression: kinetic + semantic + NOT verified + budget spent used to fall… (+3 more)

### Community 14 - "Community 14"
Cohesion: 0.13
Nodes (8): shift(), Translate, parametrize, grid_demos(), prog(), TestGridEncodingAlgebra, TestGridOperators, top_by_shared()

### Community 15 - "Community 15"
Cohesion: 0.16
Nodes (13): _Done, Any, Example, Trace, VerificationReport, Simplest (best MDL) program among `programs` that passes verification., Builds the Solution: executes the chosen program on the test input and measures…, Other exact, non-memorizing explanations: do they predict the same answer? (+5 more)

### Community 16 - "Community 16"
Cohesion: 0.14
Nodes (16): fixture, GridTasks, LearningConfig, PerceptionConfig, Sleep cycle: consolidation behind the frozen benchmark gate., Domain vocabularies (bounds on the enumerated base hypotheses)., StudyBudget, SequenceTasks (+8 more)

### Community 17 - "Community 17"
Cohesion: 0.09
Nodes (13): _literal(), Any, Graded agreement in [0, 1]; exactly 1 iff ``equal``., Number of atomic elements in an input (bounds program complexity)., Build a program from parsed ``name(args)`` terms; ValueError if invalid., Inverse of ``Program.describe``: ``"translate(dx=1, dy=0) .…, Plain-text form of one input/output: rows of numbers separated by ' / '., JSON-serializable form of one input/output (for episodic memory). (+5 more)

### Community 18 - "Community 18"
Cohesion: 0.12
Nodes (6): New random symbol not tied to any name., PhasorVector, ndarray, Unit-phasor vector. ``a * b`` bind, ``a / b`` unbind, ``a ** x`` power, ``~a``…, Rotate every phase by factor x; exact for any real x., Phase-space perturbation z * exp(i sigma xi): the restart kick.

### Community 19 - "Community 19"
Cohesion: 0.13
Nodes (9): CycleReport, KnowledgeItem, gate(), Reasons a candidate core must be discarded (empty list = keep)., Retire from SSD retrieval what the live core now solves without a lookup., SleepCycle, One consolidation cycle behind the frozen benchmark gate., result() (+1 more)

### Community 20 - "Community 20"
Cohesion: 0.18
Nodes (13): Enum, Action, ControllerFeatures, Decision, Budget controller: ACCEPT / CONTINUE / RESTART / ABSTAIN plus the KM…, Exploratory (low lambda) far from acceptance, damped (high) near it., featurize(), ndarray (+5 more)

### Community 21 - "Community 21"
Cohesion: 0.11
Nodes (13): argparse, GridTasks, BenchmarkTask, Harness, HarnessLockError, Path, RuntimeError, True when some single enumerated program reproduces the demonstrations. An… (+5 more)

### Community 22 - "Community 22"
Cohesion: 0.11
Nodes (9): GridProgram, Example, A recolor table consistent with every demo, if the data implies one., Identity, Ortho3D, Primitive, ABC, ndarray (+1 more)

### Community 23 - "Community 23"
Cohesion: 0.13
Nodes (7): InnerSpeech, LanguageSection, One sentence in plain language, followed by the literal statement., DomainRegistry, Name -> domain instance; resolves which domain can represent a task., TestInnerSpeech, TestRegistry

### Community 24 - "Community 24"
Cohesion: 0.13
Nodes (11): Rotate, prog(), Composability must be decided by algebra, not by family name. The base test…, Every pair: pruned iff the composition is literally a single op., A depth label is a claim about the task and must be true of the task., The whole point: a depth-3 program must be accepted, not called memorized., Guards the original defect: k random rotations that collapse to one op. The…, Apply ops in order, independent of VectorProgram, as a cross-check. (+3 more)

### Community 25 - "Community 25"
Cohesion: 0.10
Nodes (16): contextlib, Exception, KnowledgeConfig, SSD-resident knowledge store (keys on disk, streamed in chunks)., AnswerRecord, Episode, SSD-resident knowledge: the dynamic part of the model file. Layout inside the…, Raise inside ``store.transaction()`` to discard it without an error. (+8 more)

### Community 26 - "Community 26"
Cohesion: 0.15
Nodes (13): cayley(), CertifiedLinear, Tensor, Operators whose Lipschitz constants are known exactly from their parameters…, Square map W = Q1 diag(s) Q2 with s = budget * sigmoid(logit), so ||W||_2 = max…, Singular values of W., ||W||_2 of the matrix actually used in compute (float64 SVD)., Scale a nonnegative (..., n, n) matrix so row and column sums are <= 1, hence… (+5 more)

### Community 27 - "Community 27"
Cohesion: 0.17
Nodes (6): Codebook, _name_hash(), Atomic symbol with i.i.d. uniform phases (near-orthogonal to all other names)., Symbol whose phases are multiples of 2*pi/period, so ``c ** period`` is the…, ``symbol(name) ** exponent``, or ``circular(name, period) ** exponent``., TestCodebook

### Community 28 - "Community 28"
Cohesion: 0.19
Nodes (8): GridObject, _color(), GridDomain, _is_grid(), Any, PhasorVector, Partial credit: exact cells, nonzero-mask IoU (structure), color histogram…, Superposition of every non-background cell: a checksum, never decoded cell-wise.

### Community 29 - "Community 29"
Cohesion: 0.15
Nodes (10): _key(), ndarray, 3-vectors in, 3-vectors out — the geometry axis of benchmark v2. Ground truth…, Generators of the deep families, from the domain's shared vocabulary. Must stay…, Fewest enumerated single ops whose product equals `ops`, or None past `limit`.…, A product of exactly `depth` steps that nothing shorter explains. Draws from…, Unrelated vectors: no op composition explains them., Unrelated input/output pairs: no program explains them. (+2 more)

### Community 30 - "Community 30"
Cohesion: 0.15
Nodes (10): CandidateSet, Codebook, ndarray, (D, C) conjugated signatures: `states @ conj_t` gives D * similarities., (..., D) states -> (..., C) similarities in [-1, 1]., Lift an existing candidate's prior, or append the program with it., Parameters (the core's search scalars): log_beta per-stream softmax inverse…, (C,) how well each candidate explains the task core, disambiguated by the… (+2 more)

### Community 31 - "Community 31"
Cohesion: 0.18
Nodes (12): deque, TaskEncoding, EquilibriumEngine, EquilibriumResult, EnergyFn, ndarray, PhasorVector, Trace (+4 more)

### Community 32 - "Community 32"
Cohesion: 0.20
Nodes (10): _depth_band(), Solve every task with the (read-only) reasoner and score exactly. Tasks are…, Group tasks by how many non-identity steps their explanation needs., Learner, Any, Example, Solution, Record the interaction; store a newly synthesized explanation. Returns its id. (+2 more)

### Community 33 - "Community 33"
Cohesion: 0.13
Nodes (9): Project, ABC, ndarray, Mirror one coordinate: a signed permutation, exact in floating point., Zero one component: a lossy step, so a program using it is not invertible., Exact rotation matrix. Right angles and cube diagonals are exact in binary…, Reflect, _rot_matrix() (+1 more)

### Community 34 - "Community 34"
Cohesion: 0.22
Nodes (18): HRM-EQB Specification v0.2 (Theorems and Proofs), C1 Construction certificate Γ < 1, E1 Fast-level equilibrium l*(h), E2 Nested equilibrium existence and uniqueness, E3 HRM schedule converges for every n_L, E4 Neumann gradient error bound, E5 Context-independent convergence and sensitivity, E6 Certified halting (+10 more)

### Community 35 - "Community 35"
Cohesion: 0.17
Nodes (13): DepthAttention, Tensor, Attention Residuals (arXiv:2603.15031) over a chain of states: softmax over…, u_q = Σ_i softmax_i(w_q · RMSNorm(v_i)) v_i with zero-initialized pseudo-…, (queries, S, ...) depth weights for a (S, ..., dim) chain; each sums to 1 over…, (S, ..., dim) chain -> (queries, ..., dim) convex combinations of its states., RMSNorm without a learned scale., rms_normalize() (+5 more)

### Community 36 - "Community 36"
Cohesion: 0.13
Nodes (12): gravity(), _mat(), _program(), Any, Every colored cell falls to the bottom of its column (outside the grid…, Matrix product of rotation/linear ops, rounded to the exact lattice., Fall to the +y axis: drop every horizontal component (outside the vector3d…, A random concrete transformation from `family` (e.g. one fixed shift). (+4 more)

### Community 37 - "Community 37"
Cohesion: 0.22
Nodes (8): Domain, True when composing ``first`` and ``then`` adds no expressive power. The…, The program language, described for the language section's prompts., Program that applies ``first`` and then ``then``., Example, (semantic energy, verification margin) for the reasoning loop., Largest program complexity that cannot be memorization. Two bounds, and the…, Verifier

### Community 38 - "Community 38"
Cohesion: 0.17
Nodes (10): _program_matrix(), 3x3 matrix of a linear program, by running it on the basis vectors. Every…, Evaluation, The independent judge. Runs the executable face of a program against every…, VerificationReport, itertools, os, Language section: program language parsing, inner speech, text perception,… (+2 more)

### Community 39 - "Community 39"
Cohesion: 0.21
Nodes (6): as_sequence(), Any, Example, Integer y = a x + b fitted across all demos, if the data implies one., Plain list of ints for a 1-D integer sequence, else None., SequenceDomain

### Community 40 - "Community 40"
Cohesion: 0.17
Nodes (8): ContractionBounds, Norm bounds of the four certified maps plus step sizes; decides provable…, Γ = v_h + v_l s_h / (1 - s_l); the model is certified iff Γ < 1., (ρ_L, κ_L, a, b) = Lip_l T_L, Lip_h T_L, Lip_h T_H, Lip_l T_H., Nested contraction rate a + b κ_L / (1 - ρ_L) of h -> T_H(h, l*(h)) (E2)., M_n bounding one cycle componentwise (E3)., Cycle rate r_n and left Perron weights (1, w_l) of M_n, normalized so w_h = 1…, Lipschitz constant of one cycle in (c_L, c_H), weighted norm vs ||Δc_L|| +…

### Community 41 - "Community 41"
Cohesion: 0.21
Nodes (7): KnowledgeRecord, ndarray, PhasorVector, A piece of knowledge a domain wants stored., Store verified knowledge; returns the item id, or None if refused., Top-k active items by similarity + confidence bonus - complexity penalty., Retrieved

### Community 42 - "Community 42"
Cohesion: 0.14
Nodes (5): PriorTable, Any, Raise the program's prior by `amount` (capped); returns the new prior., Drop a consolidated program (e.g. a human said it produced a wrong answer)., _Key

### Community 43 - "Community 43"
Cohesion: 0.17
Nodes (7): pytest, grid_task(), model(), fixture, Model bundle lifecycle and online learning (loop A): episodes, answers,…, TestFeedback, TestOnlineLearning

### Community 44 - "Community 44"
Cohesion: 0.16
Nodes (10): ArgumentParser, LanguageConfig, The native language section., main(), _parser(), json, Offsets, Command line: create, ask, feedback with correction, explain, report, study,… (+2 more)

### Community 45 - "Community 45"
Cohesion: 0.14
Nodes (10): Neuromodulators, Tensor, Hormone system: four global neuromodulators in a box, updated by a projected…, Levels m in [lower, upper] ⊂ (0, 1)^4; step: m <- Π_box((1 - rate) m + rate…, Start at the box midpoint., One projected EMA update toward drive; returns the new levels., Level of one hormone by name., Maximum refinement segments ⌈1 / (1 - serotonin)⌉ (patience as a discount). (+2 more)

### Community 46 - "Community 46"
Cohesion: 0.18
Nodes (10): _new_task_id(), Example, Answer a task; what is learned from it is kept (verified only)., Learn from demonstrations: find and keep a verified explanation., Infer what the sender wants, then do it, or ask when unsure. A task written as…, Human: this is what that message meant. Trains the intent matrix., What `perceive` did with a message, or what it needs to know first., Response (+2 more)

### Community 47 - "Community 47"
Cohesion: 0.24
Nodes (10): decode_angles(), encode_angles(), ndarray, quantization_error(), Phase quantization: unit phasors stored as uint16 angle codes. A unit phasor…, complex array -> uint16 angle codes (same shape)., uint16 angle codes -> unit phasors., Max |v - decode(encode(v))|. (+2 more)

### Community 48 - "Community 48"
Cohesion: 0.14
Nodes (6): Codebook, Example, PhasorVector, Knowledge-store key: the task's invariant core, tagged with this domain., True when every example is representable in this domain., Data-conditioned base hypotheses for these demonstrations.

### Community 49 - "Community 49"
Cohesion: 0.18
Nodes (8): Program, ABC, An explanation with two synchronized faces: executable (``apply``, what…, Canonical text: equal strings mean the same program., Number of non-identity factors (the MDL cost)., Concept region this program belongs to (its kinds of steps, not their values)., Deepest level: compose new programs when no single hypothesis explains the…, Synthesizer

### Community 50 - "Community 50"
Cohesion: 0.20
Nodes (6): Codebook, PerceptionConfig, AlgorithmicSegmenter, GridObject, ndarray, Segmentation

### Community 51 - "Community 51"
Cohesion: 0.27
Nodes (4): ColorCycle, grid_task(), reasoner(), TestReasoner

### Community 52 - "Community 52"
Cohesion: 0.27
Nodes (11): apply(), color_permutation(), _planar(), op(), ndarray, Symmetry operators on 2-D grids and (Z, H, W) volumes. D4 planar dihedral group…, from_volume(), ndarray (+3 more)

### Community 53 - "Community 53"
Cohesion: 0.16
Nodes (8): BackendRejected, cross_validate(), Install `backend` only if it reproduces the numpy reference., A candidate backend disagreed with the numpy reference., Names of the probes where `candidate` disagrees with the reference., set_backend(), RuntimeError, TestBackendGate

### Community 54 - "Community 54"
Cohesion: 0.23
Nodes (3): GridProgram, Any, TestEngine

### Community 55 - "Community 55"
Cohesion: 0.24
Nodes (6): KernelBackend, _logsumexp(), ndarray, Numpy reference. Subclasses must match it within float32 tolerance., Log-domain Sinkhorn-Knopp projection onto the Birkhoff polytope. Double…, TestSinkhorn

### Community 56 - "Community 56"
Cohesion: 0.21
Nodes (6): The independent judge: exact replay + anti-memorization + MDL., VerifyConfig, Scale, Input width must not be mistaken for permission to think. complexity_limit…, The heuristic must still apply where input width tracks program depth., TestComplexityLimit

### Community 57 - "Community 57"
Cohesion: 0.27
Nodes (3): Codebook, PerceptionConfig, PhasorVector

### Community 58 - "Community 58"
Cohesion: 0.27
Nodes (10): most_informative(), Tensor, Curiosity: the most informative question is where the committee disagrees most…, Entropy in bits of K members' exact votes at one input; outputs is (K, ...)., (probe index, bits) maximizing expected information gain over (P, K, ...)…, vote_entropy(), test_q1_expected_information_gain_is_vote_entropy(), Curiosity protocol (Answer or Question) (+2 more)

### Community 60 - "Community 60"
Cohesion: 0.29
Nodes (3): rand(), TestBindUnbind, TestFractionalPower

### Community 61 - "Community 61"
Cohesion: 0.24
Nodes (5): Hashable, ByteLRUCache, ndarray, LRU bounded by bytes, not entry count: at D=4096 one phasor is 32 KB, so a…, test_byte_lru_bounds_bytes()

### Community 62 - "Community 62"
Cohesion: 0.29
Nodes (9): keep_margin(), Tensor, Self-improvement keep gate with a provable false-keep rate (spec §8, G1)., Keep iff mean(candidate - baseline) >= margin at δ / experiments (Bonferroni)., Hoeffding margin sqrt(2 ln(1/δ) / n) for paired differences in [-1, 1]., should_keep(), test_g1_keep_gate_false_keep_rate(), karpathy/autoresearch (+1 more)

### Community 63 - "Community 63"
Cohesion: 0.25
Nodes (8): product_topk(), Tensor, Product-key retrieval: exact top-k over S1 x S2 slots from two top-k searches…, Top-k of scores_a[i] + scores_b[j] over all pairs; returns (values, slot = i *…, test_m1_product_topk_equals_brute_force(), torch, F2 knowledge.pt SSD unified memory, M1 Product-key top-k exactness

### Community 64 - "Community 64"
Cohesion: 0.25
Nodes (3): ndarray, Annealed phase kick sigma_k = sigma0 * gamma^k; every other restart also swaps…, RestartPolicy

### Community 65 - "Community 65"
Cohesion: 0.28
Nodes (5): Snap a noisy vector to its nearest named candidates., bundle(), (Weighted) superposition projected back to the torus. Per-item retrieval SNR ~…, parametrize, TestSuperposition

### Community 66 - "Community 66"
Cohesion: 0.25
Nodes (3): Any, Path, Trace

### Community 68 - "Community 68"
Cohesion: 0.29
Nodes (3): A 90 deg turn about the cube diagonal is not exactly representable., Project would make the gravity family solvable, so it is not a decoy., TestVocabulary

### Community 69 - "Community 69"
Cohesion: 0.29
Nodes (3): The stored benchmark is the only path that goes through JSON. The base Domain…, A generated task must mean the same thing after a JSON round trip., TestJSONRoundTrip

### Community 71 - "Community 71"
Cohesion: 0.60
Nodes (5): core_from_dict(), core_to_dict(), Any, Reasoner, QHead

### Community 72 - "Community 72"
Cohesion: 0.50
Nodes (3): Example, Instruction + one worked example of the exact reply format. The example's…, (parsed proposals, raw reply). A failing backend yields no proposals.

### Community 73 - "Community 73"
Cohesion: 0.50
Nodes (3): Codebook, KnowledgeStore, PriorTable

### Community 74 - "Community 74"
Cohesion: 0.50
Nodes (3): EnergyFn, Example, VerificationReport

## Knowledge Gaps
- **6 isolated node(s):** `hrm`, `HRM (arXiv:2506.21734)`, `TRM (arXiv:2510.04871)`, `RealFormer score passthrough`, `QuatE / RotatE (rotation knowledge-graph embeddings)` (+1 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 476 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **32 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Program` connect `Community 49` to `Community 0`, `Community 2`, `Community 4`, `Community 7`, `Community 11`, `Community 15`, `Community 17`, `Community 19`, `Community 22`, `Community 23`, `Community 28`, `Community 30`, `Community 31`, `Community 36`, `Community 37`, `Community 38`, `Community 39`, `Community 42`, `Community 48`, `Community 54`, `Community 72`, `Community 74`?**
  _High betweenness centrality (0.100) - this node is a cross-community bridge._
- **Why does `KnowledgeStore` connect `Community 5` to `Community 32`, `Community 2`, `Community 41`, `Community 10`, `Community 12`, `Community 19`, `Community 51`, `Community 23`, `Community 25`?**
  _High betweenness centrality (0.069) - this node is a cross-community bridge._
- **Why does `Model` connect `Community 3` to `Community 0`, `Community 2`, `Community 38`, `Community 9`, `Community 43`, `Community 44`, `Community 12`, `Community 46`, `Community 16`, `Community 19`, `Community 21`, `Community 23`?**
  _High betweenness centrality (0.063) - this node is a cross-community bridge._
- **Are the 15 inferred relationships involving `Program` (e.g. with `InnerSpeech` and `literal_explanation()`) actually correct?**
  _`Program` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 25 inferred relationships involving `Model` (e.g. with `_ask()` and `_explain()`) actually correct?**
  _`Model` has 25 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `KnowledgeStore` (e.g. with `KnowledgeConfig` and `SubstrateConfig`) actually correct?**
  _`KnowledgeStore` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `Domain` (e.g. with `InnerSpeech` and `PriorTable`) actually correct?**
  _`Domain` has 6 INFERRED edges - model-reasoned connections that need verification._