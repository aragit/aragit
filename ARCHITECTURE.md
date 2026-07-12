# The Composable Intelligence Stack

Over more than a decade of engineering high-stakes intelligence engines, my focus has shifted from standard deep learning models to engineering complete, deterministic-bounded cognitive workflows.

My work targets the critical boundary where probabilistic neural models intersect with strict symbolic guardrails—specifically in highly constrained domains like clinical oncology, medical triage, energy grid optimization, financial compliance, and real-time enterprise orchestration. By combining state-of-the-art open-source execution engines (`vLLM`, `LangGraph`, `CrewAI`) with high-performance optimization layers (`Unsloth`, custom PEFT), I build systems that do not just predict outcomes, but safely execute and self-correct in production.

### Repository Navigation
This portfolio is organized by **domain and industry** to demonstrate how the same neuro-symbolic architecture principles transfer across verticals:

*   **[Cross-Domain Neuro-Symbolic Architecture](#-cross-domain-neuro-symbolic-architecture):** Foundational reasoning, memory, and orchestration layers that transfer across all verticals.
    * **Edge SLM Optimizer**
    * **Speculative Graph RAG**
    * **DeepSeek Reasoning Fine-Tuning**
    * **Enterprise Intelligence Crew**
*   **[Healthcare & Clinical](#-healthcare--clinical)**
    * **ICU Vitals Transformer(MCP-native tool)**
    * **Autonomous Medication Reconciliation**
    * **Biomedical Entity Extraction Engine**
    * **Autonomous Lab Interpretation & Critical Value Triage Agent**
    * **Clinical Differential Diagnosis Copilot**
    * **Agentic Medicare Authorization**
    * **Edge Fall Detector**
    * **Surgical Vision Copilot**
    * **Spatial Event Detector**
*   **[Marketing & Advertising](#-marketing--advertising):** Competitive Nash equilibrium bidding, real-time intent transformation, and generative ad rendering.
*   **[Supply Chain & Logistics](#-supply-chain--logistics)** Zero-Shot Demand Foundation(MCP Agentic Forecaster Skill),  Autonomous Procurement Swarm
*   **[Energy & Utilities](#-energy--utilities):** Agentic Energy Grid Balancing System
*   **[Computational Biology](#-computational-biology):** Protein Binder Flow,  Quantum-Bound Molecular Generator (QBMG, Zero-Waste Neuro-Symbolic Molecular Engine)
*   **[Finance & RegTech](#-finance--regtech):** Automated KYC & AML Screening Agent, Regulatory Intelligence Agent
*   **[Smart Cities & Urban Systems](#-smart-cities--urban-systems)** Agentic Smart City Traffic Optimization
*   **[Education & Research](#-education--research):** Autonomous Research Synthesizer, Agentic Educational Tutoring Swarm

---

## Architectural Philosophy

This repository documents a **Composable Intelligence Stack**—a living, evolving architecture that transfers across industries.

The frontier of machine learning has moved beyond deploying single, monolithic models. Building reliable, production-grade AI is now an exercise in engineering **Compound AI Systems**. This portfolio showcases the architecture required to make that shift: transforming isolated probabilistic models into closed-loop, multi-agent pipelines that perceive environments, execute deterministic logic, and monitor their own real-world impact.

> ***The future of AI is not about better Brains and Prompts,***
> ***it's about better Systems and Body.***

### Architecture as a Transferable Engine

Every project in this portfolio implements a **Production Grade Complete  Pipeline**, but each makes its **primary architectural contribution** to a specific layers:

| Layer | Function | Cross-Domain Transfer |
|-------|----------|----------------------|
| **Perception** | Transforming unstructured signals into structured intent | Clinical NLP → Manufacturing sensor fusion → Marketing sentiment analysis |
| **Memory** | Temporal state persistence and episodic tracking | Patient care episodes → Customer journeys → Grid load history |
| **Tool Registry** | Deterministic calculators, ontologies, and schema enforcement | Clinical scoring systems → Financial risk models → Grid stability solvers |
| **Reasoning** | Dynamic planning, constraint satisfaction, and DAG compilation | Differential diagnosis → Credit underwriting → Load forecasting |
| **Execution** | Tool dispatch, API actuation, and operational closure | Treatment authorization → Trade execution → Grid frequency control |
| **Governance** | Hard safety guardrails, policy enforcement, and audit | Clinical dosing limits → Brand safety rules → Frequency stability bounds |
| **Meta-Cognition** | Self-monitoring, drift detection, and confidence scoring | Diagnostic confidence → Creative quality scoring → Yield prediction reliability |

The clinical domain was chosen as the first proving ground because it is the most regulated, safety-critical environment imaginable: if a system can safely reason about chemotherapy dosing with deterministic guardrails, the same architectural patterns transfer directly to financial compliance, energy grid stability, and competitive market bidding.

### Evolution: From Domain Proof to Cross-Domain Platform

This portfolio represents a deliberate, phased evolution:

1. **Phase 1 (Clinical Foundation):** Proved the full 7-layer stack in the hardest possible domain—FHIR-native episodic care, temporal workflow compilation, and deterministic safety guardrails for oncology protocols.
2. **Phase 2 (Layer Extraction):** Extracted individual layers into standalone, production-grade systems across marketing (game-theoretic bidding), energy (grid balancing), finance (regulatory compliance), and supply chain (demand forecasting). Each extraction validates that the layer is domain-agnostic.
3. **Phase 3 (Convergence):** Domain-specific implementations feed back into hardened core abstractions. The compliance kernel refines governance patterns for clinical systems; the energy grid balancer refines execution patterns for financial trading; the marketing simulator refines meta-cognitive scoring for scientific research.

The result is a **genuinely transferable decision intelligence platform** with concrete implementations in 8+ industries, each sharing 55–90% core architectural patterns with the clinical reference implementation.

---

## ✨ Cross-Domain Neuro-Symbolic Architecture

> Projects that define the foundational neuro-symbolic stack and transfer across industries.

### [• Edge SLM Optimizer](https://github.com/aragit/edge-slm-optimizer)
**Edge-First Small Language Model Compression & Deployment Pipeline**
> PyTorch, ONNX Runtime Mobile, ExecuTorch, bitsandbytes, llama.cpp, pytest   
> 🟢 `Active` • `Edge AI` • `Model Compression`

**Architecture Insight**

- **Multi-Stage Quantization Pipeline**: FP32 → INT8 (static) → INT4 (dynamic via bitsandbytes/auto-gptq) with perplexity guardrails on WikiText-2
- **Dual Export Targets**: ONNX Runtime Mobile for cross-platform CPU inference; ExecuTorch XNNPACK delegate for ARM NEON optimization
- **Raspberry Pi 5 Benchmarking**: Latency, memory, power draw (INA219 + vcgencmd), thermal throttle detection — all under 5W sustained
- **Speculative Decoding**: 100M-parameter draft model distilled from main 1B model for 2× token generation speedup on edge
- **Telemetry Suite**: Real-time watts-per-token, CPU frequency monitoring, thermal event logging for edge reliability validation
- **Accuracy Preservation**: <15% perplexity degradation vs. FP32 baseline; MMLU subset evaluation for task-specific quality
- **CI/CD Reproducibility**: GitHub Actions with lint, pytest, Docker build — benchmarks versioned per commit

### [• Speculative Clinical GraphRAG (Hybrid Architecture)](https://github.com/aragit/speculative-clinical-graphrag)
**Hybrid Neuro-Symbolic Clinical Knowledge Core with Hybrid RAG and Reasoning-Aware Verification**
> FastAPI, Pydantic v2, LangGraph, Neo4j, Qdrant, vLLM, DeepSeek-R1-Distill-Qwen-32B, SNOMED-CT/ICD-10-CM/RxNorm parsers, pytest    
> 🟢 `Active` • `Neuro-Symbolic Hybrid` • `Clinical Decision Support` • `Hybrid RAG`

<details>
<summary><b>Expand Architecture Insight →</b></summary>

- **9-Node LangGraph Workflow**: INGEST → RETRIEVE_CONTEXT → EXTRACT_SYMPTOMS → MAP_TO_ONTOLOGY → ASSESS_DIFFERENTIAL → VERIFY_SAFETY → [CORRECT_DIFFERENTIAL → ASSESS_DIFFERENTIAL (loop) | SYNTHESIZE | ESCALATE]; cyclic correction with max 3 iterations, recursion limit 20, full audit trace
- **Hybrid RAG Stack**: Qdrant vector store (384-d sentence-transformer embeddings over ontology concepts) + Neo4j graph traversal (taxonomic relationships, fallback to in-memory EDGES) + symbolic Cypher validation (existence proofs for every proposed edge) + fusion scoring (α=0.7)
- **Medical Ontology Support**: ETL pipeline parsers for SNOMED-CT RF2, ICD-10-CM text, RxNorm RRF formats; 178 in-memory ontology triples (126 unique clinical concepts) ship with repo; real data requires licensed SNOMED-CT/UMLS files
- **Quad-Track LLM Backend**:
  - **MockLLM**: Deterministic keyword lookup for CI/testing (zero-dep, 20 categories, 65 triplets)
  - **Ollama**: Local CPU inference (gemma2:2b, JSON-structured generation) for development
  - **DeepSeekR1Backend**: OpenAI-compatible, extracts `<think>` reasoning traces, bounded extract_symptoms/assess_differential
  - **VLLMBackend**: Any OpenAI-compatible server, same bounded subroutine contract
  - **SemanticRouter**: Classifies patient notes to select optimal backend automatically
- **DeepSeek-R1 Reasoning Integration**: Extracts Chain-of-Thought reasoning traces from `<think>` tags via `OpenAICompatBackend.generate_path()`, validates diagnostic logic against medical ontologies before surface generation, surfaces reasoning steps in API response for clinician review
- **Self-Correcting Feedback with Reasoning Awareness**: On validation failure, violations + prior reasoning are fed back via `regenerate_with_feedback()`; confidence decay (-0.1 per correction) with `validate_reasoning_coherence()` check; violations from all 3 verifiers (Neo4j, SymbolicVerifier, OPA) included in correction prompt
- **Deterministic Escalation Guardrail**: Unvalidated paths after max iterations always route to human review with full reasoning trace, proposed path, and violation log — never to patient-facing output; zero PHI persistence (in-memory only, no DB writes of patient data)
- **FastAPI Production Gateway**: `/v1/speculate` principal endpoint, `/v1/reasoning_trace/{trace_id}` for clinician review, `/health` with Neo4j/Qdrant/OPA/Redis probes, asynccontextmanager lifespan with startup ontology seeding; RequestID/APIKey/RateLimit middleware
- **Docker Compose Production Stack**: vLLM container (GPU profile), Neo4j Community (ontology graph), Qdrant (vector store), FastAPI orchestrator, OPA governance sidecar, Redis (idempotency/session), Jaeger (tracing profile)
- **Comprehensive Test Suite**: 53 tests (4 skipped without Docker: Neo4j×2, OPA×1, Ollama×1): valid path (1 iteration), invalid→escalate after 3 correction attempts, nonsensical input escalation, reasoning trace presence, all 4 backends + semantic router, hybrid retrieval with fusion scoring, ontology ETL not-found paths, symbolic drug interaction detection, API middleware (auth/rate-limit/request-id), full pipeline via FastAPI TestClient

</details>

### [• Post-RAG Drift Evaluator](https://github.com/aragit/post-rag-drift-evaluator)
**Automated Latent Space Drift Telemetry & Comparative RAG Architecture Benchmark**
> Python 3.12, LiteLLM, Polars, pgvector, scikit-learn, SciPy, Streamlit, Docker, pytest, ruff, mypy
> 🟢 `Active` • `Embedding Drift Telemetry` • `Comparative RAG Evaluation` • `Statistical MLOps`

<details>
<summary><b>Expand Architecture Insight →</b></summary>

- **Multidimensional Drift Pipeline**: Projects 1536-dimensional embedding vectors using Principal Component Analysis (PCA) to isolate primary variance coordinates; fits non-parametric continuous distributions using Gaussian Kernel Density Estimation (KDE) to calculate population-level Jensen-Shannon Divergence ($D_{JS}$) bounded strictly between $0 \le D_{JS} \le 1$.
- **Dual-Configuration Duality**: Features a zero-cost local fallback engine that applies additive noise ($\sigma=0.10$) to mimic real semantic distributions, shifting dynamically to live, non-blocking asynchronous inference paths through `litellm` when validated provider keys are present in the runtime lifecycle.
- **Native Vector Infrastructure**: Executes raw `<=>` cosine distance operations directly against containerized PostgreSQL and `pgvector` backends; profiles live performance metrics across parallel processing branches to analyze structural trade-offs between Naive RAG and multi-hop Agentic RAG state machines.
- **Deterministic Quality Judges**: Implements automated context precision and answer faithfulness evaluation layers utilizing structured `json_object` configurations, enforcing a strict fallback penalty of `0.0` on any validation or parsing anomaly to eliminate silent scoring failures.
- **Spatial Telemetry Observability**: Renders a dedicated Streamlit metrics interface graphing real-time 2D PCA coordinate transformations to isolate live production query distribution shifts from historical data manifolds.
- **Rigorous Production Quality Gates**: Enforces a multi-stage Docker build separating dependency compilation from the final runtime container; backed by GitHub Actions workflows driving automated execution runs via `pytest`, strict static type audits via `mypy`, and syntax validation via `ruff`.

</details>

### [• DeepSeek Reasoning Fine-Tuning](https://github.com/aragit/deepseek-reasoning-finetuning)
**Medical chain-of-thought LoRA alignment pipeline**
> Unsloth, PyTorch, Hugging Face, TRL   
> 🟢 `ACTIVE` • `REASONING OPTIMIZATION LAYER`

<details>
<summary><b>Expand Architecture Insight →</b></summary>

**Architecture insight**
- Efficient 4-bit parameter fine-tuning for reasoning behavior
- Maps diagnostic reasoning patterns into model weights
- Improves structured clinical response generation

</details>

### [• Enterprise Intelligence Crew](https://github.com/aragit/enterprise-intelligence-crew/tree/main)
**Autonomous enterprise trend intelligence pipeline**
> CrewAI, Ollama, FastAPI, ChromaDB, Pydantic V2   
> 🟢 `Active` • `Local-First` • `3-Agent Sequential Pipeline`

<details>
<summary><b>Expand Architecture Insight →</b></summary>

- **Sequential 3-agent pipeline**: Trend Investigator → Risk Analyst → Copywriter
- **LangGraph risk gate**: State-machine guardrail (`analyze → evaluate → approve|reject`) with circuit-breaker
- **Local-only LLM inference**: Native Ollama `/api/chat` adapter — zero API keys, zero cloud dependency
- **ChromaDB semantic memory**: Sentence-transformer embeddings for research persistence & recall
- **Enforced Pydantic V2 contracts**: `TrendPayload`, `RiskPayload`, `ContentPayload` validated at every stage
- **FastAPI + Prometheus**: Sync/async `/crew/run` endpoints, async polling, health checks, and metrics scraping

</details>

---

## 🏥 Healthcare & Clinical

### [• Clinical Triage Agentic Orchestrator](https://github.com/aragit/clinical-triage-agentic-orchestrator)
**Neuro-Symbolic Agentic Orchestrator for High-Stakes Clinical Triage with OPA Guardrails**
> FastAPI, llama-cpp-python, Gemma 3n E4B, Qdrant (Hybrid BM25+Dense+RRF), SNOMED-CT/ICD-10-CM, Pydantic v2, Streamlit, Docker Compose, pytest
> 🟢 Active • Neuro-Symbolic • Clinical Decision Support • Edge-First

<details>
<summary><b>Expand Architecture Insight →</b></summary>
   
- Multi-Step Agentic Pipeline: Perception (episodic history retrieval) → OPA Guardrails (deterministic emergency bypass) → Memory (Qdrant hybrid guideline lookup) → Executor (SNOMED/ICD-10 entity extraction) → Cognition (llama-cpp + Gemma 3n instructor-forced JSON) → Action (FSM state transition); LLM is ONLY invoked AFTER passing guardrails + context enrichment — the neuro-symbolic boundary
- Dual-Pathway Execution Pattern: Fast-Path (emergency bypass) short-circuits the entire LLM tier — OPA detects life-threatening patterns, extracts clinical codes, transitions FSM to escalation, returns in 24.4ms with zero model hallucination risk; Slow-Path (cognitive loop) runs full 6-step pipeline for non-emergent cases (~1.2-2.5s)
- OPA-Style Policy Engine (opa_policies.py): 3-rule deterministic evaluation chain — (1) Emergency detection: 30+ regex patterns across cardiac, respiratory, psychiatric, neurological, hemorrhagic, toxicological, airway categories → instant ROUTE_TO_EMERGENCY with llm_bypassed: true; (2) Escalation detection: 6 patterns (obstetric, pediatric, medication, allergy) → ESCALATE_TO_HUMAN with LLM processing but human-review flag; (3) Content safety: minimum-length gate → ALLOW_TRIAGE or DENY
- Hybrid Vector Store (vector_store.py): Qdrant in-memory backend with dual retrieval — dense semantic search (384-dim pseudo-embeddings, cosine distance) + sparse BM25 keyword search (Okapi BM25, k1=1.5, b=0.75) — fused via Reciprocal Rank Fusion (k=60) for clinical guideline lookup; 5 seed guidelines (chest pain, stroke, asthma, diabetic emergency, anaphylaxis) pre-loaded at startup
- Atomic FSM State Machine (episodic_state.py): 7 clinical state nodes (intake → symptom_extraction → guideline_lookup → risk_assessment → triage_decision → escalation → resolved) with strict valid-transition guard; StateTransitionError prevents illegal state mutations; TTL-based session expiry (7200s) replaces Redis for local deployment
- Clinical NLP Entity Extraction (healthcare_nl.py): 40+ curated SNOMED CT + ICD-10-CM terminology entries covering cardiac, respiratory, neurological, gastrointestinal, endocrine, musculoskeletal, psychiatric, immunological, infectious, and hematological systems; regex-based extraction with severity escalation detection (5 severity patterns); deterministic output with confidence: 1.0
- DiagnosticCoT Schema (triage_agent.py): Instructor-wrapped local LLM forced into strict Pydantic output — clinical_observations (array), step_by_step_rationale (array), urgency_level (emergent/urgent/semi-urgent/non-urgent/deferrable), next_state_action (maps to FSM), extracted_symptoms (array), recommended_department (ER/urgent_care/primary_care/telehealth/self_care), confidence (0-1); fail-safe: non-JSON or LLM failure always over-triages to ER — never under-triages
- FastAPI Production Gateway: /webhook/fulfillment principal endpoint, /health with LLM reachability + guideline count probes; asynccontextmanager lifespan with startup subsystem initialization (episodic store, vector store, guardrail, extractor, LLM client, triage agent) + guideline seeding; CORS middleware, structured logging
- Docker Compose Production Stack: 4-service architecture — llama-cpp-server (CPU-native GGUF inference, port 8000, 4G memory limit, health check via /v1/models), redis (session cache, port 6379), orchestrator-api (FastAPI, port 8080, depends on LLM + Redis healthy), streamlit-ui (observability dashboard, port 8501, depends on API healthy); all services have health checks with start periods
- Streamlit Observability Dashboard: Dual-column layout showing real-time FSM state tracking, ontology extraction matrix (SNOMED/ICD-10 codes), conversation history, and pipeline latency metrics; chat interface for clinical input with live triage feedback
- Production Verification: Emergency scenario "severe chest pain and difficulty breathing" → OPA triggers cardiac-emergency + respiratory-emergency → FSM transitions intake → escalation → extracts SNOMED:29857009 (Chest Pain) + ICD-10:R07.9 → LLM bypassed → 24.4ms total pipeline latency (vs 90+ seconds for raw CPU LLM generation)
  
</details>

---

### [• ICU Vitals Transformer (MCP-native tool)](https://github.com/aragit/icu-vitals-transformer)
**MCP Clinical Forecasting Skill**     
> 🟢 `Active` • `MCP Tool` • `Clinical Temporal Monitoring`

**Architecture Insight**

- MCP-native tool — exposes ingest_vitals, get_forecast, get_deterioration_index via Model Context Protocol
- Deterministic forecasting — multi-horizon trend extrapolation (1h/4h/12h) with clinical uncertainty bounds, no GPU required
- FHIR R4 ingestion — parses LOINC-coded Observation resources into sliding 5-minute windows
- NEWS2-inspired governance — deterministic deterioration index + severity classification (NORMAL → WARNING → ALERT → EMERGENCY)
- Stateless by design — caller decides action; tool returns structured predictions only


### [• Autonomous Medication Reconciliation](https://github.com/aragit/medication-reconciliation-agent)
**Cross-Source Medication Safety Engine**  
> Gemma3 / Qwen2.5, MCP, FHIR R4, RxNorm, DrugBank, FastAPI, Pydantic, Neo4j, pytest — CI/CD        
> 🟡 `Coming Soon` • `Medication Safety` • `Neuro-Symbolic AI` • `Dynamic Tool Use`

<details>
<summary><b>Expand Architecture Insight →</b></summary>

- **Hybrid Framework:** Neural primary controller ingests medication lists from fragmented sources (EHR medication lists, pharmacy records, discharge summaries, patient-reported histories), normalizes free-text drug names to RxNorm concepts via local LLM inference, and dynamically invokes symbolic safety tools — with deterministic validation at the output boundary.
- **Cross-Source Discrepancy Detection:** The LLM autonomously identifies duplicates (same drug, different names), omissions (chronic medication missing from one source), and temporality conflicts (discontinued drug still active in another system) — no pre-encoded matching rules.
- **Dynamic Tool Orchestration:** MCP-native tool registry exposes 8+ clinical APIs (RxNorm resolver, drug interaction checker, allergy cross-reference, therapeutic duplication detector, renal dose adjuster, pregnancy category checker, lab value interpreter for dose validation, temporal logic engine for washout periods). The LLM decides *which* tools, *when*, and *in what order* — not a fixed pipeline.
- **Epistemic Confidence Scoring:** Each reconciliation step is tagged with uncertainty metadata. The LLM performs meta-reasoning over source reliability (EHR > pharmacy > patient-reported) and confidence scores to flag items requiring pharmacist verification.
- **Symbolic Safety Boundary:** Final reconciled medication list passes through a deterministic verifier ensuring no severe drug-drug interactions (Class X), no allergy conflicts, no therapeutic duplications, and dose limits within renal/hepatic function — all blocked from reaching the patient record without explicit pharmacist override and full audit trail.

</details>

### [• Biomedical Entity Extraction Engine](https://github.com/aragit/bionlp-llama3-service)
**FastAPI microservice for biomedical NER via 4-bit quantized LLaMA-3 with deterministic structured output**
> LLaMA-3 8B, Unsloth, FastAPI, Pydantic v2, LoRA, Triton    
> 🟢 `ACTIVE` • `Dual runtime (local / gpu)` • `Five entity types: DNA, RNA, protein, cell_type, cell_line`

<details>
<summary><b>Expand Architecture Insight →</b></summary>

**Architecture insight**
- Decoupled ingestion-inference architecture isolates the FastAPI schema layer from the Unsloth/Triton compute chassis, enabling independent scaling of API and model execution
- Environment-aware engine factory switches between MockEngine (local/CI validation) and TritonEngine (4-bit quantized GPU inference) via `RUNTIME_ENV` injection
- Structured output pipeline forces Alpaca-formatted generation into deterministic tuples, bridged through terminal-delimiter truncation and Pydantic contract validation

</details>

<details>
<summary><b>EXPAND MORE HEALTHCARE SOLUTIONS →</b></summary>

### [• Speculative Clinical Graph RAG, Hybrid Architecture](https://github.com/aragit/speculative-clinical-graphrag)
**Hybrid Neuro-Symbolic Clinical Knowledge Core with Hybrid RAG and Reasoning-Aware Verification**
> FastAPI, Pydantic v2, LangGraph, Neo4j, LlamaIndex, vLLM, DeepSeek-R1-Distill-Qwen-32B, SNOMED-CT, ICD-10-CM, RxNorm, pytest    
> 🟢 `Active` • `Neuro-Symbolic Hybrid` • `Clinical Decision Support` • `Hybrid RAG`

<details>
<summary><b><i>Architecture Insight ...</i></b></summary>

- **Six-State LangGraph Workflow**: INGEST → SPECULATE → RETRIEVE → VERIFY → [VALIDATE|CORRECT|ESCALATE] → SYNTHESIZE → END; cyclic correction with max 3 iterations, recursion limit 10, full audit trace
- **Hybrid RAG Stack**: LlamaIndex vector store (dense embeddings over SNOMED-CT/ICD-10 concepts) + Neo4j graph traversal (taxonomic relationships) + symbolic Cypher validation (existence proofs for every proposed edge)
- **Real Medical Ontologies**: SNOMED-CT US Edition 2024 (clinical findings, disorders, procedures), ICD-10-CM 2024 (diagnosis classification), RxNorm (drug names, ingredients, dose forms), UMLS Metathesaurus 2024AB (cross-vocabulary mapping) — ingested via automated ETL pipeline
- **Triple-Track LLM Backend**:
  - **MockLLM**: Deterministic keyword lookup for CI/testing (zero-dep, instant)
  - **Ollama**: Local CPU inference (gemma2:2b, JSON-structured generation) for development
  - **vLLM + DeepSeek-R1-Distill-Qwen-32B**: Production GPU inference with tensor-parallelism, OpenAI-compatible API, structured reasoning trace extraction
- **DeepSeek-R1 Reasoning Integration**: Extracts Chain-of-Thought reasoning traces from R1's `<think>` tags, validates diagnostic logic against medical ontologies before surface generation, surfaces reasoning steps in API response for clinician review
- **Self-Correcting Feedback with Reasoning Awareness**: On validation failure, violations + reasoning trace mismatches are fed back to R1 with correction prompt; confidence decay (-0.1 per correction) with reasoning coherence check
- **Deterministic Escalation Guardrail**: Unvalidated paths after max iterations always route to human review with full reasoning trace, proposed path, and violation log — never to patient-facing output; zero PHI persistence
- **FastAPI Production Gateway**: `/v1/speculate` principal endpoint, `/v1/reasoning_trace` for clinician review, `/health` with dependency probes, startup ontology seeding, graceful shutdown
- **Docker Compose Production Stack**: vLLM container (GPU, tensor-parallel), Neo4j Community (ontology graph), LlamaIndex vector store (Qdrant/Pinecone), FastAPI orchestrator, OPA governance sidecar
- **Comprehensive Test Suite**: Valid path (1 iteration), invalid-then-corrected (≤3 iterations), escalation after max iterations, reasoning trace extraction, ontology ETL validation, hybrid retrieval accuracy

</details>

### [• Autonomous Lab Interpretation & Critical Value Triage Agent](https://github.com/aragit/lab-interpretation-triage-agent)
**Context-Aware Laboratory Intelligence Engine**
> Ollama (gemma3:1b / qwen2.5:0.5b), MCP, FHIR R4, HL7 v2, LOINC, FastAPI, Pydantic, SQLite, pytest — CI/CD   
> <span style="color:#8B0000">⬤</span> `Private` • `Clinical Laboratory` • `Neuro-Symbolic AI` • `Critical Value Management`

### [• Clinical Differential Diagnosis Copilot](https://github.com/aragit/clinical-differential-copilot)
**Autonomous Clinical Reasoning Engine**
> Claude 4.5 Sonnet, MCP, FHIR R4, SNOMED-CT, Clinical Calculators, LangSmith - CI/CD    
> <span style="color:#8B0000">⬤</span> `Private` • `Clinical Decision Support` • `Neuro-Symbolic AI` • `Dynamic Tool Use`

### [• Edge Fall Detector](https://github.com/aragit/edge-fall-detector)
**Real-time patient fall detection on edge devices**
> `YOLOv11-Pose` `TensorRT` `MQTT` `OpenCV`     
> 🟢 `ACTIVE` • `EDGE SAFETY SYSTEM`

### [• Surgical Vision Copilot](https://github.com/aragit/surgical-vision-copilot)
**Real-time surgical understanding with vision-language models**
> `Video-LLaVA` `OpenCV` `Temporal` `Action Modeling`   
> 🟢 `ACTIVE` • `VISION PERCEPTION SYSTEM`

### [• Spatial Event Detector](https://github.com/aragit/spatial-event-detector)
**Kinematic telemetry → structured motion event extraction**
> `YOLOv11-Pose` `OpenCV` `NumPy`    
> 🟢 `ACTIVE` • `MOTION PERCEPTION SYSTEM`

</details>

<br>

---

## 🎯 Marketing & Advertising

### [• Nash Marketing Agents (Game Theory)](https://github.com/aragit/agentic-nash-marketing)
**Neuro-Symbolic Multi-Agent Ad Auction Simulator with Nash Equilibrium Solver**
> FastAPI, Pydantic v2, SQLAlchemy 2.0, SciPy, SQLite/PostgreSQL, Docker, pytest   
> 🟢 `Active` • `Neuro-Symbolic` • `Game Theory` • `Ad Tech Simulation`

**Architecture Insight**

- **Neuro-Symbolic Hybrid**: LLM engine proposes stochastic bidding strategies; symbolic Nash solver validates equilibrium via iterative best-response with softmax annealing
- **VCG Second-Price Auction Engine**: Winners pay the next-highest bid; mathematical asserts enforce `paid ≤ bid` invariant on every round
- **Multi-Layer Budget Guardrails**: Soft warning (20%), hard cap (10%), emergency mode (5%) prevent catastrophic depletion in competitive bidding wars
- **Monte Carlo Nash Solver**: 5,000-sample stochastic win-probability estimation; converges to mixed-strategy equilibrium where no agent can improve utility unilaterally
- **49-Test Suite with Property-Based Verification**: Monotonicity, individual rationality, Nash convergence bounds, VCG payment correctness, guardrail action validity
- **Interactive Chart.js Dashboard**: Real-time simulation config, visualization panels, live event log

### [• Real-Time Intent Transformer](https://github.com/aragit/real-time-intent-transformer)
**Real-Time E-Commerce Intent Classification with Action Governance**
> FastAPI, Pydantic v2, Polars, scikit-learn, aiokafka, SQLite, OPA, pytest (84+ tests)      
> 🟢 `Active` • `E-Commerce Personalization` • `Behavioral Analytics`

<details>
<summary><b>Expand Architecture Insight →</b></summary>

- **7-Layer Neuro-Symbolic Pipeline**: Ingestion → Perception (Polars feature engineering) → Reasoning (rule-based + ML ensemble) → Governance (OPA/Rego + Python fallback) → Execution (action dispatch with suppression) → Memory (SQLite session store) → Meta-Cognition (drift detection stub)
- **18 Behavioral Features**: Session duration, page views, cart adds/removes, checkouts, searches, cart value, category switches, exploration ratio, conversion rates, inter-event time — engineered via Polars DataFrame operations
- **Dual-Track Intent Classification**: Rule-based heuristic (<10ms, confidence ≥0.6 threshold) falls back to sklearn RandomForest/XGBoost ensemble (<50ms) if model file exists; 7 intent classes (BROWSE, COMPARE, CART_BUILDER, CHECKOUT_INTENT, PRICE_SENSITIVE, CHURN_RISK, LOYAL_RETURNER)
- **Six Action Types with Suppression**: RECOMMEND_ALTERNATIVE, SHOW_COMPARISON_TOOL, APPLY_DISCOUNT, SHOW_URGENCY, SEND_ABANDON_EMAIL, LOYALITY_REWARD — 15-minute deduplication window per session, governance deny override
- **Business Rules Governance**: Anti-gaming (max 3 discounts/month, 24h cooldown), minimum cart value ($50), inventory threshold for urgency, session duration for abandon email, demographic fairness guardrail (no pricing discrimination)
- **Best-Effort Kafka Streaming**: aiokafka producer with start/stop lifecycle; SQLite is primary persistence, Kafka is fire-and-forget fallback when unavailable
- **84+ Tests with 70%+ Coverage**: Intent classification, feature engineering, action dispatch, governance rules, event store, API endpoints
- **Docker Compose Infrastructure**: Zookeeper + Kafka + OPA (Open Policy Agent) with healthchecks
- **Synthetic Data Generation**: 34,148 events across 5,000 sessions with balanced intent distribution; perfect F1 on synthetic data (expected — labels derived from rules)

</details>

### [• Generative Dynamic Ad Renderer](mailto:anicomanesh@gmail.com?subject=Access%20Request%3A%20generative-dynamic-ad-renderer)
**Telemetry-driven ad generation pipeline**
> LLM generation, behavioral signals, rendering automation          
> <span style="color:#8B0000">⬤</span> `Private` • `Media Execution`

<details>
<summary><b>Expand Architecture Insight →</b></summary>

- Converts user behavioral signals into generated creative content
- Connects inference pipelines directly to media rendering systems
- Enables real-time adaptive advertising generation
- Designed for continuous personalization loops

</details>

<br>

---

## 📦 Supply Chain & Logistics

### [• Zero-Shot Demand Foundation (MCP Agentic Forecaster Skill)](https://github.com/aragit/zero-shot-demand-foundation)
**Zero-Shot Time-Series Demand Forecasting with Foundation Models**
> PyTorch, TimesFM, Transformers, Amazon Chronos-2, Pydantic v2, PyYAML    
> 🟢 `Active` • `Zero-Shot Forecasting` • `Retail Demand Prediction`

**Architecture Insight**

- **Foundation Models**: google/timesfm-2.5-200m-pytorch, Amazon Chronos-2 (`amazon/chronos-2`) via `BaseChronosPipeline` — zero-shot inference
- **Dual-Track Evaluation**: Point forecast (Accuracy Track) + quantile/sample trajectory parsing (Uncertainty Track) aligned with M5 Competition framework
- **3D Tensor Integration**: Strict `(n_series, n_variates, history_length)` input format; shape-agnostic output parser handles 3D point forecasts and 4D sample/quantile tensors via median extraction
- **Pydantic Input Validation**: `TimeSeriesInputPayload` enforces context bounds [16, 16,000] timesteps, horizon [1, 1024], and exogenous array alignment (price_index, promo_flag must match `context + horizon` length)
- **Pydantic Output Validation**: `ForecastOutputPayload` enforces mean prediction dimension match, optional p10/p90 quantile bands, model identifier tracking
- **M5 Competition Benchmarking**: Evaluates against Walmart daily sales (3,049 products, 10 stores, 3 states) with WAPE and RMSSE metrics; 128-step backtest window with active high-volume item filtering
- **Corporación Favorita Compatibility**: Secondary validation on Ecuadorian retail data with inflation markers and regional holidays for cross-locale zero-shot generalization testing
- **Exogenous Signal Support**: Optional price elasticity (`price_index`) and binary promotional event flags (`promo_flag`) aligned chronologically with target + horizon

### [• Autonomous Procurement Swarm](https://github.com/aragit/autonomous-procurement-swarm)
**LLM-Powered Multi-Agent Contract Negotiation for Supply Chain Optimization**
> FastAPI, Pydantic v2, SciPy, Transformers, Matplotlib, pytest   
> 🟢 `Active` • `Turn-Based Negotiation` • `Market Simulation` 

<details>
<summary><b>Expand Architecture Insight →</b></summary>

- **4 Agents**: Buyer, Seller, Market Intelligence, and Arbiter — each with role-specific LLM system prompts and structured JSON output
- **3 LLM Backends**: MockLLM (deterministic, instant, default), HuggingFace Transformers (CPU, ~2-6GB download), optional vLLM (CPU batched inference, manual build required)
- **Stochastic Market Simulator**: Geometric Brownian Motion price dynamics with regime-switching drift/volatility, 4-state Markov chain geopolitical risk model (LOW→MEDIUM→HIGH→CRISIS), Poisson supply shocks with log-normal magnitude
- **Reward Engineering**: Buyer reward = negative normalized total cost (purchase + risk premium + logistics + stockout penalty + spot-price bonus); Seller reward = normalized margin + capacity utilization bonus
- **Centralized Episode Orchestrator**: `NegotiationEpisode` manages alternating buyer/seller turns, market context injection, arbiter validation, ledger logging, and terminal condition detection (ACCEPT/REJECT/timeout)
- **Test Coverage**: 3 test modules covering ledger hash-chain integrity, market GBM/shock dynamics, and protocol message validation/terminal detection

</details>

<br>

---

## ⚡ Energy & Utilities

### [• Agentic Energy Grid Balancing System](https://github.com/aragit/agentic-energy-grid-balancer)
**Neuro-Symbolic Multi-Agent Energy Market Simulator**
> FastAPI, Pydantic v2, SQLAlchemy 2.0, Docker, CI/CD, pytest (~120 tests), black, flake8    
> 🟢 `Active` • `Energy Market Simulation` 

**Architecture Insight**

- **Symbolic First Architecture**: Symbolic `GridSimulation._run_step()` owns the hour-by-hour execution loop; neural LLM (ReasoningEngine rule-based or Ollama local) is a bounded, swappable subroutine for battery arbitrage only
- **6 Agents**: SolarFarm, WindFarm, CoalPlant (ramp-limited, 820 gCO₂/kWh), NuclearPlant (must-run, 5% ramp), GridBattery (LLM-driven), MetroCity (price-elastic demand curve)
- **Pydantic Neural Boundary**: `BidStrategy` schema validates LLM JSON output with `bid_price ∈ [1.0, 200.0]`, canonical action normalization (9 valid actions), confidence ∈ [0.0, 1.0], non-empty reasoning — plus `ValidatedBid` for post-guardrail execution contract
- **Battery SoC Guardrails**: LLM decision honored at 15%–85% SoC; forced charge below 5%, forced discharge above 95%, hold override at 15%/85% boundaries — symbolic clamps, not replacement
- **Double-Sided Auction Engine**: Continuous matching at midpoint prices, carbon cost per trade ($25/ton, coal only), price clamped to [25, 120] USD/MWh, buyer/seller surplus computation
- **Grid Physics**: Seasonal sinusoidal + Perlin noise weather (irradiance, wind speed, temperature, storm probability), piecewise demand model (hour-of-day + temperature + price elasticity), damped frequency model with inertia constant (clamped 47–53 Hz, stability window 49.5–50.5 Hz)
- **Regulatory Oversight**: Frequency violation logging (±1 Hz bounds), per-agent carbon cap (50,000 kg), market manipulation detection (identical bid detection)
- **Agent Memory**: Episodic `Experience` recording (price, profit, weather, decision, outcome), pattern recognition (best price range, storm frequency, peak demand hours), strategy advice generation
- **CI/CD Pipeline**: GitHub Actions with 3 jobs — pytest with coverage, black + flake8 linting, Docker build + health check + 5 endpoint smoke tests
- **~120 Tests Across 9 Modules**: Grid physics (17), auction (14), agents (9), API (14), simulation (12), orchestrator stabilization (8), bid validation (15), battery guardrails (12), Pydantic boundary (~18)

<br>

---

## 🧬 Computational Biology & Chemistry

### [• Protein Binder Flow](https://github.com/aragit/Flow-Matching-Protein-Binder-Generator)
**Flow-matching protein binder generator**
> PyTorch, Biopython, Flow Matching, FoldSeek   
> 🟢 `Active` • 🧬 `Computational Biology Research`

<details>
<summary><b><i>Architecture Insight ...</i></b></summary>

- Uses flow matching for structural molecular generation
- Moves beyond diffusion-based protein design approaches
- Targets novel protein–ligand binding discovery
- Expands AI systems into generative bio-molecular design

</details>

### [• Quantum-Bound Molecular Generator (QBMG)](https://github.com/aragit/quantum-bound-generator/tree/main)
**Zero-Waste Neuro-Symbolic Molecular Engine**
> 100% physically valid generation • Differentiable convex constraints • IFT gradient propagation • Zero compute waste   
> 🟢 `Active` • `Generative Chemistry`

**Architecture Insight**

- **Zero-waste generation** — every forward pass outputs a chemically valid bond adjacency matrix; no post-generation filtering or discard pipelines required
- **Implicit Function Theorem (IFT) backprop** — analytical Jacobian computation through the KKT equilibrium bypasses solver unrolling, enabling end-to-end gradient flow without memory explosion
- **Single substrate design** — neural backbone (SE(3)-GNN / transformer) and physics engine exist on the same mathematical substrate; no API boundaries, no JSON orchestration
- **Hard valency enforcement** — convex optimization boundary strictly caps per-atom bond sums (e.g., C≤4, O≤2) while minimizing Frobenius distortion from the neural guess
- **Modular backbone** — hot-swappable generators: dense MLPs, E(n)-Equivariant GNNs, or text-conditioned projections (e.g., MedGemma-4B-IT) all feed into the same physics core
- **Composable** — designed as a structural generative engine for Type 2 and Type 6 clinical intelligence pipelines, including multi-drug optimization and materials discovery

---

## 🏦 Finance & RegTech

### [• Regulatory Intelligence Agent](https://github.com/aragit/regulatory-intelligence-agent)
**Autonomous Compliance Monitoring Engine**
> Claude Opus 4.6, MCP, Neo4j Policy Graph, QuantLib, SEC EDGAR API, OpenTelemetry - CI/CD    
> <span style="color:#8B0000">⬤</span> `Private` • `FinTech / RegTech` • `Neuro-Symbolic AI` • `Dynamic Tool Use`

<details>
<summary><b>Expand Architecture Insight →</b></summary>

- **Hybrid Framework:** Neural primary controller monitors regulatory landscapes, dynamically orchestrating symbolic tools for impact quantification, policy tracing, and stakeholder notification — reasoning across jurisdictions and business lines.
- **Cross-Domain Regulatory Reasoning:** The LLM reads unstructured regulatory text (SEC, FINRA, ECB, FCA), identifies affected internal policies via Neo4j knowledge graph traversal, and decides which risk models require re-validation — interpreting intent beyond keyword matching.
- **Dynamic Impact Quantification:** Monte Carlo simulation (QuantLib) runs only when the LLM determines quantitative impact is material. The LLM decides simulation parameters, interprets tail-risk outputs, and decides whether to escalate to human risk officers.
- **Multi-Channel Orchestration:** The LLM decides notification strategy — which trading desks (Slack), which compliance officers (email), which legal teams (Jira) — based on policy graph analysis of organizational ownership and historical response patterns.
- **Symbolic Audit Boundary:** Every regulatory text → tool call → output decision is fully traced (OpenTelemetry). Deterministic policy verifier ensures no recommendation violates hard constraints (capital requirements, position limits, blackout periods). Immutable audit trail for regulator examination.

</details>

### [• KYC-Auto (Know Your Customer)](https://github.com/aragit/kyc-auto)
**Automated KYC & AML Screening Agent**
> Qwen2.5-7B-Instruct, LangChain, OpenSanctions, Neo4j UBO Graph, PostgreSQL, Redis, FastAPI, OpenTelemetry — CPU-First / vLLM-Ready    
> 🟢 `Active` • `FinTech / RegTech` • `SLM-First Agent` • `Deterministic Risk Scoring`

<details>
<summary><b>Expand Architecture Insight →</b></summary>

- **SLM-First ReAct Agent:** Single-threaded LangChain ReAct loop runs entirely on CPU using 4-bit quantized Qwen2.5-7B-Instruct (GGUF via llama-cpp-python). No dependency on Claude or GPT-4o. Architected with a pluggable `BaseLLMBackend` abstraction so the agent swaps to vLLM GPU inference (Qwen-14B/70B, Mixtral-8x7B) via one-line config change when hardware is available.
- **Deterministic Tool Orchestration:** The agent plans and executes exactly one tool per ReAct turn — `pep_screen`, `sanctions_check`, `adverse_media_search`, `ubo_extract`, `risk_score_combine`. The final `risk_score_combine` tool is a pure deterministic algorithm (no LLM), ensuring the overall risk rating is reproducible and regulator-auditable.
- **Document-to-Graph UBO Extraction:** Corporate documents (PDF, CSV) are chunked into 500-token windows and parsed by the SLM with structured JSON prompts. Extracted beneficial owners are deduplicated via fuzzy matching (`rapidfuzz`) and written to a Neo4j graph as `(:Person)-[:OWNS]->(:CorporateBody)` relationships for network analysis.
- **Offline-Resilient Screening:** OpenSanctions API responses are cached in Redis (24h TTL) and backed by a local SQLite mirror of OFAC SDN CSV. If external APIs fail, the agent degrades gracefully to offline deterministic screening with full trace logging.
- **Structured Audit Boundary:** Every tool call, LLM reasoning step, and final `KYCPacket` output is validated by Pydantic v2 and persisted to PostgreSQL with immutable JSON audit trails. OpenTelemetry spans trace the full ReAct loop for regulator examination. FastAPI exposes `POST /screen` and `GET /case/{id}` for synchronous onboarding platform integration.

</details>

<br>

---

## 🏙️ Smart Cities & Urban Systems

### [• Agentic Smart City Traffic Optimization](#)
**Multi-agent traffic signal and routing optimizer**
> Graph networks, city simulation, intersection agents, routing optimization    
> 🔵 `Concept` • `Urban Coordination`

<details>
<summary><b>Expand Architecture Insight →</b></summary>

- Intersection agents negotiate real-time signal timing
- Transit agents optimize passenger flow across networks
- Emergency agents override routing for critical response vehicles
- Global orchestrator resolves congestion and systemic deadlocks

</details>

<br>

---

## 🎓 Education & Research

### [• Autonomous Research Synthesizer](https://github.com/aragit/autonomous-research-synthesizer)
**Self-Directed Scientific Discovery Engine**
> Gemini 3 Pro, MCP, Semantic Scholar API, E2B Sandbox, Jupyter Kernel, Neo4j Citation Graph - CI/CD     
> <span style="color:#8B0000">⬤</span> `Private` • `Scientific Research` • `Neuro-Symbolic AI` • `Dynamic Tool Use`

<details>
<summary><b>Expand Architecture Insight →</b></summary>

- **Hybrid Framework:** Neural primary controller plans multi-step research workflows, dynamically calling symbolic tools for computation, retrieval, and verification — adapting strategy based on intermediate results.
- **Opportunistic Research Planning:** The LLM decomposes open-ended research questions into sub-goals, decides which literature APIs to query (PubMed, Semantic Scholar, bioRxiv, arXiv), and adapts when sources conflict or gaps emerge — no static retrieval pipeline.
- **Reproducible Analysis Execution:** E2B-sandboxed Jupyter kernel executes Python/R statistical analyses on raw datasets (GEO, Figshare) with full provenance tracking. The LLM generates analysis code, interprets outputs, and decides whether to re-run with modified parameters.
- **Cross-Modal Synthesis:** Native multimodal reasoning over text, tables, figures, and code. The LLM decides when to regenerate visualizations, when to query structured databases, and when to perform citation verification via Crossref DOI resolution.
- **Validation Boundary:** Citation verifier ensures all claims are grounded in retrieved sources. Conflict detector flags contradictory findings across papers. Human-in-the-loop gate for conclusions with >3 standard deviation novelty scores.

</details>

### [• Agentic Educational Tutoring Swarm](#)
**Adaptive tutoring system with concept mastery modeling**
> Knowledge graphs, tutoring agents, adaptive questioning, progress tracking    
> 🔵 `Concept` • 🎓 `Adaptive Learning`

<details>
<summary><b>Expand Architecture Insight →</b></summary>

- Assessment agents identify knowledge gaps dynamically
- Subject-specific agents provide targeted instruction
- Pedagogy agent adapts teaching strategy per learner profile
- Progress tracking agent measures mastery and retention trends

</details>

<br>

---

## Vision: Aethron AI

Transforming inference into measurable impact.

Aethron AI focuses on deploying next-generation Agentic AI that doesn't just exist in a sandbox. The goal is to build autonomous, multi-agent frameworks that:

- Seamlessly integrate into existing high-stakes workflows.
- Resolve complex reasoning pathways through deterministic safety guardrails.
- Shift the paradigm from human-in-the-loop to human-on-the-loop.

## Let's Connect

- LinkedIn: https://linkedin.com/in/arashnicoomanesh
- GitHub: https://github.com/aragit
- Website: https://aragit.github.io

---

⭐ If you find this interesting, follow my work — I'm building the future of Agentic AI ...!
