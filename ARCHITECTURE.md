# The Composable Intelligence Stack

### Repository Navigation
This portfolio is organized by **domain and industry** to demonstrate how the same neuro-symbolic architecture principles transfer across verticals:

*   **[Cross-Domain Neuro-Symbolic Architecture](#-cross-domain-neuro-symbolic-architecture):** Foundational reasoning, memory, and orchestration layers that transfer across all verticals.    
*   **[Healthcare & Clinical](#-healthcare--clinical)**
    * **Speculative Clinical GraphRAG (Hybrid Architecture)**
    * **Clinical Triage Agentic Orchestrator**
    * **ICU Vitals Transformer(MCP-native tool)**
    * **Autonomous Medication Reconciliation**
    * **Biomedical Entity Extraction Engine**
    * **Autonomous Lab Interpretation & Critical Value Triage Agent**
    * **Clinical Differential Diagnosis Copilot**    
*   **[Marketing & Advertising](#-marketing--advertising):** Competitive Nash equilibrium bidding, real-time intent transformation, and generative ad rendering.
*   **[Supply Chain & Logistics](#-supply-chain--logistics)** Zero-Shot Demand Foundation(MCP Agentic Forecaster Skill),  Autonomous Procurement Swarm
*   **[Energy & Utilities](#-energy--utilities):** Agentic Energy Grid Balancing System
*   **[Computational Biology](#-computational-biology):** Protein Binder Flow,  Quantum-Bound Molecular Generator (QBMG, Zero-Waste Neuro-Symbolic Molecular Engine)
*   **[Finance & RegTech](#-finance--regtech):** Automated KYC & AML Screening Agent, Regulatory Intelligence Agent
*   **[Smart Cities & Urban Systems](#-smart-cities--urban-systems)** Agentic Smart City Traffic Optimization
*   **[Education & Research](#-education--research):** Autonomous Research Synthesizer, Agentic Educational Tutoring Swarm

---



## ✨ Cross-Domain Neuro-Symbolic Architecture

> Projects that define the foundational neuro-symbolic stack and transfer across industries.

### [• Edge SLM Optimizer](https://github.com/aragit/edge-slm-optimizer)
**Edge-First Small Language Model Compression & Deployment Pipeline**   
> PyTorch, ONNX Runtime Mobile, ExecuTorch, bitsandbytes, llama.cpp, pytest      
> 🟢 `Active` • `Edge AI` • `Model Compression`

<details>
<summary><b>Expand Architecture Insight →</b></summary>

- **Multi-Stage Quantization Pipeline**: FP32 → INT8 (static) → INT4 (dynamic via bitsandbytes/auto-gptq) with perplexity guardrails on WikiText-2
- **Dual Export Targets**: ONNX Runtime Mobile for cross-platform CPU inference; ExecuTorch XNNPACK delegate for ARM NEON optimization
- **Raspberry Pi 5 Benchmarking**: Latency, memory, power draw (INA219 + vcgencmd), thermal throttle detection — all under 5W sustained
- **Speculative Decoding**: 100M-parameter draft model distilled from main 1B model for 2× token generation speedup on edge
- **Telemetry Suite**: Real-time watts-per-token, CPU frequency monitoring, thermal event logging for edge reliability validation
- **Accuracy Preservation**: <15% perplexity degradation vs. FP32 baseline; MMLU subset evaluation for task-specific quality
- **CI/CD Reproducibility**: GitHub Actions with lint, pytest, Docker build — benchmarks versioned per commit

</details>

### [• Speculative Clinical GraphRAG (Type 2 Symbolic[Neuro] Architecture)](https://github.com/aragit/speculative-clinical-graphrag)
**Hybrid Neuro-Symbolic Clinical Knowledge Core with Hybrid RAG and Reasoning-Aware Verification**
> FastAPI, Pydantic v2, LangGraph, Neo4j, Qdrant, vLLM, DeepSeek-R1-Distill-Qwen-32B, SNOMED-CT/ICD-10-CM/RxNorm parsers, pytest , React 18, Vite, Tailwind CSS, Redis, Open Policy Agent (OPA), google/MedGemma-4B-IT (fine-tuned)        
> 🟢 `Active` • `Neuro-Symbolic Hybrid` • `Clinical Decision Support` • `MCP Control Plane` • `MAS Glass Box UI` • `100% Test Coverage`    



### [• Post-RAG Drift Evaluator](https://github.com/aragit/post-rag-drift-evaluator)
**Automated Latent Space Drift Telemetry & Comparative RAG Architecture Benchmark**
> Python 3.12, LiteLLM, Polars, pgvector, scikit-learn, SciPy, Streamlit, Docker, pytest, ruff, mypy
> 🟢 `Active` • `Embedding Drift Telemetry` • `Comparative RAG Evaluation` • `Statistical MLOps`


### [• DeepSeek Reasoning Fine-Tuning](https://github.com/aragit/deepseek-reasoning-finetuning)
**Medical chain-of-thought LoRA alignment pipeline**
> Unsloth, PyTorch, Hugging Face, TRL   
> 🟢 `ACTIVE` • `REASONING OPTIMIZATION LAYER`

### [• Enterprise Intelligence Crew](https://github.com/aragit/enterprise-intelligence-crew/tree/main)
**Autonomous enterprise trend intelligence pipeline**
> CrewAI, Ollama, FastAPI, ChromaDB, Pydantic V2   
> 🟢 `Active` • `Local-First` • `3-Agent Sequential Pipeline`

---

## 🏥 Healthcare & Clinical

### [• Clinical Triage Agentic Orchestrator](https://github.com/aragit/clinical-triage-agentic-orchestrator)
**Neuro-Symbolic Agentic Orchestrator for High-Stakes Clinical Triage with OPA Guardrails**     
> FastAPI, llama-cpp-python, Gemma 3n E4B, Qdrant (Hybrid BM25+Dense+RRF), SNOMED-CT/ICD-10-CM, Pydantic v2, Streamlit, Docker Compose, pytest     
> 🟢 `Active` • `Neuro-Symbolic` • `Clinical Decision Support` • `Edge-First`      

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

<details>
<summary><b>Expand Architecture Insight →</b></summary>

- MCP-native tool — exposes ingest_vitals, get_forecast, get_deterioration_index via Model Context Protocol
- Deterministic forecasting — multi-horizon trend extrapolation (1h/4h/12h) with clinical uncertainty bounds, no GPU required
- FHIR R4 ingestion — parses LOINC-coded Observation resources into sliding 5-minute windows
- NEWS2-inspired governance — deterministic deterioration index + severity classification (NORMAL → WARNING → ALERT → EMERGENCY)
- Stateless by design — caller decides action; tool returns structured predictions only

</details>


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


### [• Speculative Clinical Graph RAG, Hybrid Architecture](https://github.com/aragit/speculative-clinical-graphrag)
**Hybrid Neuro-Symbolic Clinical Knowledge Core with Hybrid RAG and Reasoning-Aware Verification**
> FastAPI, Pydantic v2, LangGraph, Neo4j, LlamaIndex, vLLM, DeepSeek-R1-Distill-Qwen-32B, SNOMED-CT, ICD-10-CM, RxNorm, pytest    
> 🟢 `Active` • `Neuro-Symbolic Hybrid` • `Clinical Decision Support` • `Hybrid RAG`



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

<br>

---

## 🎯 Marketing & Advertising

### [• Nash Marketing Agents (Game Theory)](https://github.com/aragit/agentic-nash-marketing)
**Neuro-Symbolic Multi-Agent Ad Auction Simulator with Nash Equilibrium Solver**
> FastAPI, Pydantic v2, SQLAlchemy 2.0, SciPy, SQLite/PostgreSQL, Docker, pytest   
> 🟢 `Active` • `Neuro-Symbolic` • `Game Theory` • `Ad Tech Simulation`

<details>
<summary><b>Expand Architecture Insight →</b></summary>

- **Neuro-Symbolic Hybrid**: LLM engine proposes stochastic bidding strategies; symbolic Nash solver validates equilibrium via iterative best-response with softmax annealing
- **VCG Second-Price Auction Engine**: Winners pay the next-highest bid; mathematical asserts enforce `paid ≤ bid` invariant on every round
- **Multi-Layer Budget Guardrails**: Soft warning (20%), hard cap (10%), emergency mode (5%) prevent catastrophic depletion in competitive bidding wars
- **Monte Carlo Nash Solver**: 5,000-sample stochastic win-probability estimation; converges to mixed-strategy equilibrium where no agent can improve utility unilaterally
- **49-Test Suite with Property-Based Verification**: Monotonicity, individual rationality, Nash convergence bounds, VCG payment correctness, guardrail action validity
- **Interactive Chart.js Dashboard**: Real-time simulation config, visualization panels, live event log

</details>

### [• Real-Time Intent Transformer](https://github.com/aragit/real-time-intent-transformer)
**Real-Time E-Commerce Intent Classification with Action Governance**
> FastAPI, Pydantic v2, Polars, scikit-learn, aiokafka, SQLite, OPA, pytest (84+ tests)      
> 🟢 `Active` • `E-Commerce Personalization` • `Behavioral Analytics`

<details>
<summary><b>Expand Architecture Insight →</b></summary>

A production-grade, dual-path neuro-symbolic system that classifies live shopping sessions into 7 intent categories and dispenses targeted interventions within 50ms on CPU — with deterministic governance, agentic reasoning, and closed-loop learning.

🧠 Architecture: Perceive → Reason → Govern → Execute → Learn
├─ System 1 (Fast Path): Rule-based heuristic + ML ensemble (RF/XGBoost) + Markov chain → <50ms
├─ System 2 (Agentic Path): LangGraph orchestrator → LLM Planner + GraphRAG (Neo4j) + Critic Agent → OPA-validated
└─ Meta-Cognition: Background evaluator with LLM-as-a-Judge, drift detection, and efficacy analytics

🔒 Security & Governance
• OPA/Rego v1 policies with fail-closed behavior (deny rules enforced, not ignored)
• Anti-gaming: 15-min suppression windows, max 3 discounts/month, 24h cooldown
• Fairness guardrails: demographic parity, no pricing discrimination
• Immutable audit ledger with SHA-256 idempotency keys
• RCE-patched: all user input deserialized via json.loads (no eval)

⚡ Performance
• Polars feature engineering: 18 behavioral features in <5ms
• Platt-scaled sigmoid calibration for statistically valid confidence scores
• Async SQLite I/O via asyncio.to_thread (no event loop blocking)
• Per-instance HTTP clients, bounded batch ingestion, defensive timeouts

📊 Observability
• Langfuse distributed tracing: System 1 vs System 2 routing, OPA evaluation spans, LLM token usage
• Prometheus metrics: REQUEST_COUNT, INTENT_PREDICTIONS, ACTIONS_DISPATCHED, evaluator drift gauges
• Configurable CORS, structured logging via Loguru

🧪 Quality
• 261 tests, 0 failures, 71% coverage
• Prompt injection security suite (markdown jailbreak, Unicode bypass, adversarial reasoning)
• Latency regression benchmarks (p95 <50ms target)
• Integration tests for end-to-end dual-path flows

🛠 Stack
FastAPI • LangGraph • Pydantic v2 • Polars • scikit-learn/XGBoost • aiokafka • SQLite/PostgreSQL • OPA/Rego • Neo4j • Langfuse • Prometheus • pytest-asyncio

📦 Infrastructure
Docker Compose (Kafka KRaft + OPA) • Synthetic data generation (34K events, 5K sessions) • Pre-trained model pipeline

Status: Production-hardened after comprehensive security & architectural audit

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

<details>
<summary><b>Expand Architecture Insight →</b></summary>

- **Foundation Models**: google/timesfm-2.5-200m-pytorch, Amazon Chronos-2 (`amazon/chronos-2`) via `BaseChronosPipeline` — zero-shot inference
- **Dual-Track Evaluation**: Point forecast (Accuracy Track) + quantile/sample trajectory parsing (Uncertainty Track) aligned with M5 Competition framework
- **3D Tensor Integration**: Strict `(n_series, n_variates, history_length)` input format; shape-agnostic output parser handles 3D point forecasts and 4D sample/quantile tensors via median extraction
- **Pydantic Input Validation**: `TimeSeriesInputPayload` enforces context bounds [16, 16,000] timesteps, horizon [1, 1024], and exogenous array alignment (price_index, promo_flag must match `context + horizon` length)
- **Pydantic Output Validation**: `ForecastOutputPayload` enforces mean prediction dimension match, optional p10/p90 quantile bands, model identifier tracking
- **M5 Competition Benchmarking**: Evaluates against Walmart daily sales (3,049 products, 10 stores, 3 states) with WAPE and RMSSE metrics; 128-step backtest window with active high-volume item filtering
- **Corporación Favorita Compatibility**: Secondary validation on Ecuadorian retail data with inflation markers and regional holidays for cross-locale zero-shot generalization testing
- **Exogenous Signal Support**: Optional price elasticity (`price_index`) and binary promotional event flags (`promo_flag`) aligned chronologically with target + horizon

</details>

### [• Autonomous Procurement Swarm](https://github.com/aragit/autonomous-procurement-swarm)
**LLM-Powered Multi-Agent Contract Negotiation for Supply Chain Optimization**
> FastAPI, Pydantic v2, SciPy, Transformers, Matplotlib, pytest   
> 🟢 `Active` • `Turn-Based Negotiation` • `Market Simulation` 

<br>

---

## ⚡ Energy & Utilities

### [• Agentic Energy Grid Balancing System](https://github.com/aragit/agentic-energy-grid-balancer)
**Neuro-Symbolic Multi-Agent Energy Market Simulator**
> FastAPI, Pydantic v2, SQLAlchemy 2.0, Docker, CI/CD, pytest (~120 tests), black, flake8    
> 🟢 `Active` • `Energy Market Simulation` 

<details>
<summary><b>Expand Architecture Insight →</b></summary>

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

</details>

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


<details>
<summary><b>Expand Architecture Insight →</b></summary>

- **Zero-waste generation** — every forward pass outputs a chemically valid bond adjacency matrix; no post-generation filtering or discard pipelines required
- **Implicit Function Theorem (IFT) backprop** — analytical Jacobian computation through the KKT equilibrium bypasses solver unrolling, enabling end-to-end gradient flow without memory explosion
- **Single substrate design** — neural backbone (SE(3)-GNN / transformer) and physics engine exist on the same mathematical substrate; no API boundaries, no JSON orchestration
- **Hard valency enforcement** — convex optimization boundary strictly caps per-atom bond sums (e.g., C≤4, O≤2) while minimizing Frobenius distortion from the neural guess
- **Modular backbone** — hot-swappable generators: dense MLPs, E(n)-Equivariant GNNs, or text-conditioned projections (e.g., MedGemma-4B-IT) all feed into the same physics core
- **Composable** — designed as a structural generative engine for Type 2 and Type 6 clinical intelligence pipelines, including multi-drug optimization and materials discovery

</details>

---

## 🏦 Finance & RegTech

### [• KYC-Auto (Know Your Customer)](https://github.com/aragit/kyc-auto)
**Automated KYC & AML Screening Agent**
> Qwen2.5-7B-Instruct, LangChain, OpenSanctions, Neo4j UBO Graph, PostgreSQL, Redis, FastAPI, OpenTelemetry — CPU-First / vLLM-Ready    
> 🟢 `Active` • `FinTech / RegTech` • `SLM-First Agent` • `Deterministic Risk Scoring`

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

### [Agentic Neuro-Symbolic Tutoring Swarm](https://github.com/aragit/agentic-educational-tutoring-swarm)
**Closed-Loop Multi-Agent Neuro-Symbolic Educational Platform**     
> LangGraph, LangChain, NetworkX, Pydantic v2, FastAPI, Redpanda, OPA, HTMX+SSE, Docker Compose, confluent-kafka, SQLite Checkpointer, PEFT LoRA — CPU-First / vLLM-Ready     
> 🟢 `Active` • `EdTech / AI Education` • `SLM-First Agent` • `Neuro-Symbolic Architecture`


<details>
<summary><b>Expand Architecture Insight →</b></summary>
   
- **Neuro-Symbolic Dual-Layer Architecture:** The system enforces a strict boundary between symbolic orchestration (LangGraph state machine + NetworkX knowledge graph) and neural execution (engine-level guided decoding via vLLM's `guided_json` logit masking). Pydantic schemas are compiled into grammar FSMs that mask invalid tokens at the logits level, mathematically guaranteeing 0% schema violation probability during inference.
- **Closed-Loop Multi-Agent Workflow:** Four specialized nodes execute in a deterministic LangGraph pipeline: `Assessment` (sandbox-informed LLM scoring) → `Progress Tracker` (EMA mastery arithmetic with α=0.4/0.6 blending) → `Pedagogy Governor` (dynamic ontology routing with runtime graph mutations at <0.35 critical failure) → `SME Instructor` (curriculum-grounded conversational teaching via vector store retrieval). Each node returns partial state mutations that flow through a persistent SQLite checkpointer.
- **Dynamic Curriculum Graph Mutation:** The `DynamicOntologyManager` allows the pedagogy node to programmatically inject remedial nodes and dependency edges into the NetworkX curriculum graph at runtime when a student exhibits critical prerequisite failures (<0.35 mastery score). The symbolic graph physically evolves based on learner performance data.
- **Sandbox-Informed Assessment:** Student code submissions are detected via markdown fence parsing, executed in an isolated Python sandbox with stdout/stderr capture, and the raw execution telemetry (compilation status, runtime errors, created variables) is injected directly into the logit-masked LLM evaluator. The model grades based on hard execution facts, not probabilistic guesswork.
- **Event-Driven Distributed Architecture:** The FastAPI gateway publishes student turns to Redpanda (Kafka-compatible) via `student.turn.submitted`. A background `Swarm Worker` consumes events, executes the full LangGraph pipeline, calls OPA for governance validation, and publishes resolved responses to `student.turn.resolved`. The gateway streams results back to the frontend via SSE.
- **Semantic Governance Layer (OPA):** Every agent output passes through an Open Policy Agent sidecar that enforces deterministic Rego policies. The `governance.rego` policy blocks `[CRITICAL_SYSTEM_BYPASS]` markers and prevents raw code spoon-feeding when mastery scores fall below 0.50, ensuring pedagogical compliance at the network boundary.
- **Zero-Build Real-Time Dashboard:** HTMX + SSE + Tailwind CSS frontend served directly by FastAPI's Jinja2 templates. Three SSE event channels (`agent_state`, `metrics`, `telemetry_log`) stream live execution grid status, token economics (Refinement Cost Ratio), and alignment telemetry to the browser without any JavaScript build toolchain.
- **Alignment Telemetry & Offline Fine-Tuning:** The `AlignmentLogger` captures every teacher-student interaction pair as JSONL SFT training data. The `train_lora.py` script runs PEFT LoRA fine-tuning on harvested datasets to progressively reduce logit masking overhead and improve structured output compliance across optimization cycles.
- **Multi-Provider LLM Factory:** `get_llm()` auto-detects the runtime environment — returns a deterministic `_MockChatModel` when no endpoint is configured (tests/simulation), connects to llama-cpp-python CPU server, or targets any OpenAI-compatible cloud API. `configure_guided_decoding()` attempts vLLM `guided_json` binding first, falls back to `with_structured_output()` for compatibility across providers.
- **111-Test Suite with 84% Coverage:** Unit tests cover KG traversal, EMA arithmetic, sandbox execution, ontology mutation, OPA fallback, graceful shutdown, and malformed payload handling. E2E tests validate session CRUD, Kafka publishing, SSE streaming, SQLite persistence, and edge cases. All passing with 0 lint errors and 0 type errors under `ruff` and `mypy --strict`.


</details>

<br>

---


## Let's Connect

- LinkedIn: https://linkedin.com/in/arashnicoomanesh
- GitHub: https://github.com/aragit
- Website: https://aragit.github.io

---

