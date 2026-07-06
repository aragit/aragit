<h1 align="center">Arash Nicoomanesh</h1>

<p align="center">
  <b>Building Enterprise-Grade, Agentic AI Systems for High-Stakes Domains</b><br>
</p>

<p align="center">
  <a href="https://aragit.github.io">
    <img src="https://img.shields.io/badge/Website-aragit.github.io-14B8A6?style=for-the-badge&logo=google-chrome&logoColor=white&labelColor=0F172A">
  </a>
</p>

---
## About

I  do not just deploy models or optimize prompts. I engineer compound Agentic AI systems — architectures where neural reasoning and symbolic governance are composed into standardized layers that transfer across high-liability industries.

The agentic stack across my repositories is organized into three composable architectural pillars. Each pillar maps directly to specific operational layers of my **Multi-Layer Composable Intelligence Stack**:

### 01 / Perception & Infrastructure
* **Layers Wrapped:** `01 // Perception`, `02 // Memory`, `03 // Tool Registry` (via Model Context Protocol)
* **Data Flow:** Unstructured enterprise telemetry $\rightarrow$ Validated, context-hydrated semantic state.

### 02 / Reasoning & Context
* **Layers Wrapped:** `04 // Reasoning`, `05 // Execution`, `07 // Meta-Cognition`
* **Data Flow:** State evaluation $\rightarrow$ Dynamic planning state-machines $\rightarrow$ Verified, idempotent execution.

### 03 / Governance & Security
* **Layers Wrapped:** `06 // Governance` (Policy-as-Code / Hard Mathematical Invariants)
* **Data Flow:** Explicit logical guardrails enforced natively at every structural component boundary.

> **The Reusability Moat:** All pillars share a unified neuro-symbolic type algebra where neural networks propose and symbolic components verify. This mathematical separation allows for **55%–90% core architectural reuse** across seemingly divergent verticals, including clinical oncology, game-theoretic marketing, and predictive forecasting.

---

## Deep Dives

- [Architecture Specification](./ARCHITECTURE.md) — Multi-layer transferability matrix, 3-phase evolution narrative, full repository breakdowns with per-project engineering details
- [My Website](https://aragit.github.io/) — Learn more about Neuro Symbolic Architecure and Patterns as well as mapping to domain specific use cases and repos. 

---

## Agentic AI Repositories Navigation
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

#### **Example: From ✨ Cross-Domain Neuro-Symbolic Architecture category**

### [2. Speculative GraphRAG](https://github.com/aragit/speculative-clinical-graphrag) 
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
<br>

## Core Infrastructure & Stack


`PyTorch` `vLLM` `Unsloth` `Triton Inference` `ExecuTorch` `ONNX Runtime` `TensorRT` `llama.cpp`
`LangGraph` `CrewAI` `FastAPI` `MCP` `Pydantic` `Qdrant` `Redis` `pgvector` `Polars` `Docker` `GitHub Actions` `OPA` `OpenTelemetry` `Prometheus` `cvxpy`

---

<p align="center">
  <a href="https://linkedin.com/in/arashnicoomanesh">LinkedIn</a> ·
  <a href="https://github.com/aragit">GitHub</a> ·
  <a href="https://aragit.github.io">Website</a>
</p>
