<h1 align="center">Hi, I'm Arash — Agentic AI Engineer & Architect</h1>

<p align="center">
  <b>Building Enterprise-Grade, Agentic AI Systems for High-Stakes Domains</b><br>
</p>



<!-- SECTION 1: Capability Architecture (Icons + consistent dark-slate colors) -->
<div align="center">
  <img src="https://img.shields.io/badge/Neuro--Symbolic-vLLM_|_cvxpylayers-333333?style=flat&logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/Multi--Agent-LangGraph_|_AutoGen-333333?style=flat&logo=diagram-project&logoColor=white">
  <img src="https://img.shields.io/badge/Dynamic_Tooling-Model_Context_Protocol-333333?style=flat&logo=server&logoColor=white">
  <img src="https://img.shields.io/badge/Edge_SLMs-Unsloth_|_ExecuTorch-333333?style=flat&logo=bolt&logoColor=white">
  <img src="https://img.shields.io/badge/Auditability-OpenTelemetry-333333?style=flat&logo=check-double&logoColor=white">
</div>
<br><br>



---
## Agentic AI Architecture

Agentic AI stack is organized across three composable pillars. Each maps to a subset of the full 7-layer architecture documented in [`ARCHITECTURE.md`](./ARCHITECTURE.md).


```yml
1. PERCEPTION & INFRASTRUCTURE
   Layers: Ingestion, Memory, Tool Registry
   Signal: Unstructured signals ──► Structured state

2. REASONING & CONTEXT
   Layers: Planning, Execution, Meta-Cognition
   Signal: State ──► Decision ──► Verified action

3. GOVERNANCE & SECURITY
   Layers: Deterministic Guardrails, Audit, Escalation
   Signal: Safety invariants enforced at every structural boundary
```

All three pillars share a common type algebra — neuro-symbolic architectures where neural components propose and symbolic components verify — enabling transfer across clinical, financial, energy, and marketing domains with 55–90% architectural reuse.

---

## Core Infrastructure & Stack

### - Compute, Acceleration & Edge
`PyTorch` • `vLLM` • `Unsloth` • `Triton Inference Server` • `ExecuTorch` • `ONNX Runtime` • `TensorRT` • `llama.cpp`

### -  Orchestration & Cognition
`LangGraph` • `CrewAI` • `Model Context Protocol (MCP)` • `FastAPI` • `Pydantic v2` • `Python`

### - Neuro-Symbolic & Mathematical Optimization
`cvxpy` • `cvxpylayers` • `SciPy` • `NetworkX`

### - Data, Vector & Knowledge Memory
`Neo4j` • `Qdrant` • `Redis` • `pgvector` • `Polars` • `SQLite (vec)`

### - Governance, Observability & DevOps
`Open Policy Agent (OPA)` • `OpenTelemetry` • `Jaeger` • `Prometheus` • `Docker` • `GitHub Actions CI/CD`

---

## Deep Dives

- [Architecture Specification](./ARCHITECTURE.md) — Multi-layer transferability matrix, 3-phase evolution narrative, full repository breakdowns with per-project engineering details
- [Technical Publications](https://github.com/aragit) — case studies and engineering notes across 8 industry verticals

---

<p align="center">
  <a href="https://linkedin.com/in/arashnicoomanesh">LinkedIn</a> ·
  <a href="https://github.com/aragit">GitHub</a> ·
  <a href="https://aragit.github.io">Website</a>
</p>
