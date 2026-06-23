<h1 align="center">Hi, I'm Arash — Agentic AI & LLM Engineer</h1>

<p align="center">
  Building multi-agent systems, clinical LLM pipelines, and neuro-symbolic AI infrastructure.
  <a href="https://aragit.github.io/#home">Portfolio →</a>
</p>

<div align="center">

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/arashnicoomanesh)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/arashnic)
[![Medium](https://img.shields.io/badge/Medium-000000?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@anicomanesh)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/Arnic)
[![Substack](https://img.shields.io/badge/Substack-FF6719?style=for-the-badge&logo=substack&logoColor=white)](https://anicomanesh.substack.com)
[![Twitter/X](https://img.shields.io/badge/-Twitter-000000?style=for-the-badge&logo=x&logoColor=white&labelColor=000000)](https://x.com/ANicoomanesh)
[![Google Cloud](https://img.shields.io/badge/Google_Cloud-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)](https://www.cloudskillsboost.google/profile/badges)

</div>

<div align="center">

[![Typing SVG](https://readme-typing-svg.herokuapp.com?font=Fira+Code&pause=1000&color=2F81F7&center=true&vCenter=true&width=650&lines=Agentic+AI+Engineer;Medical+ML+Specialist;Multi-Agent+Systems+Architect;LLM+Engineering+Expert)](https://git.io/typing-svg)

</div>

---

## 🧠 System Overview

This is not a simple project list.

It is a **composable intelligence stack** — a portfolio of systems that transform raw signals into structured reasoning, coordinated action, prediction, and real-world impact.

> The future of AI is not better prompts.  
> It is better systems.

---

## 🔭 Currently Building

- **[Agentic CBT System](#-agentic-systems--orchestration--autonomy)** — multi-agent architecture for autonomous cognitive behavioral therapy delivery
- **[Nash Marketing Agents](#-agentic-systems--orchestration--autonomy)** — game-theoretic equilibrium engine for competitive ad-bidding simulations
- **[Zero-Shot Demand Forecasting](#-temporal-intelligence--prediction--foresight)** — supply-chain telemetry with zero-shot time-series foundation models
- **[Neuro-Symbolic Agentic RAG](#-reasoning--knowledge-cores--validation-truth-control)** — deterministic multi-agent clinical core

---

## 🗺️ System Capability Map

This portfolio is organized by **capability layer** rather than by domain.  
The goal is to show how each project contributes to a larger agentic intelligence stack: perception → reasoning → orchestration → prediction → execution → frontier research.

<p align="center">
  <img src="assets/capability_map.png" alt="Capability Map" width="850px">
</p>

| Capability Layer | How It Is Categorized | What It Represents | Representative Projects | Typical Stack Signals |
|---|---|---|---|---|
| **👁️ Perception Systems** | Projects that convert raw input into structured signals | Extraction from text, images, motion, or clinical streams | `bionlp-llama3-service`, `surgical-vision-copilot`, `spatial-event-detector`, `edge-fall-detector` | Vision models, NLP extraction, stream processing, edge inference |
| **🧠 Reasoning & Knowledge Cores** | Projects that validate, ground, and organize knowledge | Retrieval, verification, symbolic control, reasoning alignment | `speculative-clinical-graphrag`, `deepseek-reasoning-finetuning`, `neuro-symbolic-agentic-rag`, research paper synthesis swarm | RAG, graph databases, policy validation, fine-tuning, claim verification |
| **🤖 Agentic Systems** | Projects built around autonomous multi-agent coordination | Negotiation, task delegation, strategy, and collaborative action | `enterprise-intelligence-crew`, `nash-marketing-agents`, `autonomous-procurement-swarm`, supply chain response, cybersecurity swarm, smart city traffic, legal negotiation, disaster response, tutoring swarm | Crew-based orchestration, game theory, negotiation, distributed control |
| **⏱️ Temporal Intelligence** | Projects that model change over time | Forecasting, early warning, time-series reasoning | `zero-shot-demand-foundation`, `icu-vitals-transformer` | TSFMs, forecasting pipelines, event prediction, streaming telemetry |
| **⚙️ Execution Systems** | Projects that turn intelligence into operational action | Pricing, authorization, personalization, resource allocation | `agentic-medicare-auth`, `realtime-intent-transformer`, `generative-dynamic-ad-renderer`, `spatial-dynamic-markdown-engine`, recipe optimization, energy grid balancing, sentiment market maker | Optimization, decision systems, market simulation, workflow automation |
| **🧬 Frontier Research** | Projects that expand beyond standard LLM/RAG patterns | Novel scientific and generative research directions | `protein-binder-flow` | Flow matching, computational biology, structural generation |

| Categorization Signal | What We Look For |
|---|---|
| **Input type** | Does the project begin with text, video, telemetry, graphs, or time series? |
| **Primary function** | Is it extracting, reasoning, coordinating, predicting, or executing? |
| **Control style** | Is the system deterministic, agentic, probabilistic, or hybrid? |
| **Core algorithm** | Does it rely on graph search, optimization, game theory, forecasting, or fine-tuning? |
| **System role** | Does it feed perception, enforce truth, orchestrate agents, forecast future state, or close the loop operationally? |

**In short:**  
a project belongs here because of the **role it plays in the intelligence stack**, not because of the industry it happens to target.

---

## 👁️ Perception Systems — Structure Extraction from the Real World

Systems that convert unstructured text, images, motion, and clinical streams into machine-readable representations.

### 🔹 [BioNLP LLaMA3 Service](https://github.com/aragit/bionlp-llama3-service/tree/main)

**Clinical entity extraction from unstructured EHR pipelines**

- **Stack:** LLaMA-3-8B, Unsloth, PEFT/LoRA, FastAPI
- **Core idea:** memory-efficient fine-tuning for multi-token biomedical NER
- **Role in system:** perception layer for structured clinical understanding

<details>
<summary><b>Architecture insight</b></summary>

- Extracts biomarkers and clinical terms from messy text
- Optimized for low-memory fine-tuning and fast inference
- Designed to feed downstream retrieval and reasoning layers

</details>

### 🔹 [Surgical Vision Copilot](mailto:anicomanesh@gmail.com?subject=Access%20Request%3A%20surgical-vision-copilot)

**Real-time surgical understanding with vision-language models**

- **Stack:** PyTorch, Video-LLaVA, OpenCV, temporal action localization
- **Core idea:** tracks procedural steps and predicts the next needed tool
- **Status:** private / request access
- **Role in system:** visual perception layer for clinical action understanding

<details>
<summary><b>Architecture insight</b></summary>

- Encodes streaming video into temporal event sequences
- Adds action understanding on top of raw frame perception
- Built to support assistive decision loops in time-sensitive environments

</details>

### 🔹 [Spatial Event Detector](https://github.com/aragit/spatial-event-detector)

**Real-time kinematic telemetry engine**

- **Stack:** PyTorch, YOLOv11-Pose, OpenCV, NumPy
- **Core idea:** converts raw pose estimation into deterministic movement states
- **Role in system:** motion-to-symbol pipeline for edge intelligence

<details>
<summary><b>Architecture insight</b></summary>

- Separates pixel ingestion from logical inference
- Computes joint vectors over temporal windows
- Uses a state machine to detect meaningful spatial events

</details>

### 🔹 [Edge Fall Detector](https://github.com/aragit/edge-fall-detector)

**Real-time patient fall detection on NVIDIA Jetson**

- **Stack:** YOLOv11-Pose, TensorRT, Jetson Orin, MQTT, OpenCV
- **Core idea:** on-device fall detection with privacy-preserving inference
- **Role in system:** edge perception layer for clinical safety monitoring

<details>
<summary><b>Architecture insight</b></summary>

- Converts pose estimation into TensorRT-optimized inference
- Supports local processing for privacy and low latency
- Designed for continuous monitoring in constrained environments

</details>



## 🧠 Reasoning & Knowledge Cores — Validation, Truth Control, and Retrieval

Systems that turn perception into structured reasoning, grounded answers, and verifiable decisions.

### 🔹 [Speculative Graph RAG](https://github.com/aragit/speculative-clinical-graphrag)

**Self-correcting clinical knowledge core**

- **Stack:** LlamaIndex, Neo4j, vLLM, DeepSeek-R1
- **Core idea:** graph-based retrieval with a verification layer for clinical facts
- **Role in system:** reasoning core for grounded medical intelligence

<details>
<summary><b>Architecture insight</b></summary>

- Combines dense graph retrieval with structured verification
- Validates extracted pathways against medical taxonomies
- Designed to reduce hallucination and improve traceability

</details>

### 🔹 [DeepSeek Reasoning Fine-Tuning](https://github.com/aragit/deepseek-reasoning-finetuning)

**Medical CoT LoRA alignment pipeline**

- **Stack:** Unsloth, PyTorch, Hugging Face, TRL
- **Core idea:** parameter-efficient reasoning alignment with 4-bit quantization
- **Role in system:** reasoning optimization layer for chain-of-thought behavior

<details>
<summary><b>Architecture insight</b></summary>

- Fine-tunes reasoning behavior efficiently under memory constraints
- Maps diagnostic thought chains into model behavior
- Useful for improving structured response quality in expert workflows

</details>

### 🔹 [Neuro-Symbolic Agentic RAG](mailto:anicomanesh@gmail.com?subject=Access%20Request%3A%20neuro-symbolic-agentic-rag)

**Deterministic multi-agent clinical core**

- **Stack:** request access
- **Core idea:** cyclic planning and execution with policy validation
- **Status:** private / request access
- **Role in system:** control plane for safe clinical reasoning

<details>
<summary><b>Architecture insight</b></summary>

- Coordinates single- and multi-agent graphs
- Adds an Open Policy Agent validation ring
- Designed for deterministic workflows where correctness matters

</details>

### 🔹 [Agentic Research Paper Review & Synthesis Swarm](#)

**Swarm of agents that ingests papers, verifies claims, and synthesizes literature reviews**

- **Stack:** PDF parsing, ArXiv API, claim extraction, contradiction detection, graph synthesis
- **Core idea:** identifies hypotheses, validates findings, and resolves conflicting evidence
- **Status:** concept / not yet implemented
- **Role in system:** research reasoning layer for scientific synthesis

<details>
<summary><b>Architecture insight</b></summary>

- Ingestion agent extracts paper sections and structure
- Verification agent cross-references claims against citations
- Conflict detection agent flags contradictions across sources
- Synthesis agent generates reviews and identifies research gaps

</details>



## 🤖 Agentic Systems — Orchestration & Autonomy

Systems that coordinate multiple agents, strategies, and tools to act in dynamic environments.

### 🔹 [Enterprise Intelligence Crew](https://github.com/aragit/enterprise-intelligence-crew/tree/main)

**Autonomous content lifecycle platform**

- **Stack:** CrewAI, LangChain, Pydantic, ChromaDB
- **Core idea:** hierarchical multi-agent workflow with strict schema validation
- **Role in system:** orchestration layer for collaborative agent execution

<details>
<summary><b>Architecture insight</b></summary>

- Includes specialized agents for trend investigation, risk analysis, and copywriting
- Uses memory syncs and delegation constraints
- Enforces structured outputs through Pydantic containers

</details>

### 🔹 [Nash Marketing Agents](https://github.com/aragit/agentic-nash-marketing)

**Multi-agent competitive market simulation engine**

- **Stack:** NumPy, SciPy, SQLite/PostgreSQL, FastAPI, Pydantic, SQLAlchemy, Docker, pytest
- **Core idea:** models non-cooperative ad-bidding using mixed-strategy Nash equilibria
- **Role in system:** decision-making layer for strategic competition

<details>
<summary><b>Architecture insight</b></summary>

- Simulates autonomous brand agents under resource constraints
- Helps prevent budget-depletion loops in bidding environments
- Useful for strategic experimentation before production deployment

</details>

### 🔹 [Autonomous Procurement Swarm](https://github.com/aragit/autonomous-procurement-swarm)

**Multi-agent contract negotiation swarm**

- **Stack:** Ray/RLlib, CrewAI, vLLM, Python
- **Core idea:** buyer and seller agents negotiate procurement contracts autonomously
- **Role in system:** decentralized negotiation layer for constrained environments

<details>
<summary><b>Architecture insight</b></summary>

- Simulates market pricing, inventory constraints, and geopolitical risk
- Designed for collaborative yet adversarial agent behavior
- Shows how autonomy can be applied to real business operations

</details>

### 🔹 [Agentic Supply Chain Disruption Response](#)

**Multi-agent supply chain disruption simulator**

- **Stack:** multi-agent orchestration, graph routing, local LLM negotiation, optimization
- **Core idea:** suppliers, warehouses, and retailers autonomously re-plan when disruptions hit
- **Status:** concept / not yet implemented
- **Role in system:** coordination layer for logistics under uncertainty

<details>
<summary><b>Architecture insight</b></summary>

- Orchestrator detects disruptions and triggers re-planning
- Supplier agents manage inventory, pricing, and capacity
- Logistics agent computes alternative routes with graph algorithms
- Retailer agents negotiate orders and demand shifts

</details>

### 🔹 [Agentic Cybersecurity Threat Hunting Swarm](#)

**Autonomous SOC swarm for anomaly detection and response**

- **Stack:** synthetic network telemetry, log correlation, state machine workflows, local LLM reasoning
- **Core idea:** scouts detect anomalies, correlation agents link them, response agents isolate threats
- **Status:** concept / not yet implemented
- **Role in system:** defensive autonomy layer for security operations

<details>
<summary><b>Architecture insight</b></summary>

- Implements alert → triage → contain → eradicate → recover flow
- Forensics agent reconstructs attack chains post-incident
- Designed around incident-response realism and testability

</details>

### 🔹 [Agentic Smart City Traffic Optimization](#)

**Multi-agent traffic signal and routing optimizer**

- **Stack:** graph road network, city simulation, intersection agents, routing optimization
- **Core idea:** traffic lights, transit, and emergency routing negotiate in real time
- **Status:** concept / not yet implemented
- **Role in system:** urban autonomy layer for city-scale coordination

<details>
<summary><b>Architecture insight</b></summary>

- Intersection agents negotiate green-light durations
- Transit agent balances passenger load and schedules
- Emergency agent overrides for ambulances and fire trucks
- City orchestrator resolves global conflicts and deadlocks

</details>

### 🔹 [Agentic Legal Contract Negotiation Engine](#)

**Autonomous contract negotiation system for legal agreements**

- **Stack:** Pareto frontier computation, Nash bargaining, clause generation, risk analysis
- **Core idea:** legal agents negotiate terms and identify risks for both parties
- **Status:** concept / not yet implemented
- **Role in system:** negotiation layer for formal agreements

<details>
<summary><b>Architecture insight</b></summary>

- Party agents represent buyer/seller or tenant/landlord sides
- Mediator proposes compromises and detects deadlocks
- Risk analyst flags problematic clauses and hidden tradeoffs

</details>

### 🔹 [Agentic Disaster Response Coordination](#)

**Multi-service disaster coordination and rescue prioritization system**

- **Stack:** incident simulation, resource allocation, routing, triage, local LLM strategy generation
- **Core idea:** fire, medical, police, and logistics units coordinate under uncertainty
- **Status:** concept / not yet implemented
- **Role in system:** high-stakes autonomy layer for public safety

<details>
<summary><b>Architecture insight</b></summary>

- Incident commander sets global priorities
- Fire, medical, police, and logistics agents coordinate response
- Ethical triage reasoning under scarce resources
- Real-time map dashboard for operational awareness

</details>

### 🔹 [Agentic Educational Tutoring Swarm](#)

**Adaptive tutoring system with concept mastery modeling**

- **Stack:** knowledge graph, tutoring agents, adaptive questioning, progress reporting
- **Core idea:** subject agents, pedagogy agent, and motivation agent personalize learning
- **Status:** concept / not yet implemented
- **Role in system:** autonomous education layer for personalized instruction

<details>
<summary><b>Architecture insight</b></summary>

- Assessment agent diagnoses knowledge gaps
- Subject agents explain and reinforce concepts
- Pedagogy agent adapts teaching style
- Reporting agent tracks mastery and at-risk students

</details>



## ⏱️ Temporal Intelligence — Prediction & Foresight

Systems that model time, anticipate outcomes, and enable proactive decision-making.

### 🔹 [Zero-Shot Demand Foundation](https://github.com/aragit/zero-shot-demand-foundation)

**Predictive supply-chain telemetry pipeline**

- **Stack:** Amazon Chronos-2, Google TimesFM 2.5, Hugging Face
- **Core idea:** zero-shot forecasting with time-series foundation models
- **Role in system:** foresight layer for inventory and demand planning

<details>
<summary><b>Architecture insight</b></summary>

- Moves beyond traditional ARIMA/LSTM pipelines
- Uses foundation models for long-context temporal reasoning
- Incorporates exogenous signals for more realistic forecasting

</details>

### 🔹 [ICU Vitals Transformer](#)

**Transformer-based ICU vitals forecaster**

- **Stack:** TimesFM 2.5, PatchTST, Redpanda/Kafka, FastAPI, WebSockets
- **Core idea:** predicts critical events from streaming physiological signals
- **Status:** coming soon
- **Role in system:** temporal reasoning layer for patient monitoring

<details>
<summary><b>Architecture insight</b></summary>

- Ingests high-frequency vitals from an HL7 FHIR gateway
- Converts streams into forecastable windows
- Intended to support early warning for critical deterioration

</details>



## ⚙️ Execution Systems — Closing the Loop in the Real World

Systems that transform inference into measurable business or clinical action.

### 🔹 [Spatial Dynamic Markdown Engine](#)

**Dynamic pricing and liquidation engine**

- **Stack:** TimesFM 2.5, Ray, SciPy, PostgreSQL
- **Core idea:** real-time pricing for seasonal and stagnant inventory
- **Status:** coming soon
- **Role in system:** commercial execution layer for margin protection

<details>
<summary><b>Architecture insight</b></summary>

- Adjusts pricing from temporal signals and inventory state
- Designed to reduce margin erosion through faster action
- Bridges forecasting with operational execution

</details>

### 🔹 [Agentic Medicare Authorization](mailto:anicomanesh@gmail.com?subject=Access%20Request%3A%20agentic-medicare-auth)

**Agentic prior authorization engine**

- **Stack:** request access
- **Core idea:** maps EHR evidence against CMS guidance to generate authorization forms
- **Status:** private / request access
- **Role in system:** execution layer for healthcare operations

<details>
<summary><b>Architecture insight</b></summary>

- Ingests raw EHR data and regulatory documents
- Compiles evidence-backed submissions to reduce denials
- Designed to automate repetitive, high-friction administrative work

</details>

### 🔹 [Real-Time Intent Transformer](mailto:anicomanesh@gmail.com?subject=Access%20Request%3A%20realtime-intent-transformer)

**Session-based e-commerce intent telemetry engine**

- **Stack:** request access
- **Core idea:** maps live session behavior to contextual incentives
- **Status:** private / request access
- **Role in system:** agentic decision layer for adaptive commerce

<details>
<summary><b>Architecture insight</b></summary>

- Ingests clickstream and cart activity in real time
- Uses dense behavioral stores for session interpretation
- Triggers contextual actions from inferred intent

</details>

### 🔹 [Generative Dynamic Ad Renderer](mailto:anicomanesh@gmail.com?subject=Access%20Request%3A%20generative-dynamic-ad-renderer)

**Telemetry-driven ad generation pipeline**

- **Stack:** request access
- **Core idea:** dynamically generates ad scripts from live user telemetry
- **Status:** private / request access
- **Role in system:** execution layer for personalized media generation

<details>
<summary><b>Architecture insight</b></summary>

- Converts behavioral signals into creative output
- Connects LLM generation with rendering automation
- Built for adaptive content delivery at runtime

</details>

### 🔹 [Agentic Recipe & Nutrition Optimization System](#)

**Personalized meal planning through negotiation between nutrition, taste, budget, and ingredients**

- **Stack:** linear programming, USDA nutrition data, grocery simulation, LLM recipe generation
- **Core idea:** balances nutritional targets with preferences and cost constraints
- **Status:** concept / not yet implemented
- **Role in system:** optimization layer for food-tech decisioning

<details>
<summary><b>Architecture insight</b></summary>

- Nutrition agent enforces macro and micro targets
- Preference agent learns taste profiles and substitutions
- Budget agent optimizes cost under ingredient constraints
- Shopping agent navigates dynamic pricing in simulated stores

</details>

### 🔹 [Agentic Energy Grid Balancing System](#)

**Smart grid simulation with decentralized energy trading**

- **Stack:** double auction engine, carbon optimization, generation and storage agents, real-time grid control
- **Core idea:** balances supply and demand while minimizing carbon impact
- **Status:** concept / not yet implemented
- **Role in system:** infrastructure execution layer for energy systems

<details>
<summary><b>Architecture insight</b></summary>

- Solar and wind agents model variable generation
- Battery agent handles storage arbitrage
- Consumer agents respond to price and demand
- Grid orchestrator stabilizes frequency and balance

</details>

### 🔹 [Agentic Sentiment-Driven Market Maker](#)

**Autonomous market-making system driven by sentiment and risk signals**

- **Stack:** order book simulation, sentiment scoring, risk control, regulatory detection
- **Core idea:** adjusts spreads and inventory using news sentiment and market signals
- **Status:** concept / not yet implemented
- **Role in system:** execution layer for algorithmic trading simulation

<details>
<summary><b>Architecture insight</b></summary>

- Market makers compete with different risk appetites
- Sentiment agent interprets synthetic news
- Regulatory agent detects spoofing, layering, and wash trading
- Exchange engine handles matching and discovery

</details>



## 🧬 Frontier Research — Beyond Conventional AI

Systems that push beyond standard LLM/RAG pipelines into new computational frontiers.

### 🔹 [Protein Binder Flow](https://github.com/aragit/Flow-Matching-Protein-Binder-Generator)

**Flow-matching protein binder generator**

- **Stack:** PyTorch, Biopython, Flow Matching Primitives, FoldSeek
- **Core idea:** structural molecular generation through flow matching
- **Role in system:** research frontier for generative bio-AI

<details>
<summary><b>Architecture insight</b></summary>

- Moves beyond diffusion-style generation
- Targets novel protein-ligand binding behavior
- Demonstrates capability expansion into computational biology

</details>



## 🛠️ Core Engineering Showcases

### 🔹 [DeepSeek Reasoning Fine-Tuning](https://github.com/aragit/deepseek-reasoning-finetuning)
### 🔹 [BioNLP LLaMA3 Service](https://github.com/aragit/bionlp-llama3-service/tree/main)
### 🔹 [Spatial Event Detector](https://github.com/aragit/spatial-event-detector)
### 🔹 [Edge Fall Detector](https://github.com/aragit/edge-fall-detector)

These projects show the foundation beneath the system: fine-tuning, extraction, perception, and edge deployment.



## 🛒 E-Commerce & MarTech

### 🔹 [Generative Dynamic Ad Renderer](mailto:anicomanesh@gmail.com?subject=Access%20Request%3A%20generative-dynamic-ad-renderer)
### 🔹 [Real-Time Intent Transformer](mailto:anicomanesh@gmail.com?subject=Access%20Request%3A%20realtime-intent-transformer)
### 🔹 [Spatial Dynamic Markdown Engine](#)
### 🔹 [Agentic Sentiment-Driven Market Maker](#)

These projects translate AI into revenue, pricing, conversion, and adaptive customer response.



## ✍️ Recent Articles & Insights

### Agentic AI
- [The Planning-Rubicon: Why the Vast Majority of AI Agents Are Just Expensive Chatbots — Part I](https://medium.com/@anicomanesh/the-planning-rubicon-why-the-vast-majority-of-ai-agents-are-just-expensive-chatbots-part-i-fa0409a10d8e)
- [From Generative to Agentic AI: A Roadmap in 2026](https://medium.com/@anicomanesh/from-generative-to-agentic-ai-a-roadmap-in-2026-8e553b43aeda)
- [Beyond the Hype of Expensive Chatbots: Bridging Strategic Business Intent with Adaptive Agentic Systems](https://medium.com/@anicomanesh/beyond-the-hype-of-expensive-chatbots-bridging-strategic-business-intent-with-adaptive-agentic-d1144e9df041)

### Generative AI and LLM Engineering
- [A Dive into Unsloth & Gemma 3](https://medium.com/@anicomanesh/a-dive-into-unsloth-gemma-3-fine-tune-gemma-3-12b-with-unsloth-trl-for-custommer-service-53e93692d4d6)
- [A Dive Into LLM Output Configuration, Prompt Engineering Techniques and Guardrails Part I](https://medium.com/@anicomanesh/a-dive-into-advanced-prompt-engineering-techniques-for-llms-part-i-23c7b8459d51)
- [Token Efficiency and Compression Techniques in Large Language Models](https://medium.com/@anicomanesh/token-efficiency-and-compression-techniques-in-large-language-models-navigating-context-length-05a61283412b)

<details>
<summary><b>📚 See all articles</b></summary>

### Applied Machine Learning
- [First Steps Toward Building an Autonomous Agentic AI for CBT](https://anicomanesh.substack.com/p/first-steps-toward-building-an-autonomous)
- [Model Drift: Identifying and Monitoring for Model Drift](https://anicomanesh.substack.com/p/model-drift-identifying-and-monitoring)
- [Evolution of Recommendation Algorithms, Part I](https://medium.com/@anicomanesh/evolution-of-recommendation-algorithms-part-i-fundamentals-and-classical-recommendation-bb1c0bce78a9)
- [Machine Learning Interpretability (MLI) with XGBoost and SHAP](https://medium.com/@anicomanesh/interpretable-machine-learning-iml-with-xgboost-and-additive-tools-42258fb1f14)
- [Data Leakage: Causes, Effects and Solutions](https://medium.com/@anicomanesh/data-leakage-causes-effects-and-solutions-6cc44a149e1c)

</details>



## 🧭 Design Principles

- Systems > Models  
- Agents > Pipelines  
- Reasoning > Generation  
- Constraints > Prompts  
- Architecture > Hacks  



## 🏗️ Vision: Aethron AI

Building next-generation Agentic AI systems that:

- operate autonomously
- reason under constraints
- coordinate as intelligent systems
- integrate into real-world workflows

---



<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/HuggingFace-FCC624?style=for-the-badge&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/vLLM-000000?style=for-the-badge&logo=github&logoColor=white" />
  <img src="https://img.shields.io/badge/Unsloth-111827?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenAI-000000?style=for-the-badge&logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/Anthropic-111827?style=for-the-badge&logo=anthropic&logoColor=white" />
  <img src="https://img.shields.io/badge/Ollama-3B82F6?style=for-the-badge&logo=ollama&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-0F172A?style=for-the-badge&logo=langchain&logoColor=white" />
  <img src="https://img.shields.io/badge/CrewAI-111827?style=for-the-badge&logo=github&logoColor=white" />
  <img src="https://img.shields.io/badge/Haystack-005F6B?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Chainlit-2D6CDF?style=for-the-badge&logo=chainlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Ray-0284C7?style=for-the-badge&logo=ray&logoColor=white" />
  <img src="https://img.shields.io/badge/Temporal-111827?style=for-the-badge&logo=temporal&logoColor=white" />
  <img src="https://img.shields.io/badge/FAISS-0F766E?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-7C3AED?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Neo4j-4581C3?style=for-the-badge&logo=neo4j&logoColor=white" />
  <img src="https://img.shields.io/badge/PostgreSQL-336791?style=for-the-badge&logo=postgresql&logoColor=white" />
  <img src="https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=redis&logoColor=white" />
  <img src="https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=mongodb&logoColor=white" />
  <img src="https://img.shields.io/badge/Elasticsearch-005571?style=for-the-badge&logo=elasticsearch&logoColor=white" />
  <img src="https://img.shields.io/badge/Milvus-00A1EA?style=for-the-badge&logo=milvus&logoColor=white" />
  <img src="https://img.shields.io/badge/Supabase-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/Kubernetes-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white" />
  <img src="https://img.shields.io/badge/Celery-37814A?style=for-the-badge&logo=celery&logoColor=white" />
  <img src="https://img.shields.io/badge/Kafka-231F20?style=for-the-badge&logo=apachekafka&logoColor=white" />
  <img src="https://img.shields.io/badge/Prefect-070E10?style=for-the-badge&logo=prefect&logoColor=white" />
  <img src="https://img.shields.io/badge/Kedro-3D2C8D?style=for-the-badge&logo=kedro&logoColor=white" />
  <img src="https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white" />
  <img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white" />
  <img src="https://img.shields.io/badge/Google_Cloud-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white" />
  <img src="https://img.shields.io/badge/Azure-0078D4?style=for-the-badge&logo=microsoftazure&logoColor=white" />
  <img src="https://img.shields.io/badge/Azure_Functions-0062AD?style=for-the-badge&logo=azurefunctions&logoColor=white" />
  <img src="https://img.shields.io/badge/Cloudflare-F38020?style=for-the-badge&logo=cloudflare&logoColor=white" />
  <img src="https://img.shields.io/badge/Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white" />
  <img src="https://img.shields.io/badge/Databricks-EF3E2C?style=for-the-badge&logo=databricks&logoColor=white" />
  <img src="https://img.shields.io/badge/Terraform-7B42BC?style=for-the-badge&logo=terraform&logoColor=white" />
  <img src="https://img.shields.io/badge/Pulumi-8A3391?style=for-the-badge&logo=pulumi&logoColor=white" />
  <img src="https://img.shields.io/badge/Prometheus-E6522C?style=for-the-badge&logo=prometheus&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenTelemetry-000000?style=for-the-badge&logo=opentelemetry&logoColor=white" />
  <img src="https://img.shields.io/badge/Go-00ADD8?style=for-the-badge&logo=go&logoColor=white" />
  <img src="https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust&logoColor=white" />
  <img src="https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white" />
  <img src="https://img.shields.io/badge/SQL-336791?style=for-the-badge&logo=postgresql&logoColor=white" />
  <img src="https://img.shields.io/badge/Notebook-Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />
</p>



## 📫 Let's Connect

- LinkedIn: https://linkedin.com/in/arashnicoomanesh
- GitHub: https://github.com/aragit

---

⭐ If you find this interesting, follow my work — I’m building the future of Agentic AI.
