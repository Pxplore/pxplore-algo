# 📚 **Pxplore: Personalized Learning Path Planning with Goal-Driven Learner State Modeling**

> *Personalized Learning Path Planning with Goal-Driven Learner State Modeling* has been **submitted to The Web Conference (WWW 2026)**.
> The full paper will be released soon.

## Overview

**Pxplore** is an open-source adaptive learning path planning framework that personalizes learning experiences through **fine-grained learner state modeling** and **goal-driven reinforcement learning**.

Built on **large language models (LLMs)**, Pxplore dynamically constructs learner profiles, plans optimal content paths, and adapts teaching delivery to cognitive levels and learning styles — forming a closed pedagogical feedback loop between **profiling → planning → delivery → evaluation**.


## Key Features

| Module                        | Description                                                                                        |
| ----------------------------- | -------------------------------------------------------------------------------------------------- |
| 🧠 **Learner Profiling**      | Extracts cognitive and linguistic signals from student discussions, quizzes, and activity logs.    |
| 📖 **Learning Path Planning** | Combines hybrid retrieval and policy optimization to recommend fine-grained content snippets.      |
| 🎨 **Adaptive Delivery**      | Transforms recommended content into personalized instructional scripts using LLM-based adaptation. |
| 💬 **Session Management**     | Maintains conversational continuity, progress tracking, and contextual response generation.        |


## System Architecture

```
Pxplore/
├─ app.py                 # Application entry point
├─ config.py              # Environment configuration
├─ base.py                # Core data model definitions
├─ requirements.txt       # Dependency list
├─ data/                  # Data structure definitions
│   ├─ profile.py         # Learner profile schema
│   ├─ session.py         # Session model
│   ├─ snippet.py         # Learning content unit
│   └─ task.py            # Task and pipeline orchestration
├─ dataset/               # Dataset preparation scripts
│   ├─ gen_label.py
│   ├─ import_data.py
│   ├─ parse_snippets.py
│   └─ prompts/
├─ model/                 # Training and evaluation modules
│   ├─ calculate_reward.py
│   ├─ data_preprocessing.py
│   ├─ evaluation.py
│   ├─ prepare_data.py
│   ├─ prompts/
│   └─ test/
└─ service/
    ├─ llm/               # LLM service integration
    ├─ scripts/
    │   ├─ hybrid_retriever.py
    │   ├─ snippet_recommender.py
    │   ├─ student_profiling.py
    │   ├─ style_adaptation.py
    │   └─ session_controller.py
    ├─ utils/
    │   ├─ dense_embedding.py
    │   ├─ data_transformer.py
    │   └─ episodes_processor.py
    └─ test/
```

## 🚀 Getting Started

### Prerequisites

* Python **3.12+**
* **FastAPI**
* LLM provider (OpenAI GPT / Qwen / compatible APIs)
* *(Optional)* [Qdrant](https://qdrant.tech/) vector database for dense retrieval

### Installation

```bash
git clone https://github.com/Pxplore/pxplore-algo.git
cd pxplore-algo
pip install -r requirements.txt
```

### Configuration

Update `config.py` with your LLM API credentials and model settings.

### Run the Service

```bash
python -m app
```

Server starts at: [http://localhost:8899](http://localhost:8899)

---

## API Reference

### Learner Profiling

```http
POST /student_profile
Content-Type: application/json
{
  "behavioral_data": {
    "discussion_threads": [...],
    "page_interactions": [...],
    "review_loops": [...],
    "quizzes": [...]
  }
}
```

### Learning Path Planning

```http
POST /plan
{
  "student_profile": {...},
  "interaction_history": "previous context",
  "title": "current topic",
  "model": "model_id"
}
```

### Adaptive Delivery

```http
POST /style_adapt
{
  "student_profile": {...},
  "history_content": "...",
  "title": "current topic",
  "recommend_id": "snippet_id",
  "recommend_reason": "reason"
}
```


## Core Algorithms

### Hybrid Retrieval

Balances dense and sparse representations for optimal recall:

```math
HybridScore = α × BM25 + (1 − α) × DenseSim
```

### Learner Cognitive Modeling

* Follows **Bloom’s Taxonomy** (Knowledge → Understanding → Application → Analysis → Synthesis → Evaluation)
* Tracks cognitive evolution with **adaptive window analysis** and **trend regression**

### Policy Optimization

Employs **Group Relative Policy Optimization (GRPO)** guided by a **pedagogical reward function**, enabling reinforcement-based adaptation toward learners’ long-term developmental goals.


## Data Schemas

### Example — Learner Behavior Data

```json
{
  "discussion_threads": [
    {
      "thread_id": "thread_001",
      "messages": [
        {"author_type": "student", "content": "I think...", "timestamp": "2025-05-21T10:00:00Z"}
      ]
    }
  ],
  "page_interactions": [...],
  "review_loops": [...],
  "quizzes": [...]
}
```

### Example — Recommendation Output

```json
{
  "selected_candidate": {
    "id": "snippet_002",
    "bloom_level": "analysis",
    "summary": "Explains reinforcement learning adaptation.",
    "content": "..."
  },
  "reason": "Aligned with learner’s current cognitive stage."
}
```

---

## 📮 Citation

If you find our work helpful, please cite:

```bibtex
[coming soon]
```