# Agentic Career Advisor

This repository contains a **multi-agent AI system** that generates personalized career development plans by analyzing job market data from the **Google Job Skills dataset**, identifying skill gaps, and recommending free learning resources using **LangChain and a locally-run LLM**.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Jupyter Notebooks](#jupyter-notebooks)
- [Installation](#installation)
- [Key Engineering Decisions](#key-engineering-decisions)
- [Author](#author)

## Project Overview

This project builds a full pipeline that leverages a system of four autonomous AI agents to provide data-driven, personalized career guidance. The system takes a user's target job title, analyzes job postings to determine required skills, compares them against the user's current skillset, finds free learning resources for the identified gaps, and synthesizes all information into a final, human-readable report.

## Dataset

The project uses the **Google Job Skills dataset** from Kaggle. This dataset contains **1,250 job postings** from Google, including fields like `Title`, `Category`, `Responsibilities`, and `Qualifications`.

The dataset is available here: [Google Job Skills on Kaggle](https://www.kaggle.com/datasets/niyamatalmass/google-job-skills)

## Project Structure

```
agentic-skill-mapper/
│
├── data/
│   ├── raw/
│   │   └── job_skills.csv                # Raw Google Job Skills dataset
│   └── processed/
│       └── processed_google_jobs.csv    # Enriched data with extracted skills
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_batch_processing.ipynb
│   ├── 03_skills_analysis.ipynb
│   ├── 04_gap_analyst_test.ipynb
│   ├── 05_learning_planner_test.ipynb
│   └── 06_report_synthesizer_test.ipynb
│
├── src/
│   ├── agents/
│   │   ├── market_analyst.py
│   │   ├── gap_analyst.py
│   │   ├── learning_planner.py
│   │   └── report_synthesizer.py
│   └── core/
│       └── skill_processing.py           
│
├── requirements.txt
├── README.md
└── .gitignore

```

## Jupyter Notebooks

The development process is documented through a series of notebooks, each with a specific purpose.

| Step | Notebook | Description |
| :--- | :--- | :--- |
| **1. Data Exploration** | [01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb) | Initial analysis of the raw dataset and viability testing of the first agent. |
| **2. Batch Processing** | [02_batch_processing.ipynb](notebooks/02_batch_processing.ipynb) | GPU-accelerated pipeline on Google Colab to process all 1,250 jobs with the `Market Analyst` agent. Implements robust checkpointing. |
| **3. Analysis & Viz** | [03_skills_analysis.ipynb](notebooks/03_skills_analysis.ipynb) | Analysis of the extracted skills, including frequency counts and generation of the final data visualizations. |
| **4. Gap Analyst Dev** | [04_gap_analyst_test.ipynb](notebooks/04_gap_analyst_test.ipynb) | Development and testing of the `Gap Analyst` agent and the semantic skill normalization pipeline using sentence embeddings. |
| **5. Learning Planner Dev** | [05_learning_planner_test.ipynb](notebooks/05_learning_planner_test.ipynb) | Development and iterative prompt engineering for the `Learning Planner` agent, including handling LLM limitations like knowledge cutoff. |
| **6. Report Synthesizer Dev**| [06_report_synthesizer_test.ipynb](notebooks/06_report_synthesizer_test.ipynb) | Development and testing of the final `Report Synthesizer` agent, focusing on generating a structured Markdown report. |

## Installation

Clone the repository and install dependencies in a virtual environment. This project requires a local installation of [Ollama](https://ollama.com/).

```bash
# Clone the repository
git clone https://github.com/your-username/agentic-skill-mapper.git
cd agentic-skill-mapper

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the required LLM model via Ollama
ollama pull llama3:8b

```
## Key Engineering Decisions

Several key architectural decisions were made to ensure the project was robust, scalable, and met its constraints:

1.  **Zero-Cost & Local-First LLM Architecture:** A core constraint of this project was to achieve full functionality with zero monetary cost. By using **Ollama** to serve a powerful open-source model (**Llama 3 8B**) locally, the project completely avoids recurring API fees from providers like OpenAI or Google.
2.  **Semantic Skill Normalization:** Instead of simple string matching, a sophisticated NLP pipeline was implemented using **sentence embeddings and community detection**. This allows the system to automatically group semantically similar skills (e.g., "objective c" vs. "objective-c"), leading to a much more accurate analysis.
3.  **Robust Batch Processing Pipeline:** For the heavy task of processing 1,250 documents, a hybrid approach was used. The pipeline was executed on a **Google Colab GPU** for speed, and a robust **checkpointing system** was implemented to handle runtime crashes, ensuring the process could be resumed without data loss.
4.  **Modular, Agent-Based Design:** All core logic is separated into distinct modules (`src/core`, `src/agents`). Development was done in notebooks for rapid iteration, but the final, validated code was refactored into clean, reusable Python functions.

## Author

Developed by Gustavo Gomes

- [LinkedIn](https://www.linkedin.com/in/gustavo-alves-gomes/)
