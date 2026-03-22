# HiMem

**HiMem** is a hierarchical long-term memory framework designed for **large language models (LLMs)** operating in **long-horizon, multi-turn interactions**.  
It supports structured memory construction, efficient retrieval, and dynamic memory evolution for conversational agents.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/jojopdq/HiMem
cd HiMem
```

Create and activate a virtual environment:

```bash
conda create -n HiMem python=3.10.16
conda activate HiMem
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Init

```angular2html
docker compose up experiment/docker-compose.yaml -d
```

## Configuration

### LLM Provider

Select your preferred LLM provider and configure it according to your setup.  
At minimum, the following fields are required:

- `model`: `"YOUR_MODEL_NAME"`
- `api_key`: `"YOUR_API_KEY"`

Please update the corresponding configuration file before running any experiments.

### Memory Backend

You may freely customize:

- the **collection name** of the vector database, and
- the **index name** used for episode memory,

according to your experimental setup or storage preferences.

## Memory Construction

### Dataset Preparation

Create a data directory and place the dataset file inside:

`mkdir data`

Example directory structure:

`data/ ├── locomo_dev.json`

Make sure the dataset file is accessible before proceeding.

---

### Constructing Memory

Set the environment and choose the memory construction mode:

`ENV=dev # It should be consistent with your dataset file name, locomo_dev.json`
`CONSTRUCTION_MODE=all # Available modes: all | note | episode`

#### Without Knowledge Alignment (Note Memory Only)

`python experiment/memory_construction.py \   --env ${ENV} \   --construction_mode ${CONSTRUCTION_MODE}`

#### With Knowledge Alignment (Recommended)

`python experiment/memory_construction.py \   --env ${ENV} \   --construction_mode ${CONSTRUCTION_MODE} \   --enable_knowledge_alignment`

## Evaluation

### Environment Variables

Create a `.env` file in the root directory and set your API key:

`OPENAI_API_KEY="YOUR_API_KEY"`

### Running Evaluation

Specify the desired retrieval/search mode in the environment variables, then run:

`sh experiment/run.sh`

The evaluation script will automatically load the constructed memory and report performance metrics.

## Acknowledgement

This project is built upon the open-source codebase of [**Mem0**](https://github.com/mem0ai/mem0), while all core modeling and reasoning components are newly implemented, licensed under the Apache 2.0 License.

We thank the original authors for making their implementation publicly available.