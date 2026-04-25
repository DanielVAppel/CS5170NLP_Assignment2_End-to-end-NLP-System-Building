# CS5170 NLP Assignment 2: End-to-End NLP System Building

This project implements an end-to-end Retrieval-Augmented Generation (RAG) system for domain-specific question answering over Steam game review data.

The system loads Steam game review `.xlsx` files, converts the reviews into LangChain documents, splits them into chunks, embeds the chunks with OpenAI embeddings, stores them in a local FAISS vector database, and uses a LangChain/OpenAI agent to answer questions using retrieved review evidence.

Requirements:
os
pandas
dotenv
langchain
langchain-openai
langchain-community
langchain-core
langchain-text-splitters
faiss-cpu or faiss-gpu (For CUDA supported GPUs)
openpyxl (for reading excel files)

## Project Structure

```text
CS5170NLP_Assignment2_End-to-end-NLP-System-Building/
│
├── Rag_Agent.py
├── Rag_evaluation.py
├── run_evaluation.py
├── requirements.txt
├── .env
│
├── Anotated_Game_Reviews_Data/
│   ├── data/
│   │   ├── test/
│   │   │   ├── label_studio_import.json
│   │   │   ├── questions.txt
│   │   │   └── reference_answers.txt
│   │   │
│   │   └── train/
│   │       ├── label_studio_import.json
│   │       ├── questions.txt
│   │       └── reference_answers.txt
│   │    
│   └──knowledge_base/
│           └── game_docs.jsonl
│
├── Game_Reviews_Data/
│   ├── 1_Lethal_Company_1966720.xlsx
│   ├── 2_Counter_Strike_2_730.xlsx
│   ├── 3_Cyberpunk_2077_1091500xlsx.xlsx
│   └── ...
│
└── faiss_index/
    └── Generated after the first run
```

## What Each Main File Does

### `Rag_Agent.py`

This file builds the main RAG pipeline.

It:

- Loads the OpenAI API key from `.env`.
- Reads the `.xlsx` review files from `Game_Reviews_Data/`.
- Converts review rows into LangChain documents.
- Splits review text into smaller chunks.
- Creates OpenAI embeddings.
- Builds or loads a local FAISS vector index.
- Creates a retrieval tool.
- Builds the LangChain/OpenAI agent.
- Answers questions using the retrieved review context.

### `Rag_evaluation.py`

This file contains the evaluation logic.

It uses the files inside:

```text
Anotated_Game_Reviews_Data/data/test/
Anotated_Game_Reviews_Data/data/train/
```

Each split contains:

```text
questions.txt
reference_answers.txt
label_studio_import.json
```

It supports:

- Basic evaluation.
- Annotated evaluation.
- Comparison between basic and annotated evaluation.

### `run_evaluation.py`

This is the main file you run for evaluations.

It:

- Loads environment variables.
- Builds or loads the FAISS index.
- Builds the RAG agent.
- Runs the selected evaluation mode.

## Prerequisites

Before running the project, make sure you have:

1. Python 3.10 or newer installed.
2. An OpenAI API key.
3. The project downloaded or cloned from GitHub.
4. The review `.xlsx` files inside `Game_Reviews_Data/`.

## 1. Clone the Repository

```bash
git clone https://github.com/DanielVAppel/CS5170NLP_Assignment2_End-to-end-NLP-System-Building.git
cd CS5170NLP_Assignment2_End-to-end-NLP-System-Building
```

## 2. Create a Virtual Environment

### Windows PowerShell

Run this inside the project folder:

```powershell
python -m venv .venv
```

Activate the virtual environment:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

After activation, your terminal should show:

```text
(.venv)
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install the Requirements

With the virtual environment activated, run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If FAISS gives an installation issue, try:

```bash
python -m pip install faiss-cpu
```

## 4. Create the `.env` File

Create a file named:

```text
.env
```

Place it in the root project folder, at the same level as `Rag_Agent.py`.

Inside `.env`, add this line:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Example:

```env
OPENAI_API_KEY=sk-your-key-goes-here
```

Do not upload this `.env` file to GitHub because it contains your private API key.

## 5. Confirm the Data Folder Paths

Your review data should be here:

```text
Game_Reviews_Data/
```

Your evaluation data should be here:

```text
Anotated_Game_Reviews_Data/data/test/
Anotated_Game_Reviews_Data/data/train/
```

## 6. Run the RAG Agent Directly

To run the basic demo question inside `Rag_Agent.py`:

```bash
python Rag_Agent.py
```

On the first run, the program should:

1. Load the `.xlsx` files from `Game_Reviews_Data/`.
2. Convert the reviews into documents.
3. Split the documents into chunks.
4. Create OpenAI embeddings.
5. Build a FAISS vector index.
6. Save the index as `faiss_index/`.
7. Build the RAG agent.
8. Ask the demo question.

After the first run, the program should reuse the saved `faiss_index/` folder instead of rebuilding everything.

## 8. Run the Evaluation Script

The main evaluation file is:

```bash
python run_evaluation.py
```

By default, it should run basic evaluation on the test split.

You can also run it explicitly:

```bash
python run_evaluation.py --mode basic --split test
```

## Evaluation Commands

### Basic Evaluation on Test Split

```bash
python run_evaluation.py --mode basic --split test
```

### Basic Evaluation on Train Split

```bash
python run_evaluation.py --mode basic --split train
```

### Annotated Evaluation on Test Split

```bash
python run_evaluation.py --mode annotated --split test
```

### Annotated Evaluation on Train Split

```bash
python run_evaluation.py --mode annotated --split train
```

### Compare Basic vs. Annotated Evaluation on Test Split

```bash
python run_evaluation.py --mode compare --split test
```

### Compare Basic vs. Annotated Evaluation on Train Split

```bash
python run_evaluation.py --mode compare --split train
```

### Print More Detailed Results

Add `--verbose`:

```bash
python run_evaluation.py --mode compare --split test --verbose
```

## Rebuild the FAISS Index

If you change the source review `.xlsx` files or want to force the program to rebuild the vector database, run:

```bash
python run_evaluation.py --rebuild-index
```

You can combine it with other options:

```bash
python run_evaluation.py --mode compare --split test --rebuild-index
```

## Common Errors and Fixes

### Error related to NumPy dtype size

If you see an error involving `numpy.dtype size changed`, run:

```bash
python -m pip install numpy==1.26.4
```

### Encoding Issue on Windows

If a file crashes because of encoding issues, try:

```bash
python -X utf8 Rag_Agent.py
```

or:

```bash
python -X utf8 run_evaluation.py
```

## Quick Start Summary

For Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create `.env`:

Then run:

```powershell
# For Testing and debugging
python Rag_Agent.py
python run_evaluation.py --mode basic --split test
python run_evaluation.py --mode basic --split train
python run_evaluation.py --mode annotated --split test
python run_evaluation.py --mode annotated --split train

# For Final Run:
python run_evaluation.py --mode compare --split test
python run_evaluation.py --mode compare --split train
```