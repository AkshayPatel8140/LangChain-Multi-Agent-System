# LangChain Multi-Agent System
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-Enabled-yellowgreen)
![OpenAI](https://img.shields.io/badge/OpenAI-API-0072C6)

This Python application is a proof of concept for a basic multi-agent, multi-tool system using LangChain and OpenAI. Originally designed to demonstrate query classification and tool routing with improved robustness, safety, and usability.

## Table of Contents
- [Features](#features)
- [Setup](#setup)
- [Demo](#demo)
- [Coming Soon](#coming_soon)


## Features

- **Technical Agent:** Leverages LangChain, OpenAI, Tavily Search, and FAISS for handling technical queries.
- **Math Agent:** Specialized in math-related questions, equipped with a basic calculator and an equation solver tool.
- **LLM Classification:** Utilizes a language model (LLM) to classify user queries as 'math' or 'general/technical', directing them to the appropriate agent.
- **Embeddings & FAISS:** Employs OpenAI Embeddings and FAISS for efficient information retrieval.
- **Tavily Search:** Integrated as a search tool to enhance the agent's data access and processing capabilities.
- **LangSmith:** Used for monitoring, logging, and debugging to ensure smooth operation and maintenance of the system.


## Setup

### Dependencies

- LangChain
- OpenAI
- Tavily Search
- FAISS for information retrieval
- LangSmith for system monitoring

### How to Run

1. Use python venv `python -m venv .venv` & `source .venv/bin/activate`
2. Install the dependencies `pip install -r requirements.txt`
3. Set the necessary environment variables using a `.env` file.
4. Run the Python script using the command `python app.py`.

## Demo

Below are sample inputs demonstrating agent classification and routing. You can replace these with your own queries in the terminal when prompted.

| Query                               | Classification     | Agent           | Tool                   | Response                                                                 |
|-------------------------------------|---------------------|------------------|-------------------------|--------------------------------------------------------------------------|
| What's LangSmith?                   | Technical/General   | Technical Agent  | LangSmith Search Tool   | LangSmith is a tool developed by LangChain...                            |
| What's the weather in Lyon, France? | Technical/General   | Technical Agent  | Tavily Search Tool      | The weather in Lyon, France is currently partly cloudy...                |
| What's 3+2?                          | Math                | Math Agent       | Basic Calculator Tool   | The sum of 3 and 2 is 5.                                                |
| What's the solution to 3x+5=20?     | Math                | Math Agent       | Equation Solver Tool    | I apologize, but the equation solver feature is currently under development. |


## Coming_Soon

- Enhanced Equation Solver
- Web UI for Agent Conversations
- Agent Memory Sharing with Redis