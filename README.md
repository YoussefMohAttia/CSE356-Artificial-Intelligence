<div align="center">

# AI Algorithms Collection

### Search, Adversarial Game Playing, and Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)

</div>

## Overview

This repository brings together three Artificial Intelligence implementations, each focused on a core method: informed search, adversarial search, and Markov decision processes.

## Projects

| Project | Topic | Key techniques |
| --- | --- | --- |
| Project 1 | 8-Puzzle solver | BFS, DFS, A*, Manhattan and Euclidean heuristics, Pygame interface |
| Project 2 | Connect Four | Game-tree search, solver implementation, timing analysis, GUI |
| Project 3 | Grid-world MDP | Value iteration and policy iteration under stochastic actions |

## Run the Projects

Set up a Python environment with the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pygame matplotlib
```

Then run an entry point from its own directory:

```bash
cd "Project 1"
python main.py

cd "../Project 2"
python game-window.py

cd "../Project 3"
python main.py
```

Project 1 includes a solver comparison script in `analyze.py`; Project 2 includes timing plots and analysis material.

## Repository Structure

```text
.
├── Project 1/     # Sliding-puzzle game and search algorithms
├── Project 2/     # Connect Four adversarial-search project
└── Project 3/     # Value and policy iteration experiment
```

## Author

**Youssef Mohamed Attia**  
[GitHub](https://github.com/YoussefMohAttia) · [LinkedIn](https://www.linkedin.com/in/youssefmohattia/)
