# top-m-arm-selection-non-linear

A repository containing code for top-m example subsets selection for in-context learning. The repo contains the gap-index based algorithm which employs a differentiable sorting surrogate to 
learn to identfy top-m subsets based on LLM feedback as rewards.

# Setup
1. Install ollama.
2. pull llama3.2:3b ```ollama pull llama3.2:3b```


# To reproduce WMT19 numbers in Table 1 (evaluation part only):
Run
```python3 Code/evaluation/eval_wmt19.py
```
