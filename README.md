# SEALR 

A repository containing code for top-m example subsets selection for in-context learning. The repo contains the gap-index based algorithm which employs a differentiable sorting surrogate to 
learn to identfy top-m subsets based on LLM feedback as rewards.

# Setup
1. Install ollama.
2. pull llama3.2:3b ```ollama pull llama3.2:3b```
3. ollama serve &
4. ollama run llama 3.2:3b


# To reproduce WMT19 numbers in Table 1 (evaluation part only):
Run
```
python3 Code/evaluation/eval_wmt19.py
```



# To reproduce AQUA numbers in Table 1 (evaluation part only):
Run
```
python3 Code/evaluation/eval_aqua.py
```

# To reproduce GSM8K numbers in Table 1 (evaluation part only):
Run
```
python3 Code/evaluation/eval_gsm8k.py
```

# To run the bandit algorithms from scratch 
For Aqua-RAT run:

``` 
python3 Code/bandit_run/Sealr_aqua_final_ICLR.py
```
