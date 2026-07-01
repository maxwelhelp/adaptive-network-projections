# First adaptive-rank run

Configuration:

```text
steps=5000
seeds=0,1,2
batch=128
width=384
true_rank=48
total_rank_budget=160
```

Result:

```text
seed 0: loss 0.7523 params 98496 ranks [24, 40, 40, 56]
seed 1: loss 0.7261 params 95936 ranks [24, 40, 32, 64]
seed 2: loss 0.7706 params 95936 ranks [24, 32, 40, 64]

mean loss = 0.7497 +/- 0.0182
```
