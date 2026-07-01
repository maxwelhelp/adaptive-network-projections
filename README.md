# Adaptive Network Projections

Experiments on **backpropagation through projected neural networks**.

The idea is simple: do not replace backpropagation and do not try to guess the gradient. Instead, replace the network computation itself with a cheaper learnable projection, and let normal backpropagation flow through that projected computation.

```text
Dense layer:
    y = W x

Projected layer:
    y = U (V x)
    W ~= U V

Adaptive projected layer:
    y = U_r (V_r x)
    rank r grows where gradient signals show that the layer needs more projection capacity
```

The gradient is used twice:

1. it trains the task;
2. it pushes the projection parameters and projection structure toward subspaces that reduce loss.

This repository is for collecting variants of the idea: fixed low-rank projection, adaptive rank growth, block-coordinate updates, affine local approximations, routed projections, and future projected-computation experiments.

## Current experiment

`experiments/adaptive_lowrank_only.py`

A clean adaptive-rank low-rank network experiment.

What it tests:

- teacher network with controlled high-rank layers;
- teacher output normalization, so MSE is interpretable;
- student network where each dense layer is replaced by a low-rank factorization;
- adaptive rank growth based on gradient norms;
- total rank budget, so long runs cannot simply grow every layer to max rank;
- corrected evaluation: `net(x)` is compared with the matching `y` from the same batch.

Run:

```bash
python experiments/adaptive_lowrank_only.py --steps 5000 --seeds 0 1 2 --batch 128 --eval-batches 2 --final-eval-batches 12
```

First checked result:

```text
steps=5000, seeds=[0,1,2], width=384, true_rank=48, total_rank_budget=160

seed 0: loss 0.7523  params 98496  ranks [24, 40, 40, 56]
seed 1: loss 0.7261  params 95936  ranks [24, 40, 32, 64]
seed 2: loss 0.7706  params 95936  ranks [24, 32, 40, 64]

mean loss = 0.7497 +/- 0.0182
```

Interpretation:

- zero predictor after teacher normalization is around MSE `1.0`;
- adaptive projected network reaches around `0.73-0.77`, so it learns non-trivial structure;
- it uses about `96k-98k` parameters instead of a comparable dense student with about `345k` parameters;
- ranks are not uniform, which suggests that gradient signals allocate projection capacity differently across layers.

## Roadmap

Planned variants:

- fixed low-rank vs adaptive rank comparison;
- adaptive gates instead of hard rank growth;
- block-coordinate updates on top of low-rank factors;
- affine local layer approximations;
- routed projection families;
- GPU timing and memory benchmarks;
- comparison against dense baselines with tuned learning rates;
- result tables for loss / params / speed / memory.

---

# Адаптивные проекции сети

Эксперименты по **backprop через спроецированную нейросеть**.

Идея простая: не заменять backprop и не угадывать градиент. Вместо этого мы заменяем сами вычисления сети на более дешёвую обучаемую проекцию, и обычный backprop проходит через эту проекцию.

```text
Полный слой:
    y = W x

Спроецированный слой:
    y = U (V x)
    W ~= U V

Адаптивный спроецированный слой:
    y = U_r (V_r x)
    rank r растёт там, где градиент показывает, что слою нужна большая ёмкость проекции
```

Градиент используется дважды:

1. он учит задачу;
2. он толкает параметры и структуру проекции к тем подпространствам, которые уменьшают loss.

Репозиторий нужен, чтобы собирать разные варианты идеи: fixed low-rank, адаптивный рост ранга, block-coordinate обновления, аффинные локальные приближения, routed projections и другие варианты спроецированного вычисления.

## Текущий эксперимент

`experiments/adaptive_lowrank_only.py`

Чистый тест adaptive-rank low-rank сети.

Что проверяет:

- teacher-сеть с контролируемым высоким рангом слоёв;
- нормализация выхода teacher, чтобы MSE был понятным;
- student-сеть, где каждый dense layer заменён low-rank факторизацией;
- адаптивный рост ранга по нормам градиента;
- общий бюджет ранга, чтобы на длинном запуске все слои не доросли тупо до максимума;
- исправленная оценка: `net(x)` сравнивается с соответствующим `y` из того же batch.

Запуск:

```bash
python experiments/adaptive_lowrank_only.py --steps 5000 --seeds 0 1 2 --batch 128 --eval-batches 2 --final-eval-batches 12
```

Первый проверенный результат:

```text
steps=5000, seeds=[0,1,2], width=384, true_rank=48, total_rank_budget=160

seed 0: loss 0.7523  params 98496  ranks [24, 40, 40, 56]
seed 1: loss 0.7261  params 95936  ranks [24, 40, 32, 64]
seed 2: loss 0.7706  params 95936  ranks [24, 32, 40, 64]

mean loss = 0.7497 +/- 0.0182
```

Интерпретация:

- после нормализации teacher нулевое предсказание даёт MSE около `1.0`;
- adaptive projected network доходит примерно до `0.73-0.77`, то есть реально учит структуру;
- использует около `96k-98k` параметров вместо примерно `345k` у сопоставимого dense student;
- ранги не одинаковые, значит градиент распределяет ёмкость проекции по слоям не вручную, а по сигналу обучения.

## План

Дальше добавим:

- fixed low-rank vs adaptive rank;
- adaptive gates вместо жёсткого роста ранга;
- block-coordinate поверх low-rank факторов;
- аффинные локальные приближения слоёв;
- routed projection families;
- GPU timing и memory benchmarks;
- сравнение с dense baseline с tuned learning rate;
- таблицы loss / params / speed / memory.
