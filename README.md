Code for difficulty-aware curriculum and token-weighted rationale distillation for sequential recommendation.
 
 # Raw Datasets

**MovieLens Datasets**: The origin dataset can be found [here](https://grouplens.org/datasets/movielens/).

**Amazon Datasets**: The origin dataset can be found [here](https://jmcauley.ucsd.edu/data/amazon/).

# Run Commands

```bash
# preprocess + CoT
python run.py --dataset ml100k --cot cluster
```

```bash
# main
python main.py --stage all --dataset ml100k --online-wg
```

```bash
# Evaluate
python main_test.py --dataset ml100k 
```
