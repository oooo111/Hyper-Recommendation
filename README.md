# Hyper-Recommendation

## Existing problems
- The preference vector cannot reach Pareto optimality, and behaves as a process of continuous Pareto improvement.

## Possible directions for improvement
- Increase the depth of the Gate
- Changing the Network Structure of Experts
- Replace Gate with attention mechanism

## Train

```
python main.py --model_name pomoe --dataset_name AliExpress_US
```

## Dependencies

- Python>=3.8
- gurobipy==9.5.2
- matplotlib==3.5.1
- numpy==1.22.3
- ortools==9.4.1874
- scikit_learn==1.1.2
- scipy==1.8.0
- setuptools==58.0.4
- six==1.16.0
- tensorboard_logger==0.1.0
- tensorflow==2.10.0
- torch==1.11.0+cu113
- tqdm==4.62.3

## Acknowledgements
This code is originally implemented based on [Multitask-Recommendation-Library](https://github.com/easezyc/Multitask-Recommendation-Library) and [cosmos](https://github.com/ruchtem/cosmos/blob/c198830d50575065f0dd31f8fcd5390b720bab98/multi_objective/methods/pareto_hypernet/phn_wrappers.py)
