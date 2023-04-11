# Hyper-Recommendation

The research point of this project is multi-task learning in the recommendation system. By introducing preference vectors, multiple tasks can be optimized simultaneously and hope to achieve Pareto optimality. The basic idea is that the parameters of the Gate are provided by the hypernetwork. The input of the hypernetwork is a set of preference vectors (1×task_num), and the output is the weight of the Gate.

## Existing problems
- The preference vector cannot reach Pareto optimality, and behaves as a process of continuous Pareto improvement.

## Possible directions for improvement
- Increase the depth of the Gate
- Changing the Network Structure of Experts
- Replace Gate with attention mechanism

## Datasets
* AliExpressDataset: This is a dataset gathered from real-world traffic logs of the search system in AliExpress. This dataset is collected from 5 countries: Russia, Spain, French, Netherlands, and America, which can utilized as 5 multi-task datasets. [Original_dataset](https://tianchi.aliyun.com/dataset/dataDetail?dataId=74690) [Processed_dataset Google Drive](https://drive.google.com/drive/folders/1F0TqvMJvv-2pIeOKUw9deEtUxyYqXK6Y?usp=sharing) [Processed_dataset Baidu Netdisk](https://pan.baidu.com/s/1AfXoJSshjW-PILXZ6O19FA?pwd=4u0r)

> For the processed dataset, you should directly put the dataset in './data/' and unpack it. For the original dataset, you should put it in './data/' and run 'python preprocess.py --dataset_name NL'.

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
