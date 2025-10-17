## DPDiff

### Environment

```shell
conda env create -f dpdiff.yaml
conda activate dpdiff
```

### Data and Preparation
The data preparation follows [TargetDiff](https://arxiv.org/abs/2303.03543). For more details, please refer to [the repository of TargetDiff](https://github.com/guanjq/targetdiff?tab=readme-ov-file#data).

### Training

```shell
conda activate dpdiff
python train.py
```

### Sampling

```shell
python sample_split.py --start_index 0 --end_index 99 --batch_size 25
```

### Evaluation

```shell
python eval_split.py --eval_start_index 0 --eval_end_index 99
```

### Calculate metrics

```shell
python cal_metrics_from_pt.py
```

### Acknowledgement
Our code is adapted from the repository of [TargetDiff](https://github.com/guanjq/targetdiff). We thank the authors for sharing their code.
