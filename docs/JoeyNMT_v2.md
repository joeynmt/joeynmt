# Joey NMT

Joey NMT is a minimalist neural machine translation toolkit for educational purposes.

It aims to be a **clean** and **minimalistic** code base to help novices pursuing the understanding of neural machine translation.

## Installation

Joey NMT is built on [PyTorch](https://pytorch.org/). Please make sure you have a compatible environment.
We tested Joey NMT v2.3 with
- python 3.11
- torch 2.1.2
- cuda 12.1

> **Warning**  
> When running on **GPU**, you need to manually install the suitable PyTorch version 
> for your [CUDA](https://developer.nvidia.com/cuda-zone) version.
> See [PyTorch installation instructions](https://pytorch.org/get-started/locally/).

You can install Joey NMT either A. via [pip](https://pypi.org/project/joeynmt/) or B. from source.

### A. Via pip (the latest stable version)
```bash
python -m pip install joeynmt
```

### B. From source (for local development)
```bash
git clone https://github.com/joeynmt/joeynmt.git  # Clone this repository
cd joeynmt
python -m pip install -e .  # Install Joey NMT and it's requirements
python -m unittest  # Run the unit tests
```

## Usage

For details, check out our [documentation](https://joeynmt.readthedocs.io)!

### 1. Torch Hub
```python

import torch

model = torch.hub.load("joeynmt/joeynmt", "wmt14_ende")
translations = model.translate(["Hello world!"])
print(translations[0])  # "Hallo Welt!"
```

### 2. Command-line Interface

```
python -m joeynmt {train,test,translate} CONFIG_PATH [-o OUTPUT_PATH] [-a] [-s] [-t] [-d]

positional arguments:
  {train,test,translate}    Train a model or test or translate
  CONFIG_PATH               Path to YAML config file

options:
  -h, --help                show this help message and exit
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                            Path for saving translation output
  -a, --save-attention      Save attention visualizations
  -s, --save-scores         Save scores
  -t, --skip-test           Skip test after training
  -d, --use-ddp             Invoke DDP environment
```

For example:

- training
    ```bash
    python -m joeynmt train configs/transformer_small.yaml --use_ddp --skip-test
    ```

- testing
    ```bash
    python -m joeynmt test configs/transformer_small.yaml --output-path model_dir/hyp --save-scores --save-attention
    ```

- translation
    ```bash
    python -m joeynmt translate configs/transformer_small.yaml < input.txt > output.txt
    ```


## Citation

If you use Joey NMT in a publication or thesis, please cite the following [paper](https://arxiv.org/abs/1907.12484):

```
@inproceedings{kreutzer-etal-2019-joey,
    title = "Joey {NMT}: A Minimalist {NMT} Toolkit for Novices",
    author = "Kreutzer, Julia  and
      Bastings, Jasmijn  and
      Riezler, Stefan",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP): System Demonstrations",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-3019",
    doi = "10.18653/v1/D19-3019",
    pages = "109--114",
}
```
