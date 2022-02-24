<div align="center">

<img src="docs/images/banner.png?raw=true" width="400px">

**A Social Media Natural Language Processing package for PyTorch & PyTorch Lightning.**

______________________________________________________________________

<p align="center">
  <a href="#pytorch-gleam">Key Features</a> •
  <a href="#about-me">About Me</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#examples">Examples</a>
</p>

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pytorch-gleam)](https://pypi.org/project/pytorch-gleam/)
[![PyPI Status](https://badge.fury.io/py/pytorch-gleam.svg)](https://badge.fury.io/py/pytorch-gleam)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Supermaxman/pytorch-gleam/blob/master/LICENSE.txt)

</div>

______________________________________________________________________

## PyTorch Gleam

PyTorch Gleam builds upon [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
for the specific use-case of Natural Language Processing on Social Media, such as Twitter.
PyTorch Gleam strives to make Social Media NLP research easier to understand, use, and extend.
Gleam contains models I use in my research, from fine-tuning a BERT-based model with Lexical, Emotion, and Semantic
information in a Graph Attention Network for stance identification towards COVID-19 misinformation, to
using Information Retrieval systems to identify new types of misinformation on Twitter.

## About Me

My name is [Maxwell Weinzierl](https://personal.utdallas.edu/~maxwell.weinzierl/), and I am a
Natural Language Processing researcher at the Human Technology Research Institute (HLTRI) at the
University of Texas at Dallas. I am currently working on my PhD, which focuses on COVID-19 and
HPV vaccine misinformation, trust, and more on Social Media platforms such as Twitter. I have built
PyTorch Gleam to enable easy reproducibility for my published research, and for my own quick
iterations on research ideas.

## How To Use

### Step 0: Install

Simple installation from PyPI

```bash
pip install pytorch-gleam
```

You may need to install CUDA drivers and other versions of PyTorch.
See [PyTorch](https://pytorch.org/get-started/locally/) and
[PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning#how-to-use)
for installation help.

### Step 1: Create Experiment

Create a `configs` folder with a YAML experiment file. Gleam utilizes PyTorch Lightning's CLI tools
to configure experiments from YAML files, which enables researchers to clearly look back
and identify both hyper-parameters and model code used in their experiments.
This example is from COVID-19 vaccine misinformation stance identification:

[pg_examples/covid-stance.yaml](https://github.com/Supermaxman/pytorch-gleam/tree/master/pg_examples)

```yaml
seed_everything: 0
model:
  class_path: pytorch_gleam.modeling.models.MultiClassFrameLanguageModel
  init_args:
    learning_rate: 5e-4
    pre_model_name: digitalepidemiologylab/covid-twitter-bert-v2
    label_map:
      No Stance: 0
      Accept: 1
      Reject: 2
    threshold:
      class_path: pytorch_gleam.modeling.thresholds.MultiClassThresholdModule
    metric:
      class_path: pytorch_gleam.modeling.metrics.F1PRMultiClassMetric
      init_args:
        mode: macro
        num_classes: 3
trainer:
  max_epochs: 10
  accumulate_grad_batches: 4
  check_val_every_n_epoch: 1
  deterministic: true
  num_sanity_val_steps: 1
  checkpoint_callback: false
  callbacks:
    - class_path: pytorch_gleam.callbacks.FitCheckpointCallback
data:
  class_path: pytorch_gleam.data.datasets.MultiClassFrameDataModule
  init_args:
    batch_size: 8
    max_seq_len: 128
    label_name: misinfo
    label_map:
      No Stance: 0
      Accept: 1
      Reject: 2
    tokenizer_name: digitalepidemiologylab/covid-twitter-bert-v2
    num_workers: 8
    frame_path:
      - covid19/misinfo.json
    train_path:
      - covid19/stance-train.jsonl
    val_path:
      - covid19/stance-dev.jsonl
    test_path:
      - covid19/stance-test.jsonl
```

Documentation on available `models`, `datasets`, and `callbacks`
will be provided soon.

Details about how to set up YAML experiment files are provided by
PyTorch Lightning's [documentation](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_cli.html).

Annotations for this example are provided in the *VaccineLies* repository under covid19 as the *CoVaxLies* collection:
[CoVaxLies](https://github.com/Supermaxman/vaccine-lies/tree/master/covid19).
You will need to download the tweet texts from the tweet ids from the Twitter API.

### Step 2: Run Experiment

Create a `models` folder for your saved [TensorBoard](https://www.tensorflow.org/tensorboard) logs and model weights.
Determine the GPU ID for the GPU you would like to utilize (multi-gpu supported) and provide the ID in a list, with
a comma at the end if it is a single GPU ID. You can also just specify an integer, such as `1`, and PyTorch Lightning
will try to find a single free GPU automatically.
Run the following command to start training:

```bash
gleam fit \
  --config configs/covid-stance.yaml \
  --trainer.gpus 1 \
  --trainer.default_root_dir models/covid-stance
```

Your model will train, with [TensorBoard](https://www.tensorflow.org/tensorboard) logging all metrics, and a checkpoint will be saved upon completion.

### Step 3: Evaluate Experiment

You can easily evaluate your system on a test collection as follows:

```bash
gleam test \
  --config configs/covid-stance.yaml \
  --trainer.gpus 1 \
  --trainer.default_root_dir models/covid-stance
```

______________________________________________________________________

## Examples

These are a work-in-progress, as my original research code is a bit messy, but they will be updated soon!

###### COVID-19 Vaccine Misinformation Detection on Twitter

- [CoVaxLies V1 with Graph-Link Prediction](<>)

###### COVID-19 Vaccine Misinformation Stance Identification on Twitter

- [CoVaxLies V2 with Attitude Consistency Scoring](<>)

###### COVID-19 Misinformation Stance Identification on Twitter

- [COVIDLies with Lexical, Emotion, and Semantic GATs for Stance Identification](<>)

###### Vaccine Misinformation Transfer Learning

- [COVID-19 to HPV on VaccineLies](<>)
- [HPV to COVID-19 on VaccineLies](<>)

###### Vaccine Hesitancy Profiling on Twitter

- TODO
