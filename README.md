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
```
@article{weinzierl-covid-glp,
	title        = {Automatic detection of COVID-19 vaccine misinformation with graph link prediction},
	author       = {Maxwell A. Weinzierl and Sanda M. Harabagiu},
	year         = 2021,
	journal      = {Journal of Biomedical Informatics},
	volume       = 124,
	pages        = 103955,
	doi          = {https://doi.org/10.1016/j.jbi.2021.103955},
	issn         = {1532-0464},
	url          = {https://www.sciencedirect.com/science/article/pii/S1532046421002847},
	keywords     = {Natural Language Processing, Machine learning, COVID-19, vaccine misinformation, Social Media, knowledge graph embedding}
}
```

- [CoVaxLies V1 with Graph-Link Prediction](<>)

###### COVID-19 Vaccine Misinformation Stance Identification on Twitter
```
@inproceedings{weinzierl-covid19-acs-stance,
    author = {Weinzierl, Maxwell and Harabagiu, Sanda},
    title = {Identifying the Adoption or Rejection of Misinformation Targeting COVID-19 Vaccines in Twitter Discourse},
    year = {2022},
    isbn = {9781450390965},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3485447.3512039},
    doi = {10.1145/3485447.3512039},
    booktitle = {Proceedings of the Web Conference 2022 Forthcoming},
    pages = {},
    numpages = {10},
    location = {Lyon, France},
    series = {WWW '22}
}
```

- [CoVaxLies V2 with Attitude Consistency Scoring](<>)

###### COVID-19 Misinformation Stance Identification on Twitter
```
@article{Weinzierl_Hopfer_Harabagiu_2021,
	title        = {Misinformation Adoption or Rejection in the Era of COVID-19},
	author       = {Weinzierl, Maxwell and Hopfer, Suellen and Harabagiu, Sanda M.},
	year         = 2021,
	month        = {May},
	journal      = {Proceedings of the International AAAI Conference on Web and Social Media},
	volume       = 15,
	number       = 1,
	pages        = {787--795},
	url          = {https://ojs.aaai.org/index.php/ICWSM/article/view/18103}
}
```


- [COVIDLies with Lexical, Emotion, and Semantic GATs for Stance Identification](<>)

###### Vaccine Misinformation Transfer Learning
```
@misc{weinzierl2022vaccinelies,
	title        = {VaccineLies: A Natural Language Resource for Learning to Recognize Misinformation about the COVID-19 and HPV Vaccines},
	author       = {Maxwell Weinzierl and Sanda Harabagiu},
	year         = 2022,
	eprint       = {2202.09449},
	archiveprefix = {arXiv},
	primaryclass = {cs.CL}
}
```

- [COVID-19 to HPV on VaccineLies](<>)
- [HPV to COVID-19 on VaccineLies](<>)

###### Vaccine Hesitancy Profiling on Twitter
```
@article{weinzierl-hesitancy-profiling,
	author = {Weinzierl, Maxwell A. and Hopfer, Suellen and Harabagiu, Sanda M.},
	title = {Scaling Up the Discovery of Hesitancy Profiles by Identifying the Framing of Beliefs towards Vaccine Confidence in Twitter Discourse},
	elocation-id = {2021.10.01.21264439},
	year = {2021},
	doi = {10.1101/2021.10.01.21264439},
	publisher = {Cold Spring Harbor Laboratory Press},
	URL = {https://www.medrxiv.org/content/early/2021/10/06/2021.10.01.21264439},
	journal = {medRxiv}
}
```

```
@misc{weinzierl2022hesitancy,
	title        = {From Hesitancy Framings to Vaccine Hesitancy Profiles: A Journey of Stance, Ontological Commitments and Moral Foundations},
	author       = {Maxwell Weinzierl and Sanda Harabagiu},
	year         = 2022,
	eprint       = {2202.09456},
	archiveprefix = {arXiv},
	primaryclass = {cs.CL}
}
```

- TODO
