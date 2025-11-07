# nanoGPT Reimplement

Reimplement nanoGPT, a local trainable and finetunable agent, into object-oriented style with typing checks.

Instead of using the example `karpathy/chr_rnn/tinyshakespeare/input.txt`, this repo support multiple .txt files from the folder `input/` to be trained with.

## Transformer - model architecture

> Deviation: Unlike the `Add & Norm` after each layer of the original paper, this repo implements pre-norm formulation.

![Figure from the attention research paper](data/images/attention_research_figure.png)

## Getting Started

Create virtual environment and install python libraries:

```sh
uv venv
uv sync
```

Compile this repo as a module on the local:

```sh
pip install -e .
```

## Train and Validate

<!--
[TODO] Add arg parsing 

```sh
nanogpt train
    --device=cpu
    --eval_iters=20
    --log_interval=1
    --block_size=64
    --batch_size=12
    --n_layer=4
    --n_head=4
    --n_embd=128
    --max_iters=2000
    --lr_decay_iters=2000
    --dropout=0.0
```
-->

```sh
nanogpt
```

## Sample

## Finetune

<!-- TODO: Extract finetuning hyperparameters -->

## Testing

```sh
pytest .
```
