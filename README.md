# nanoGPT Reimplement

Reimplement nanoGPT, a local trainable and finetunable agent, into object-oriented style with typing checks.

Instead of using the example `karpathy/chr_rnn/tinyshakespeare/input.txt`, this repo support multiple .txt files from the folder `input/` to be trained with.

## Transformer - model architecture

![Figure from the attention research paper](data/images/attention_research_figure.png)

## Getting Started

Create virtual environment and install python libraries:

```sh
uv venv
uv sync
```

Compile the repo as a module into the local uv env:

```sh
pip install -e .
```

## Run

<!-- [TODO] Add arg parsing -->

```sh
nanogpt
```

## Testing

```sh
pytest
```
