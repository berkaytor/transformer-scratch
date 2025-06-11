A minimal implementation of a Transformer-based sequence-to-sequence model for English-Russian translation, built from scratch using PyTorch.

## Features

- Custom dataset loading and preprocessing
- Tokenization for English and Russian
- Transformer encoder-decoder architecture
- Training and evaluation scripts
- Configurable hyperparameters
- Easy monitoring with TensorBoard

## Project Structure

- `config.py` — Configuration and utility functions
- `dataset.py` — Dataset loading and preprocessing
- `model.py` — Transformer model implementation
- `train.py` — Training and validation logic
- `tokenizer_en.json`, `tokenizer_ru.json` — Pretrained tokenizers
- `requirements.txt` — Python dependencies

## Installation

Clone the repository and install dependencies:
```sh
git clone https://github.com/yourusername/transformer-scratch.git
cd transformer-scratch
pip install -r requirements.txt
```

## Usage

Train the model:
```sh
python train.py
```

You can adjust hyperparameters and paths in `config.py`.

## Dataset

Prepare your parallel English-Russian dataset as two text files:
- `data/train.en` — English sentences (one per line)
- `data/train.ru` — Russian sentences (aligned, one per line)

Update dataset paths in `config.py` if needed.

## Monitoring

Monitor training progress with TensorBoard:
```sh
tensorboard --logdir runs
```

## Configuration

All training and model parameters can be set in `config.py`, including:
- Batch size
- Learning rate
- Number of epochs
- Model dimensions

## Results

After training, model checkpoints and logs will be saved in the `checkpoints/` and `runs/` directories.

## Acknowledgements

- Based on the "Attention is All You Need" paper.
- Inspired by PyTorch and open-source NLP projects.

## License

MIT License