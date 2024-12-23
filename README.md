# Transformer and GPT Model

This repository implements a GPT model from scratch using the Transformer architecture based on the *"Attention Is All You Need"* paper. The model is inspired by Andrej Karpathy's GPT video, with the key difference being the usage of an improved Transformer architecture derived directly from the *"Attention Is All You Need"* paper.

---
## Features

- **Transformer Architecture**: Fully implemented Encoder-Decoder structure with Multi-Head Attention, Positional Encoding, and Feed Forward layers.
- **GPT Model**: Implements causal attention for autoregressive text generation.
- **Custom Implementation**: Both the Transformer and GPT models are implemented from scratch using PyTorch.

---

## Requirements

Install the required dependencies:
```bash
#Create a virtual env. 
python3 -m venv venv
source venv/bin/activate

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 # to download torch 

python3 Transformer.py # run the Transformer.py file
```
## Note:
**The gpt_2.py file is not running. It will be fix then.**
### Error Message:
**The size of tensor a (384) must match the size of tensor b (256) at non-singleton dimension 4**