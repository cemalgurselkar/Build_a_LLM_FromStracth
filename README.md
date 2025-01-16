# GPT-2 from Scratch

This project implements a GPT-2-like transformer-based language model entirely from scratch using PyTorch. The goal is to deeply understand the architecture and functionality of GPT-2 by building it step by step, with a focus on the decoder structure of the Transformer.


## Overview

GPT-2 (Generative Pretrained Transformer 2) is a transformer-based language model designed for generating coherent and contextually relevant text. This implementation focuses on understanding:
- Transformer decoder architecture.
- Autoregressive generation using causal masking.
- Weight tying between embedding and softmax layers.
- Efficient optimization techniques like AdamW.

This project uses numerical data for debugging and testing, making it easier to verify the correctness of the implementation.

## Features

- Fully custom implementation of GPT-2's decoder structure in PyTorch.
- Support for:
  - Causal masking to ensure autoregressive text generation.
  - Token and positional embeddings.
  - Layer normalization and residual connections.
- Customizable model configurations.
- Text generation with configurable temperature and top-k sampling.
- Optimizer setup with weight decay support.

## Requirements
Python Version i used = Python3.10.12

+ Note: i was using Cuda 12.1 version during writting and running the codes. If you use another version, you can install compatible version from orginal torch website.

+ How to use the repo?
```bash
git clone git@github.com:cemalgurselkar/build_gpt.git
```

```bash
#For create a virtual enviroment and setup the required library
python3 -m venv venv
source venv/bin/activate
# Download Torch with cuda121 version
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

#Enter to the folder with the code files:
cd app
```

+ if you want to run gpt_2.py:
```bash
python3 gpt_2.py
```

+ if you want to run Transformer.py:
```bash
python3 Transformer.py
```