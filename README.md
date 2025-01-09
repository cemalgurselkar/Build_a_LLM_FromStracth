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
Python3.10.12

```bash

#For create a virtual enviroment and setup the required library
python3 -m venv venv
source venv/bin/activate
pip install requirement.txt

#Run Transformer.py
python3 Transformer.py

#Run gpt_2.py
python gpt_2.py
```