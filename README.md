# GPT-2 From Scratch

> End-to-end implementation of the 120 M-parameter GPT-2 model in PyTorch, trained from scratch on WikiText-2, with custom BPE tokenizer, CUDA kernel optimizations, cloud training & hosting, and a math-problem-solving extension.

---

## Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Prerequisites](#prerequisites)  
- [Architecture](#architecture)  
- [Setup](#setup)  
  - [1. Provision EC2 (Phase 1.1)](#1-provision-ec2-phase-11)  
  - [2. Install Miniconda & PyTorch (Phase 1.2)](#2-install-miniconda--pytorch-phase-12)  
  - [3. Implement BPE Tokenizer (Phase 1.3)](#3-implement-bpe-tokenizer-phase-13)  
- [Development](#development)  
  - [Prototype Training (3-layer)](#prototype-training-3-layer)  
  - [Full-scale Training](#full-scale-training)  
- [Optimization](#optimization)  
- [Extension: Math Problem Solving](#extension-math-problem-solving)  
- [Deployment](#deployment)  
- [Usage](#usage)  
- [Project Structure](#project-structure)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Overview

This repository walks you through **building GPT-2 from scratch**:

1. **Phase 1:** Environment setup (AWS EC2, Conda, PyTorch, BPE tokenizer)  
2. **Phase 2:** Model architecture (Transformer blocks, 12 layers, 12 heads)  
3. **Phase 3:** Training loop & data pipeline (Adam, scheduler, checkpointing)  
4. **Phase 4:** CUDA kernel fusion for softmax + layer-norm  
5. **Phase 5:** Full-scale training on spot GPUs (target perplexity ~27.5)  
6. **Phase 6:** Dockerized deployment (AWS SageMaker / REST API)  
7. **Phase 7:** Math-problem-solving fine-tune & tool-use integration  

---

## Features

- 🔤 **Custom BPE Tokenizer** built from first principles  
- 🔧 **Transformer Implementation** in pure PyTorch  
- 🏎 **CUDA Kernel** for fused softmax + layer-norm (~1.3× speed-up)  
- ☁️ **Cloud Training** on AWS spot instances  
- 🔢 **Math Solver Extension** using chain-of-thought & Sympy/Calculator hooks  
- 🐳 **Dockerized** for easy deployment  

---

## Prerequisites

- **AWS Account** with EC2 & IAM permissions  
- **AWS CLI** configured (region `us-east-1`)  
- **Linux environment** (WSL Ubuntu or native Linux)  
- **Docker** & **Docker Compose** (for deployment)  
- **Python 3.9+** (managed via Conda)  
