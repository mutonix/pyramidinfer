# PyramidInfer: Pyramid KV Cache Compression for High-throughput LLM Inference

Dongjie Yang, Xiaodong Han, Yan Gao, Yao Hu, Shilin Zhang, Hai Zhao
[![arXiv](https://img.shields.io/badge/arXiv-2110.06707-b31b1b.svg)](https://arxiv.org/abs/2405.12532)

## Updates
- [2024-06-17] We release the code for PyramidInfer.

[WIP] This repository is still under construction. We will release the full code to evaluate the performance of PyramidInfer using the OpenCompass.

## Overview

Large Language Models (LLMs) have shown remarkable comprehension abilities but face challenges in GPU memory usage during inference, hindering their scalability for real-time applications like chatbots. To accelerate inference, we store computed keys and values (KV cache) in the GPU memory. Existing methods study the KV cache compression to reduce memory by pruning the pre-computed KV cache. However, they neglect the inter-layer dependency between layers and huge memory consumption in pre-computation. To explore these deficiencies, we find that the number of crucial keys and values that influence future generations decreases layer by layer and we can extract them by the consistency in attention weights. Based on the findings, we propose PyramidInfer, a method that compresses the KV cache by layer-wise retaining crucial context. PyramidInfer saves significant memory by computing fewer keys and values without sacrificing performance. Experimental results show PyramidInfer improves 2.2x throughput compared to Accelerate with over 54% GPU memory reduction in KV cache.

<img src = "assets/pyramidinfer.png" align = "center" width="100%" hight="100%">

## Getting Started

### run a demo
We recommend using the PyramidInfer with a large batch size to see more significant memory reduction and efficiency improvement. 
```
conda create -n pyramidinfer python=3.8 -y
conda activate pyramidinfer
pip install -r requirements.txt

python simple_infer_comparison.py --model_name_or_path meta-llama/Llama-2-7b-hf
```

## Citation
```
@misc{yang2024pyramidinfer,
      title={PyramidInfer: Pyramid KV Cache Compression for High-throughput LLM Inference}, 
      author={Dongjie Yang and XiaoDong Han and Yan Gao and Yao Hu and Shilin Zhang and Hai Zhao},
      year={2024},
      eprint={2405.12532},
      archivePrefix={arXiv},
      primaryClass={id='cs.CL' full_name='Computation and Language' is_active=True alt_name='cmp-lg' in_archive='cs' is_general=False description='Covers natural language processing. Roughly includes material in ACM Subject Class I.2.7. Note that work on artificial languages (programming languages, logics, formal systems) that does not explicitly address natural-language issues broadly construed (natural-language processing, computational linguistics, speech, text retrieval, etc.) is not appropriate for this area.'}
}
```