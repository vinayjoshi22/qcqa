### Note: This is only reproduction effort and not an official release!
This is the code to reproduce the results from the paper `QCQA: Quality and Capacity-aware grouped Query Attention` [https://arxiv.org/abs/2406.10247].

A sample command to run on a huggingface model is as follows:
 - `python expt.py --model facebook/opt-125m --num_heads 12 --num_layers 12 --num_groups 6 --KV_parse_strings 'model.decoder.layers.{}.self_attn.k_proj.weight;model.decoder.layers.{}.self_attn.v_proj.weight' --n_gen 10`

### Installation
- No special installation is required.

### Dependancies (python packages)
- Numpy
- Pymoo 
- Pandas
- transformers

### Current support
1. Algorithm 1 implementation from the paper.
2. Algorithm 2 implementation from the paper.

### TODO:
Stay tuned for the following updates!
1. Add support for multi-processing for parallel execution!
1. Add support for percentile-based collation of results.
1. Add torchtune support for evaluating and finetuning LLMs.
1. (Bonus) add support for Huggingface API for evaluation and finetuning LLMs.
1. Add support for long context tasks.
1. Add support for integration with other KV-cache methods for token-based compression (e.g., H2O).
