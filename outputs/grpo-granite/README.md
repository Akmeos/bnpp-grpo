---
base_model: ibm-granite/granite-3.1-1b-a400m-instruct
library_name: peft
model_name: grpo-granite
tags:
- base_model:adapter:ibm-granite/granite-3.1-1b-a400m-instruct
- grpo
- lora
- transformers
- trl
licence: license
pipeline_tag: text-generation
---

# Model Card for grpo-granite

This model is a fine-tuned version of [ibm-granite/granite-3.1-1b-a400m-instruct](https://huggingface.co/ibm-granite/granite-3.1-1b-a400m-instruct).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 


This model was trained with GRPO, a method introduced in [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

### Framework versions

- PEFT 0.17.1
- TRL: 0.23.0
- Transformers: 4.56.1
- Pytorch: 2.7.0
- Datasets: 4.0.0
- Tokenizers: 0.22.0

## Citations

Cite GRPO as:

```bibtex
@article{shao2024deepseekmath,
    title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
    author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
    year         = 2024,
    eprint       = {arXiv:2402.03300},
}

```

Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```