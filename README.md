# Editing Factual Knowledge and Explanatory Ability of Medical Large Language Models

This is the code of the paper Editing Factual Knowledge and Explanatory Ability of Medical Large Language Models (Accepted by CIKM 2024).


## Environment Setup

    conda create -n medlasa python=3.9
    conda activate medlasa
    pip install -r requirement.txt



## Checkpoint
Downloading the checkpoints of LLMs from [Meditron-7B](https://github.com/epfLLM/meditron) and [ChatDoctor-7B](https://github.com/Kent0n-Li/ChatDoctor). Put the LLMs in `./../../LLM_checkpoint/`, or change the model_name to `YOUR_MODEL_PATH`.

## Usage
To obtain the casual tracing data, you can `cd ./rome/` follow the settings of [ROME](https://github.com/kmeng01/rome). You can also get processed casual tracing data from [Google Drive](https://drive.google.com/file/d/1kjL25SzNeTtcXtuOdnBCkZWFBoeb0sDS/view?usp=sharing). And put it in `./casual_tracing_data/`.

Runing:
```python
CUDA_VISIBLE_DEVICES=0 python mededit.py --method MedLaSA  --dataset MedFE --lora_ver LoRA --lr 5e-05  --alpha0 16 --rank0 24 --target_modules q_proj v_proj k_proj o_proj down_proj up_proj gate_proj --model_name chatdoctor

CUDA_VISIBLE_DEVICES=0 python mededit.py --method MedLaSA  --dataset MedCF --lora_ver LoRA --lr 1e-05  --alpha0 32 --rank0 24 --target_modules q_proj v_proj down_proj up_proj --model_name chatdoctor
```

