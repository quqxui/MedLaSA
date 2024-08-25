# import hydra
import sklearn
from easyeditor import BaseEditor
from easyeditor import  FTHyperParams, ROMEHyperParams, MEMITHyperParams, MENDTrainingHparams, MENDHyperParams,  LoRAHyperParams,  PMETHyperParams, MedLaSAHyperParams

from transformers import LlamaTokenizer
import numpy as np
from easyeditor import CounterFactDataset
from easyeditor import EditTrainer
import argparse
from collections import defaultdict
import json
import math

def metrics_compute(path):
    if isinstance(path, str):
        metrics = json.load(open(path, 'r', encoding='utf-8'))
    else:
        metrics = path
    results = defaultdict(list)

    for metric in metrics:
        for k,v in metric['post'].items():
            if k=='rewrite_acc' or k=='rephrase_acc': 
                results[k].append(v[0])
            if k=='fluency':
                results['fluency'].append(v['ngram_entropy'])
            if k=='locality' or k == 'portability':
                for k2,v2 in v.items():
                    results[k2].append(v2[0])
    print('------------------------------')
    result_list = []
    edit_success = []
    local_success = []
    for k,v in results.items():
        v = sum(v)/len(v)
        v = v * 100
        print(f"{k} : {v}")
        if k=='rewrite_acc' or k =='rephrase_acc':
            edit_success.append(v)
        if 'locality' in k:
            local_success.append(v)            
        result_list.append(round(v, 2))
    avg = (sum(edit_success)/len(edit_success) + sum(local_success)/len(local_success)) / 2
    print(f"avg : {avg}")
    result_list.append(round(avg, 2))
    print(result_list)

def MedCF_load(args):
    edit_data = json.load(open('./../MedCF/test.json', 'r', encoding='utf-8'))
    if args.sampledata:
        edit_data = edit_data[:args.sampledata]
    prompts = [edit_data_['prompt'] for edit_data_ in edit_data]
    subject = [edit_data_['subject'] for edit_data_ in edit_data]
    rephrase_prompts = [edit_data_['rephrase_prompt'] for edit_data_ in edit_data]
    target_new = [edit_data_['target_new'] for edit_data_ in edit_data]
    ground_truth = [edit_data_['ground_truth'] for edit_data_ in edit_data]
    locality_inputs = {
        'locality_target':{
            'prompt': [edit_data_['locality_target_prompt'] for edit_data_ in edit_data],
            'ground_truth': [edit_data_['locality_target_ground_truth'] for edit_data_ in edit_data]
        },
        'locality_mapping':{
            'prompt': [edit_data_['locality_mapping_prompt'] for edit_data_ in edit_data],
            'ground_truth': [edit_data_['locality_mapping_ground_truth'] for edit_data_ in edit_data]
        },
        'locality_struc':{
            'prompt': [edit_data_['locality_struc_prompt'] for edit_data_ in edit_data],
            'ground_truth': [edit_data_['locality_struc_ground_truth'] for edit_data_ in edit_data]
        },
        'locality_tokenSem':{
            'prompt': [edit_data_['locality_tokenSem_prompt'] for edit_data_ in edit_data],
            'ground_truth': [edit_data_['locality_tokenSem_ground_truth'] for edit_data_ in edit_data]
        },
    }
    return {'prompts':prompts, 'rephrase_prompts':rephrase_prompts, 'target_new':target_new, 'ground_truth':ground_truth,'subject':subject, 'locality_inputs':locality_inputs, 'portability_inputs':None}

def MedFE_load(args):
    tokenizer = LlamaTokenizer.from_pretrained("./../../LLM_checkpoint/chatdoctor-llama")
    def clip(text):
        tokens = tokenizer.tokenize(text)
        if len(tokens) > args.max_length:
            tokens = tokens[:args.max_length]
            text = tokenizer.convert_tokens_to_string(tokens)
        return text
    
    edit_data = json.load(open('./../MedFE/test.json', 'r', encoding='utf-8'))
    if args.sampledata:
        edit_data = edit_data[:args.sampledata]
    prompts = [edit_data_['prompt'] for edit_data_ in edit_data]
    subject = [edit_data_['subject'] for edit_data_ in edit_data]
    subject_type = [edit_data_['subject_type'] for edit_data_ in edit_data]
    topic_type = [edit_data_['topic_type'] for edit_data_ in edit_data]
    rephrase_prompts = [edit_data_['rephrase_prompt'] for edit_data_ in edit_data]
    target_new = [clip(edit_data_['target_new']) for edit_data_ in edit_data]
    locality_inputs = {
        'locality_topic':{
            'prompt': [edit_data_['locality_topic_prompt'] for edit_data_ in edit_data],
            'ground_truth': [clip(edit_data_['locality_topic_ground_truth']) for edit_data_ in edit_data]
        },
        'locality_tokenSem':{
            'prompt': [edit_data_['locality_tokenSem_prompt'] for edit_data_ in edit_data],
            'ground_truth': [clip(edit_data_['locality_tokenSem_ground_truth']) for edit_data_ in edit_data]
        },
    }
    return {'prompts':prompts, 'rephrase_prompts':rephrase_prompts, 'target_new':target_new, 'ground_truth':None, 'subject':subject, 'locality_inputs':locality_inputs,'portability_inputs':None}



def Args():
    parser = argparse.ArgumentParser(description='Example script with argparse')
    # Add arguments
    parser.add_argument('--trainedit', type=str,default='edit',)
    parser.add_argument('--method', type=str,default='FT', )
    parser.add_argument('--device', type=int,default=0, )
    parser.add_argument('--sampledata', type=int,default=0,)
    parser.add_argument('--max_length', type=int,default=200,)
    parser.add_argument('--dataset', type=str,default='MedCF',) # MedCF MedFE
    parser.add_argument('--lora_ver', type=str,default=None,)
    parser.add_argument('--lr', type=float, default=None,)
    parser.add_argument('--num_steps', type=int, default=70,)
    parser.add_argument("--alpha_dynamic", type=bool, default=True)
    parser.add_argument("--rank_dynamic", type=bool, default=True)
    parser.add_argument('--alpha0', type=float, default=64.0,)
    parser.add_argument('--rank0', type=int, default=16,)
    parser.add_argument('--norm', type=str, default='minmax',) # meanstd minmax
    parser.add_argument('--target_modules', type=str, nargs='+',default=["q_proj", "v_proj","k_proj", "o_proj","up_proj", "down_proj","gate_proj"])
    parser.add_argument('--model_name', type=str, default='chatdoctor',) # meditron
    parser.add_argument('--layers', type=int, nargs='+', default=None,)


    args = parser.parse_args()
    print(args)
    return args

def write_result(args, metrics):
    metrics_filename = f'{args.dataset}-{args.method}'
    if args.lora_ver:
        metrics_filename += f'-{args.lora_ver}'
    if args.lr:
        metrics_filename += f'-{args.lr}'
    if args.num_steps:
        metrics_filename += f'-{args.num_steps}'
    if args.method == 'MedLaSA':
        if args.alpha_dynamic: metrics_filename += f'-alpha_dynamic'
        if args.rank_dynamic: metrics_filename += f'-rank_dynamic'
        metrics_filename += f'-{args.alpha0}'
        metrics_filename += f'-{args.rank0}'
        metrics_filename += f'-{args.norm}'
        tm_str = '-'.join(args.target_modules)
        metrics_filename += f'-{tm_str}'
    metrics_filename += '.json'
    print(f'--------------dump {metrics_filename}----------------')
    with open(f'metrics_results/{metrics_filename}', 'w',encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False) 

class read_casual_tracing():
    def __init__(self,args,) -> None:
        self.args = args

    def normalization(self, data):
        if self.args.norm == 'meanstd':
            return (data - np.mean(data)) / np.std(data)  + 1
        else:
            return (data - np.min(data)) / (np.max(data) - np.min(data))

    def compute_alpha_score(self, path):
        impact = np.load(path)
        subject_range = impact['subject_range']
        score = impact['scores'][:-1,:]
        score = np.mean(score[subject_range[0]:subject_range[1],:],axis=0,keepdims=False)
        score = self.normalization(score)
        return score

    def compute_prompt2alpha(self,):
        filepath = f'../casual_tracing_data/{self.args.dataset}_causal_tracing/cases'
        samples = json.load(open(f'../casual_tracing_data/{self.args.dataset}_casual_tracing.json', 'r', encoding='utf-8'))
        prompt2alpha = {}
        for sample in samples:
            known_id = sample['known_id']
            attn_score = self.compute_alpha_score(f'{filepath}/knowledge_{known_id}_attn.npz')
            mlp_score = self.compute_alpha_score(f'{filepath}/knowledge_{known_id}_mlp.npz')
            
            alpha_pattern = {}
            for i in range(32):
                for target in self.args.target_modules:
                    if target in ["q_proj", "v_proj","k_proj", "o_proj"]:
                        alpha_pattern[f'model.layers.{i}.self_attn.{target}'] = min(max(self.args.alpha0 //2, attn_score[i] * self.args.alpha0), self.args.alpha0 *2)
                    elif target in ["up_proj", "down_proj","gate_proj"]:
                        alpha_pattern[f'model.layers.{i}.mlp.{target}'] = min(max(self.args.alpha0 //2, mlp_score[i] * self.args.alpha0), self.args.alpha0 *2)
                    else: 
                        raise
            if not self.args.alpha_dynamic:
                alpha_pattern = {}
            if args.dataset=='MedFE':
                prompt2alpha["Please provide an explanation for the following fact: \n "+sample['prompt']] = alpha_pattern
            elif args.dataset=='MedCF':
                prompt2alpha[sample['prompt']] = alpha_pattern
        return prompt2alpha

    def compute_global_score(self, samples, filepath, attn_or_mlp):
        scores = []
        for sample in samples:
            known_id = sample['known_id']
            impact = np.load(f'{filepath}/knowledge_{known_id}_{attn_or_mlp}.npz')
            score = impact['scores'][:-1,:]
            score = np.mean(score,axis=0,keepdims=False)
            scores.append(score)
        scores = np.stack(scores,axis=0)
        scores = np.mean(scores,axis=0,keepdims=False)
        scores = self.normalization(scores)
        return scores

    def compute_rank_pattern(self,):
        filepath = f'../casual_tracing_data/{self.args.dataset}_causal_tracing/cases'
        samples = json.load(open(f'../casual_tracing_data/{self.args.dataset}_casual_tracing.json', 'r', encoding='utf-8'))        
        attn_scores = self.compute_global_score( samples, filepath, 'attn')
        mlp_scores = self.compute_global_score( samples, filepath, 'mlp')
        rank_pattern = {}
        for i in range(32):
            for target in self.args.target_modules:
                if target in ["q_proj", "v_proj","k_proj", "o_proj"]:
                    rank = math.ceil(attn_scores[i] * self.args.rank0)
                    rank_pattern[f'model.layers.{i}.self_attn.{target}'] = min(max(self.args.rank0 //2, rank), self.args.rank0 *2 )
                elif target in ["up_proj", "down_proj","gate_proj"]:
                    rank = math.ceil(mlp_scores[i] * self.args.rank0)
                    rank_pattern[f'model.layers.{i}.mlp.{target}'] = min(max(self.args.rank0 //2, rank), self.args.rank0 *2 )
                else: 
                    raise
        return rank_pattern


def edit_med_llm(args):
    if args.dataset == 'MedCF': dataset = MedCF_load(args)
    elif args.dataset == 'MedFE': dataset = MedFE_load(args)
    model = globals()[f'{args.method}HyperParams']

    hparams = model.from_hparams(f'./hparams/{args.method}-llama.yaml') 
    hparams.model_name = f"./../../LLM_checkpoint/{args.model_name}-llama"
    if hparams.model_parallel:
        hparams.device = 'cuda'
    if args.method=='MEND':
        hparams.archive = f'./results/{args.dataset}/models/MEND/{args.model_name}-llama'
        hparams.results_dir = f'./results/{args.dataset}'
        hparams.tokenizer_name = f'./../../LLM_checkpoint/{args.model_name}-llama'
    

    if args.lora_ver:
        hparams.lora_ver = args.lora_ver
    if args.lr:
        hparams.lr = args.lr

    if args.layers:
        hparams.layers = args.layers

    if args.method == 'MedLaSA':
        if args.lora_ver=='AdaLoRA':
            args.rank_dynamic == False
        ct = read_casual_tracing(args)

        hparams.target_modules = args.target_modules
        hparams.num_steps = args.num_steps
        if args.rank_dynamic:
            hparams.rank_pattern = ct.compute_rank_pattern()
        else:
            hparams.rank_pattern = {}
            hparams.rank = args.rank0

        prompt2alpha = ct.compute_prompt2alpha()
        if not args.alpha_dynamic:
            hparams.lora_alpha = args.alpha0

        editor = BaseEditor.from_hparams(hparams)
        metrics, edited_model, _ = editor.mededit(
            prompts=dataset['prompts'],
            rephrase_prompts=dataset['rephrase_prompts'],
            target_new=dataset['target_new'],
            ground_truth=dataset['ground_truth'],
            subject=dataset['subject'],
            locality_inputs=dataset['locality_inputs'],
            portability_inputs=dataset['portability_inputs'],
            prompt2alpha=prompt2alpha,
            args=args,
            train_ds=None,
            keep_original_weight=True,
            test_generation=True,
        )
    else :
        editor = BaseEditor.from_hparams(hparams)
        metrics, edited_model, _ = editor.edit(
            prompts=dataset['prompts'],
            rephrase_prompts=dataset['rephrase_prompts'],
            target_new=dataset['target_new'],
            ground_truth=dataset['ground_truth'],
            subject=dataset['subject'],
            locality_inputs=dataset['locality_inputs'],
            portability_inputs=dataset['portability_inputs'],
            train_ds=None,
            keep_original_weight=True,
            test_generation=True,
        )
    metrics_compute(metrics)
    write_result(args, metrics)
    
    return metrics, edited_model


def MEND_Train_Llama(args):
    model = globals()[f'{args.method}TrainingHparams']
    training_hparams = model.from_hparams(f'./hparams/{args.method}-train-llama.yaml')
    training_hparams.results_dir = f'./results/{args.dataset}'
    train_ds = CounterFactDataset(f'./../MEND_training/{args.dataset}/train.json', config=training_hparams)
    eval_ds = CounterFactDataset(f'./../MEND_training/{args.dataset}/valid.json', config=training_hparams)
    trainer = EditTrainer(
        config=training_hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    trainer.run()


if __name__=='__main__':
    args = Args()    
    if args.trainedit=='train':
        MEND_Train_Llama(args)
    else:
        edit_med_llm(args)
    # metrics_compute('./metrics_results/MedCF-FT.json')