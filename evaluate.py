import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from utils import preprocess_dataset, compute_metrics, run_inference


def eval_few_shot_prompting():
    model_id = "mistralai/Mistral-7B-Instruct-v0.1"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                    
        bnb_4bit_use_double_quant=True,      
        bnb_4bit_quant_type="nf4",         
        bnb_4bit_compute_dtype=torch.bfloat16 
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = '[PAD]'
    
    dataset = load_dataset("tner/ontonotes5")
    dataset = preprocess_dataset(dataset, tokenizer, test_size=16)

    predicted_entities, label_entities = run_inference(model, tokenizer, dataset["test"]["few_shot_prompting"], batch_size=4)
    metrics = compute_metrics(predicted_entities, label_entities)
    print()
    print("Few shot prompting metrics:")
    print(f"Precision: {metrics['TOTAL']['precision']} | Recall: {metrics['TOTAL']['recall']}")


def eval_supervised_fine_tuning():
    model_id = "mistralai/Mistral-7B-v0.1"
    checkpoint = "outputs/checkpoint-1819"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                    
        bnb_4bit_use_double_quant=True,      
        bnb_4bit_quant_type="nf4",         
        bnb_4bit_compute_dtype=torch.bfloat16 
    )
    model = AutoModelForCausalLM.from_pretrained(checkpoint, quantization_config=bnb_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = '[PAD]'

    dataset = load_dataset("tner/ontonotes5")
    dataset = preprocess_dataset(dataset, tokenizer, test_size=16)

    predicted_entities, label_entities = run_inference(model, tokenizer, dataset["test"]["example"], batch_size=16)
    print()
    print("Supervised fine-tuning metrics:")
    metrics = compute_metrics(predicted_entities, label_entities)
    print(f"Precision: {metrics['TOTAL']['precision']} | Recall: {metrics['TOTAL']['recall']}")


if __name__ == "__main__": 
    eval_few_shot_prompting()
    eval_supervised_fine_tuning()
