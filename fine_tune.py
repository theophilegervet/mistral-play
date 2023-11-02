
import torch
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
import transformers
from trl import SFTTrainer
import bitsandbytes as bnb
from utils import WandbPredictionProgressCallback, preprocess_dataset, find_lora_module_names


if __name__ == "__main__":

    # 1. Load model

    model_id = "mistralai/Mistral-7B-v0.1"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                    
        bnb_4bit_use_double_quant=True,      
        bnb_4bit_quant_type="nf4",         
        bnb_4bit_compute_dtype=torch.bfloat16 
    )

    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = '[PAD]'
    print(model)

    # 2. Prepare dataset
    
    dataset = load_dataset("tner/ontonotes5")
    validation_size = 80
    dataset = preprocess_dataset(dataset, tokenizer, validation_size)
    print(dataset)

    # 3. Train

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    modules = find_lora_module_names(model, cls=bnb.nn.Linear4bit)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable: {trainable} | Total: {total} | Percentage: {trainable/total*100:.4f}%")

    wandb.init(entity="tgervet", project="mistral-fine-tune")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field="example",
        peft_config=lora_config,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=8,
            gradient_accumulation_steps=8,
            warmup_steps=0.03,
            max_steps=200000,
            learning_rate=1e-5,
            logging_steps=1,
            evaluation_strategy="steps",
            eval_steps=100,
            per_device_eval_batch_size=8,
            output_dir="outputs",
            optim="paged_adamw_8bit",
            save_strategy="steps",
            save_steps=500,
            report_to="wandb",
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    progress_callback = WandbPredictionProgressCallback(
        trainer=trainer, 
        tokenizer=tokenizer, 
        val_dataset=dataset["validation"], 
        num_samples=int(0.2 * validation_size), 
    )
    trainer.add_callback(progress_callback)

    trainer.train()
