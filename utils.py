import json
import tqdm
import numpy as np
import pandas as pd
from transformers.integrations import WandbCallback


tags = {
    "B-CARDINAL": 1,
    "B-DATE": 2,
    "I-DATE": 3,
    "B-PERSON": 4,
    "I-PERSON": 5,
    "B-NORP": 6,
    "B-GPE": 7,
    "I-GPE": 8,
    "B-LAW": 9,
    "I-LAW": 10,
    "B-ORG": 11,
    "I-ORG": 12, 
    "B-PERCENT": 13,
    "I-PERCENT": 14, 
    "B-ORDINAL": 15, 
    "B-MONEY": 16, 
    "I-MONEY": 17, 
    "B-WORK_OF_ART": 18, 
    "I-WORK_OF_ART": 19, 
    "B-FAC": 20, 
    "B-TIME": 21, 
    "I-CARDINAL": 22, 
    "B-LOC": 23, 
    "B-QUANTITY": 24, 
    "I-QUANTITY": 25, 
    "I-NORP": 26, 
    "I-LOC": 27, 
    "B-PRODUCT": 28, 
    "I-TIME": 29, 
    "B-EVENT": 30,
    "I-EVENT": 31,
    "I-FAC": 32,
    "B-LANGUAGE": 33,
    "I-PRODUCT": 34,
    "I-ORDINAL": 35,
    "I-LANGUAGE": 36
}


tag_to_string = {string: tag for tag, string in tags.items()}


def format_training_example(example):
    input_string = ' '.join(example['tokens'])
    output_string = json.dumps({
        token: tag_to_string[tag]
        for (token, tag) in zip(example['tokens'], example['tags'])
        if tag in tag_to_string
    })
    return f"### Input: {input_string}\n### Output: {output_string}"


def format_few_shot_prompting_example(example):
    input_string = ' '.join(example['tokens'])
    output_string = json.dumps({
        token: tag_to_string[tag]
        for (token, tag) in zip(example['tokens'], example['tags'])
        if tag in tag_to_string
    })
    prompt = "Your goal is to identify entities in the input text."
    prompt += f"\n### Allowed entities: {list(tags.keys())}"
    prompt += '\n\n### Input: American and Japanese officials offered several theories for the difference in approach betwen Mr. Mosbacher and Mrs. Hills .\n### Output: {"American": "B-NORP", "Japanese": "B-NORP", "Mosbacher": "B-PERSON", "Hills": "B-PERSON"}'
    prompt += '\n\n### Input: In the early 1980s , P&G tried to launch here a concentrated detergent under the Ariel brand name that it markets in Europe .\n### Output: {"the": "B-DATE", "early": "I-DATE", "1980s": "I-DATE", "P&G": "B-ORG", "Ariel": "B-PRODUCT", "Europe": "B-LOC"}'
    prompt += '\n\n### Input: $ 150 million of 9 % debentures due Oct. 15 , 2019 , priced at 99.943 to yield 9.008 % .\n### Output: {"$": "B-MONEY", "150": "I-MONEY", "million": "I-MONEY", "9": "B-PERCENT", "%": "I-PERCENT", "Oct.": "B-DATE", "15": "I-DATE", ",": "I-DATE", "2019": "I-DATE", "99.943": "B-CARDINAL", "9.008": "B-PERCENT"}'
    prompt += '\n\n### Input: New York , light rain , 1 to 3 degrees .\n### Output: {"New": "B-GPE", "York": "I-GPE", "1": "B-CARDINAL", "3": "B-QUANTITY", "degrees": "I-QUANTITY"}'
    prompt += f"\n\n### Input: {input_string}\n### Output: {output_string}"
    return prompt


def preprocess_dataset(dataset, tokenizer, validation_size=None, test_size=None):
    for split in dataset.keys():
        dataset[split] = dataset[split].filter(lambda example: len(example['tokens']) < 50)
        training_example_column = [format_training_example(example) for example in dataset[split]]
        dataset[split] = dataset[split].add_column("example", training_example_column)
        dataset[split] = dataset[split].map(lambda samples: tokenizer(samples["example"]), batched=True)

        if split == "test":
            few_shot_prompting_column = [format_few_shot_prompting_example(example) for example in dataset[split]]
            dataset[split] = dataset[split].add_column("few_shot_prompting", few_shot_prompting_column)

    if validation_size is not None:
        dataset["validation"] = dataset["validation"].select(range(validation_size))
    if test_size is not None:
        dataset["test"] = dataset["test"].select(range(test_size))

    return dataset


def output_string_to_dict(input_string):
    try:
        return json.loads(input_string.split("Output: ")[-1].split("}")[0] + "}")
    except:
        return {}
    

def run_inference(model, tokenizer, dataset_split, batch_size):
    all_predicted_entities = []
    all_label_entities = []

    for idx in tqdm.tqdm(range(len(dataset_split) // batch_size)):
        subset = dataset_split[idx * batch_size: (idx + 1) * batch_size]

        input_text = ["{".join(s.split("{")[:-1]) for s in subset]
        model_inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(model.device)
        predicted_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True)
        predicted_text = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)

        all_predicted_entities.extend([output_string_to_dict(pred) for pred in predicted_text])
        all_label_entities.extend([output_string_to_dict(label) for label in subset])
    
    return all_predicted_entities, all_label_entities


def decode_predictions(tokenizer, predictions):
    label_ids = np.where(predictions.label_ids != -100, predictions.label_ids, tokenizer.pad_token_id)
    label_text = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    predicted_text = tokenizer.batch_decode(predictions.predictions.argmax(axis=-1))

    label_entities = [output_string_to_dict(label) for label in label_text]
    predicted_entities = [output_string_to_dict(pred) for pred in predicted_text]

    return {
        "label_entities": label_entities, 
        "predicted_text": predicted_text,
        "predicted_entities": predicted_entities,
    }


def compute_metrics(predicted_entities, label_entities):
    metrics = {
        tag: {
            "true_positive": 0,
            "false_negative": 0,
            "false_positive": 0
        }
        for tag in tags
    }
    for predictions, labels in zip(predicted_entities, label_entities):
        for text, tag in labels.items():
            if predictions.get(text) == tag:
                metrics[tag]["true_positive"] += 1
            else:
                metrics[tag]["false_negative"] += 1
        for text, tag in predictions.items():
            if tag in metrics and labels.get(text) != tag:
                metrics[tag]["false_positive"] += 1
    metrics["TOTAL"] = {
        key: sum([values[key] for values in metrics.values()])
        for key in ["true_positive", "false_negative", "false_positive"]
    }
    metrics["TOTAL"]["precision"] =  metrics["TOTAL"]["true_positive"] / max(metrics["TOTAL"]["true_positive"] + metrics["TOTAL"]["false_positive"], 1)
    metrics["TOTAL"]["recall"] = metrics["TOTAL"]["true_positive"] / max(metrics["TOTAL"]["true_positive"] + metrics["TOTAL"]["false_negative"], 1)
    return metrics


def find_lora_module_names(model, cls):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return list(lora_module_names)


class WandbPredictionProgressCallback(WandbCallback):
    def __init__(self, trainer, tokenizer, val_dataset, num_samples):
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset.select(range(num_samples))

    def on_evaluate(self, args, state, control,  **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        predictions = self.trainer.predict(self.sample_dataset)
        predictions = decode_predictions(self.tokenizer, predictions)
        metrics = compute_metrics(predictions["predicted_entities"], predictions["label_entities"])
        predictions["label_entities"] = [str(e) for e in predictions["label_entities"]]
        predictions["predicted_entities"] = [str(e) for e in predictions["predicted_entities"]]
        predictions_df = pd.DataFrame(predictions)
        predictions_df["global_step"] = state.global_step
        records_table = self._wandb.Table(dataframe=predictions_df)
        self._wandb.log({"sample_predictions": records_table})
        self._wandb.log({"eval/precision": metrics["TOTAL"]["precision"]})
        self._wandb.log({"eval/recall": metrics["TOTAL"]["recall"]})
