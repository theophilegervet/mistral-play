import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


if __name__ == "__main__":
    # model_id = "mistralai/Mistral-7B-Instruct-v0.1"
    model_id = "Open-Orca/Mistral-7B-OpenOrca"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(model)

    device = "cuda:0"
    messages = [
        {"role": "user", "content": "Can you summarize the state of ML in 2023?"}
    ]
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to(device)
    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    print(decoded[0])
