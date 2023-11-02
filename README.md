# Play with Mistral 7B

Set up the environment:
```
conda create -n mistral python=3.10;
conda activate mistral;
pip install bitsandbytes;
pip install git+https://github.com/huggingface/transformers.git;
pip install git+https://github.com/huggingface/peft.git;
pip install git+https://github.com/huggingface/accelerate.git;
pip install datasets;
pip install trl;
pip install scipy;
pip install wandb;
```

Fine-tune base model on NER with QLoRA on a single 12GB GPU:
```
python fine_tune.py
```

Evaluate fine-tuned base model against few-shot prompted instruct model:
```
python evaluate.py
```
