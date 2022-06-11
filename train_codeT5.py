import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
#os.environ["WANDB_DISABLED"] = "true"

from transformers import Trainer, TrainingArguments, T5Config, DataCollatorWithPadding, T5ForConditionalGeneration, RobertaTokenizer
import numpy as np
from datasets import load_dataset, load_metric
import torch
from transformers import EarlyStoppingCallback

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'learning to {device}')

MODEL = 'Salesforce/codet5-base'
config = T5Config.from_pretrained(MODEL)
tokenizer = RobertaTokenizer.from_pretrained(MODEL)
model = T5ForConditionalGeneration.from_pretrained(MODEL)


TRAIN_INPUT = "data/Code/ct5/train_ct5.csv"
#VAL_INPUT = "./data/Code/sample_train.csv"
VAL_INPUT = "data/Code/ct5/val_ct5.csv"
MAX_LEN = 512
train_dataset = load_dataset("csv", data_files=TRAIN_INPUT)['train']
val_dataset = load_dataset("csv", data_files=VAL_INPUT)['train']


def example_fn(examples):
    code1 = "sst2 sentence: "+examples['code1']
    code2 = "sentence: " + examples['code2']
    outputs = tokenizer(code1, code2, return_tensors='pt', padding=True, max_length=MAX_LEN, truncation=True)
    outputs["input_ids"] = outputs.input_ids.squeeze()
    outputs["attention_mask"] = outputs.attention_mask.squeeze()
    if 'similar' in examples:
        labels = tokenizer(str(examples["similar"]), return_tensors='pt')
        outputs["labels"] = labels.input_ids.squeeze()

    return outputs

train_dataset = train_dataset.map(example_fn, remove_columns=['code1', 'code2', 'similar'])
val_dataset = val_dataset.map(example_fn, remove_columns=['code1', 'code2', 'similar'])
#val_dataset = val_dataset.train_test_split(0.1)

_collator = DataCollatorWithPadding(tokenizer=tokenizer)
_metric = load_metric("glue", "sst2")


def metric_fn(p):
    preds, labels = p
    pp = np.argmax(preds[0], axis=-1)
    new_labels = labels[:, 1].copy()
    new_preds = pp[:, 1].copy()

    output = _metric.compute(references=new_labels, predictions=new_preds)
    return output



args = TrainingArguments(
    './runs_t5/',  # output directory
    per_device_train_batch_size=27,  # 24,
    num_train_epochs=20,
    do_train=True,
    do_eval=True,
    save_strategy="epoch",
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    data_seed=42,
    eval_accumulation_steps=100,
    load_best_model_at_end=True,
    per_device_eval_batch_size=16,
    # metric_for_best_model=True
)

trainer = Trainer(
    model=model,
    args=args,
    data_collator=_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=metric_fn,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
)

trainer.train()

