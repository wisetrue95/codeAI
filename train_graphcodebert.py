import os
import wandb

from transformers import AutoTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, \
    EarlyStoppingCallback, DataCollatorWithPadding
import numpy as np
from datasets import load_dataset, load_metric

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

wandb.init(project="Code", id='codebert_val_baseline')

MODEL = "microsoft/graphcodebert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.truncation_side = 'left'
model = RobertaForSequenceClassification.from_pretrained(MODEL)

INPUT = "data/Code/graphbert/val_graphcodebert.csv"
MAX_LEN = 512
dataset = load_dataset("csv", data_files=INPUT)['train']

def example_fn(examples):
    outputs = tokenizer(examples['code1'], examples['code2'], padding=True, max_length=MAX_LEN, truncation=True)
    if 'similar' in examples:
        outputs["labels"] = examples["similar"]
    return outputs

dataset = dataset.train_test_split(0.1, seed=42)
train_dataset = dataset['train'].map(example_fn, remove_columns=['code1', 'code2', 'similar'])
val_dataset = dataset['test'].map(example_fn, remove_columns=['code1', 'code2', 'similar'])

_collator = DataCollatorWithPadding(tokenizer=tokenizer)
_metric = load_metric("glue", "sst2")


def metric_fn(p):
    preds, labels = p
    output = _metric.compute(references=labels, predictions=np.argmax(preds, axis=-1))
    return output


args = TrainingArguments(
    'runs_graphbert/',
    per_device_train_batch_size=16,
    num_train_epochs=10,
    do_train=True,
    do_eval=True,
    save_strategy="epoch",
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    data_seed=42,
    load_best_model_at_end=True,
    per_device_eval_batch_size=36,
    report_to="wandb"
)

trainer = Trainer(
    model=model,
    args=args,
    data_collator=_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=metric_fn,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)])

trainer.train()