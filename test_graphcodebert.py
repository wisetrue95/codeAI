import os
import wandb
wandb.init(project="Code", id='graphbert_left_sst2')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"

from transformers import AutoTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, \
    EarlyStoppingCallback, DataCollatorWithPadding
import pandas as pd
import numpy as np
from datasets import load_dataset, load_metric


MODEL = "microsoft/graphcodebert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.truncation_side = 'left'
model = RobertaForSequenceClassification.from_pretrained(MODEL)


TRAIN_INPUT = "data/Code/graphbert_left/train_graphcodebert_left.csv"
VAL_INPUT = "data/Code/graphbert_left/val_graphcodebert_left.csv"
MAX_LEN = 512
train_dataset = load_dataset("csv", data_files=TRAIN_INPUT)['train']
val_dataset = load_dataset("csv", data_files=VAL_INPUT)['train']


def example_fn(examples):
    outputs = tokenizer(examples['code1'], examples['code2'], padding=True, max_length=MAX_LEN, truncation=True) #MAX_LEN
    if 'similar' in examples:
        outputs["labels"] = examples["similar"]
    return outputs

train_dataset = train_dataset.map(example_fn, remove_columns=['code1', 'code2', 'similar'])
# train_dataset = train_dataset.train_test_split(0.1)

val_dataset = val_dataset.map(example_fn, remove_columns=['code1', 'code2', 'similar'])

_collator = DataCollatorWithPadding(tokenizer=tokenizer)
_metric = load_metric("glue", "sst2")


def metric_fn(p):
    preds, labels = p
    output = _metric.compute(references=labels, predictions=np.argmax(preds, axis=-1))
    return output


args = TrainingArguments(
    'runs_graphbert/',
    per_device_train_batch_size=36,
    num_train_epochs=10,
    do_train=True,
    do_eval=True,
    save_strategy="epoch",
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    data_seed=42,
    load_best_model_at_end=True,
    per_device_eval_batch_size=36,
    report_to="wandb")

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


# test
TEST = "data/Code/test.csv"
SUB = "data/Code/sample_submission.csv"

def preprocess_script(script):
    lines = script.split('\n')
    preproc_lines = []
    for line in lines:
        if line.lstrip().startswith('#'):
            continue
        line = line.rstrip()
        if '#' in line:
            line = line[:line.index('#')]
        line = line.replace('\n', '')
        line = line.replace('    ', '\t')
        if line == '':
            continue
        preproc_lines.append(line)
    preprocessed_script = '\n'.join(preproc_lines)
    return preprocessed_script

def example_test_fn(examples):
    examples['code1'] = preprocess_script(examples['code1'])
    examples['code2'] = preprocess_script(examples['code2'])
    outputs = tokenizer(examples['code1'], examples['code2'], padding=True, max_length=MAX_LEN, truncation=True)
    if 'similar' in examples:
        outputs["labels"] = examples["similar"]
    return outputs


test_dataset = load_dataset("csv", data_files=TEST)['train']
test_dataset = test_dataset.map(example_test_fn, remove_columns=['code1', 'code2'])

predictions = trainer.predict(test_dataset)

df = pd.read_csv(SUB)
df['similar'] = np.argmax(predictions.predictions, axis=-1)
df.to_csv('./graphcodebert.csv', index=False)
