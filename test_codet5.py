import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["WANDB_DISABLED"] = "true"

from transformers import Trainer, TrainingArguments, T5Config, DataCollatorWithPadding, T5ForConditionalGeneration, RobertaTokenizer
import numpy as np
from datasets import load_dataset, load_metric
import torch
import pandas as pd


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'learning to {device}')

MODEL = "runs_t5/checkpoint-95583/"
config = T5Config.from_pretrained(MODEL)
tokenizer = RobertaTokenizer.from_pretrained(MODEL)
model = T5ForConditionalGeneration.from_pretrained(MODEL)

TEST = "test_split/test1.csv"
SUB = "test_split/sub1.csv"
MAX_LEN = 512

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
    code1 = "sst2 sentence: "+preprocess_script(examples['code1'])
    code2 = "sentence: " + preprocess_script(examples['code2'])
    outputs = tokenizer(code1, code2, return_tensors='pt', padding=True, max_length=MAX_LEN, truncation=True)
    outputs["input_ids"] = outputs.input_ids.squeeze()
    outputs["attention_mask"] = outputs.attention_mask.squeeze()
    if 'similar' in examples:
        labels = tokenizer(str(examples["similar"]), return_tensors='pt')
        outputs["labels"] = labels.input_ids.squeeze()
    else:
        labels = tokenizer('none', return_tensors='pt')
        outputs["labels"] = labels.input_ids.squeeze()


    return outputs


test_dataset = load_dataset("csv", data_files=TEST)['train']
test_dataset = test_dataset.map(example_test_fn, remove_columns=['code1', 'code2'])


_collator = DataCollatorWithPadding(tokenizer=tokenizer)
_metric = load_metric("glue", "sst2")


def metric_fn(p):
    preds, labels = p
    pp = np.argmax(preds[0], axis=-1)
    new_preds = pp[:, 1].copy()
    new_labels = labels[:, 1].copy()
    output = _metric.compute(references=new_labels, predictions=new_preds)

    return output


args = TrainingArguments(
    './runs_t5/',  # output directory
    per_device_train_batch_size=8,  # 24,
    num_train_epochs=10,
    do_train=False,
    do_eval=True,
    save_strategy="epoch",
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    data_seed=42,
    eval_accumulation_steps=100,
    per_device_eval_batch_size=16,
)

trainer = Trainer(
    model=model,
    args=args,
    data_collator=_collator,
    train_dataset=test_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=metric_fn)


#trainer.train()
predictions = trainer.predict(test_dataset)
pp=np.argmax(predictions.predictions[0], axis=-1)
new_preds = pp[:, 1].copy()
new_preds=np.where(new_preds==21,1,new_preds)
new_preds=np.where(new_preds==20,0,new_preds)
print(new_preds)

df = pd.read_csv(SUB)
df['similar'] = new_preds
df.to_csv("test_split/result1.csv", index=False)

