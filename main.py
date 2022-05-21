from unicodedata import mirrored
import pandas as pd
import torch

traindata = pd.read_json('./CHIP-CTC/CHIP-CTC_train.json')
valdata = pd.read_json('./CHIP-CTC/CHIP-CTC_dev.json')
testdata = pd.read_json('./CHIP-CTC/CHIP-CTC_test.json')

examplepreddata = pd.read_excel('./CHIP-CTC/category.xlsx')

examplepreddata['label2idx'] = range(examplepreddata.shape[0])

label2idx = dict(
    zip(examplepreddata['Label Name'], examplepreddata['label2idx']))

traindata['labels'] = [label2idx[item] for item in traindata['label']]
valdata['labels'] = [label2idx[item] for item in valdata['label']]

print(len(traindata))
print(len(valdata))
print(len(testdata))

from datasets import Dataset
import datasets

traindataset = Dataset.from_pandas(traindata)
valdataset = Dataset.from_pandas(valdata)
testdataset = Dataset.from_pandas(testdata)

dataset = datasets.DatasetDict({
    'train': traindataset,
    'validation': valdataset,
    'test': testdataset
})

print(dataset)

train_dataset = dataset['train']
print(train_dataset.features)

print(train_dataset[0])

from transformers import BertTokenizer, BertModel, AutoModelForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('uer/chinese_roberta_L-8_H-512',
                                          mirror='tuna')
model = AutoModelForSequenceClassification.from_pretrained(
    "uer/chinese_roberta_L-8_H-512", num_labels=examplepreddata.shape[0])


def tokenize_function(sample):
    return tokenizer(sample['text'], truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

tokenized_datasets['train'] = tokenized_datasets['train'].remove_columns(
    ['id', 'text', 'label'])
tokenized_datasets['validation'] = tokenized_datasets[
    'validation'].remove_columns(['id', 'text', 'label'])
tokenized_datasets['test'] = tokenized_datasets['test'].remove_columns(
    ['id', 'text'])

from transformers import DataCollatorWithPadding  #实现按batch自动padding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

print(tokenized_datasets)
print(tokenized_datasets['train'][0])

smaples = data_collator(tokenized_datasets['train'][:5])

outputs = model(**smaples)

print(vars(outputs).keys())
print(outputs.loss)
print(outputs.logits)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=8,
    weight_decay=0.01,
)

trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=tokenized_datasets["train"],
                  eval_dataset=tokenized_datasets["validation"],
                  tokenizer=tokenizer,
                  data_collator=data_collator,
                  resume_from_checkpoint='./results/checkpoint-7000')

predictions = trainer.predict(tokenized_datasets['validation'])

print(predictions.predictions.shape)
