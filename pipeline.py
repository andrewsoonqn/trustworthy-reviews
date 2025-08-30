#!/usr/bin/env python
# coding: utf-8

# # Fine-tuning Sandbox
#
# Code authored by: Shawhin Talebi <br>
# Blog link: https://medium.com/towards-data-science/fine-tuning-large-language-models-llms-23473d763b91

# In[1]:

import pandas as pd

# ignore this once u get a json, replace with
# df = pd.read_json("../preprocessing/data/llmOutput/llmevaluated_reviews_Kaggle_400.json")
df = pd.read_csv("preprocessing/data/llm_output/llmevaluated_reviews_Kaggle_500.csv")
print("Loaded dataset: ")
print(df.head())

from datasets import load_dataset, DatasetDict, Dataset

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer)

from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
import evaluate
import torch
import numpy as np

torch.device("cuda")

# ### dataset

# In[2]:


# # how dataset was generated

# # load imdb data
# imdb_dataset = load_dataset("imdb")

# # define subsample size
# N = 1000
# # generate indexes for random subsample
# rand_idx = np.random.randint(24999, size=N)

# # extract train and test data
# x_train = imdb_dataset['train'][rand_idx]['text']
# y_train = imdb_dataset['train'][rand_idx]['label']

# x_test = imdb_dataset['test'][rand_idx]['text']
# y_test = imdb_dataset['test'][rand_idx]['label']

# # create new dataset
# dataset = DatasetDict({'train':Dataset.from_dict({'label':y_train,'text':x_train}),
#                              'validation':Dataset.from_dict({'label':y_test,'text':x_test})})


# In[3]:


label_cols = ["inauthentic","irrelevant", "advertisement", "rant","policy_violations"]
pos_counts = df[label_cols].sum().to_numpy()
neg_counts = len(df) - pos_counts
pos_weight = torch.tensor(neg_counts / np.maximum(pos_counts, 1), dtype=torch.float)
ds_full = Dataset.from_pandas(df)
split = ds_full.train_test_split(test_size=0.2, seed=42)
dataset = DatasetDict({"train": split["train"], "validation": split["test"]})

print("Created train and test/eval splits")


# In[4]:


# display % of training data with label=1
# np.array(dataset['train']['label']).sum()/len(dataset['train']['label'])


# ### model

# In[ ]:


model_checkpoint = 'distilbert-base-uncased'
# model_checkpoint = 'roberta-base' # you can alternatively use roberta-base but this model is bigger thus training will take longer

# define label maps
id2label = {
    0: "inauthentic",
    1: "irrelevant",
    2: "advertisement",
    3: "rant",
    4: "policy_violations"
} # for the model to predict

label2id = {v: k for k, v in id2label.items()} # to understand training data

model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=5, id2label=id2label, label2id=label2id, problem_type="multi_label_classification")


print("Created model to be trained")
# In[ ]:


# display architecture


# ### preprocess data

# In[ ]:


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))


# In[ ]:


def build_labels(examples):
    labels = []
    n = len(examples[label_cols[0]])
    for i in range(n):
        labels.append([int(examples[c][i]) for c in label_cols])
    return {"labels": labels}

def build_text(examples):
    texts = []
    n = len(examples["review_text"])
    for i in range(n):
        parts = []
        parts.append(str(examples["business_name"][i]))
        parts.append(str(examples["author_name"][i]))
        parts.append(str(examples["review_text"][i]))
        parts.append(f"rating:{examples['rating'][i]}")
        parts.append(f"sentimentNum:{examples['sentimentNum'][i]}")
        parts.append(f"subjectivity:{examples['subjectivity'][i]}")
        parts.append(f"lang:{examples['lang'][i]}")
        parts.append(f"review_length:{examples['review_length'][i]}")
        parts.append(f"exclaim_count:{examples['exclaim_count'][i]}")
        parts.append(f"caps_count:{examples['caps_count'][i]}")
        parts.append(f"contains_url:{examples['contains_url'][i]}")
        parts.append(f"reason:{examples['reason'][i]}")
        texts.append(" ".join(parts))
    return {"text": texts}

def tokenize_function(examples):
    tokenizer.truncation_side = "left"
    return tokenizer(
        examples["text"],
        return_tensors="np",
        truncation=True,
        max_length=512
    )


# In[ ]:


dataset = dataset.map(build_labels, batched=True)
dataset = dataset.map(build_text, batched=True)
tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset

print("Tokenized, built labels and text")

# In[ ]:


# create data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# ### evaluation

# In[ ]:


metric_f1 = evaluate.load("f1")


# In[ ]:


def compute_metrics(p):
    logits = p.predictions
    labels = p.label_ids.astype(int)
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    f1_micro = metric_f1.compute(predictions=preds.flatten(), references=labels.flatten(), average="micro")["f1"]
    f1_macro = metric_f1.compute(predictions=preds.flatten(), references=labels.flatten(), average="macro")["f1"]
    subset_acc = float((preds == labels).all(axis=1).mean())
    return {"f1_micro": f1_micro, "f1_macro": f1_macro, "subset_accuracy": subset_acc}


# ### Apply untrained model to text

# In[ ]:


# define list of examples - first 5 rows from CSV
text_list = df["review_text"].head(5).tolist()

print("Untrained model predictions:")
print("----------------------------")
for text in text_list:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    logits = model(**inputs).logits
    probs = torch.sigmoid(logits).detach().cpu().numpy()[0]
    idx = np.where(probs >= 0.5)[0]
    labels_pred = [id2label[i] for i in idx]
    print(text + " - " + ", ".join(labels_pred))


# ### Train model

# In[ ]:


peft_config = LoraConfig(task_type="SEQ_CLS",
                        r=4,
                        lora_alpha=32,
                        lora_dropout=0.01,
                        target_modules = ['q_lin'])


# In[ ]:


peft_config


# In[ ]:


model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


# In[ ]:


# hyperparameters
lr = 1e-3
batch_size = 4
num_epochs = 10


# In[ ]:


# define training arguments
training_args = TrainingArguments(
    output_dir= model_checkpoint + "-lora-text-classification",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)


# In[ ]:


class WeightedTrainer(Trainer):
    def __init__(self, pos_weight=None, **kwargs):
        super().__init__(**kwargs)
        self.pos_weight = pos_weight
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.pos_weight is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
        else:
            loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels.float())
        return (loss, outputs) if return_outputs else loss

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator, # this will dynamically pad examples in each batch to be equal length
    compute_metrics=compute_metrics,
    pos_weight=pos_weight,
)

# train model
trainer.train()


# ### Generate prediction

# In[ ]:


model.to('cpu')

print("Trained model predictions:")
print("--------------------------")
for text in text_list:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to("cpu")
    logits = model(**inputs).logits
    probs = torch.sigmoid(logits).detach().cpu().numpy()[0]
    idx = np.where(probs >= 0.5)[0]
    labels_pred = [id2label[i] for i in idx]
    print(text + " - " + ", ".join(labels_pred))


