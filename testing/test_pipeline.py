#!/usr/bin/env python
# coding: utf-8

# # Fine-tuning Sandbox
# 
# Code authored by: Shawhin Talebi <br>
# Blog link: https://medium.com/towards-data-science/fine-tuning-large-language-models-llms-23473d763b91

# In[1]: 


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


# load dataset
dataset_big = load_dataset('shawhin/imdb-truncated') #panda

# take only 100 samples from train/validation
dataset = DatasetDict({
    "train": dataset_big["train"].select(range(100)),
    "validation": dataset_big["validation"].select(range(100)),
})



# In[4]:


# display % of training data with label=1
np.array(dataset['train']['label']).sum()/len(dataset['train']['label'])


# ### model

# In[ ]:


model_checkpoint = 'distilbert-base-uncased'
# model_checkpoint = 'roberta-base' # you can alternatively use roberta-base but this model is bigger thus training will take longer

# define label maps
id2label = {0: "Spam", 1: "Positive"} # for the model to predict 
label2id = {"Negative":0, "Positive":1} # to understand training data

# generate classification model from model_checkpoint
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id, problem_type="multi_label_classification")


# In[ ]:


# display architecture
model


# ### preprocess data

# In[ ]:


# create tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

# add pad token if none exists
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))


# In[ ]:


# create tokenize function
def tokenize_function(examples):
    # extract text
    text = examples["text"]

    #tokenize and truncate text
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512
    )

    return tokenized_inputs


# In[ ]:


# tokenize training and validation datasets
tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset


# In[ ]:


# create data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# ### evaluation

# In[ ]:


# import accuracy evaluation metric
accuracy = evaluate.load("accuracy")


# In[ ]:


# define an evaluation function to pass into trainer later
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)

    return {"accuracy": accuracy.compute(predictions=predictions, references=labels)}


# ### Apply untrained model to text

# In[ ]:


# define list of examples
text_list = ["It was good.", "Not a fan, don't recommed.", "Better than the first one.", "This is not worth watching even once.", "This one is a pass."]

print("Untrained model predictions:")
print("----------------------------")
for text in text_list:
    # tokenize text
    inputs = tokenizer.encode(text, return_tensors="pt")
    # compute logits
    logits = model(inputs).logits
    # convert logits to label
    predictions = torch.argmax(logits)

    print(text + " - " + id2label[predictions.tolist()])


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


# creater trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator, # this will dynamically pad examples in each batch to be equal length
    compute_metrics=compute_metrics,
)

# train model
trainer.train()


# ### Generate prediction

# In[ ]:


model.to('cpu') # moving to mps for Mac (can alternatively do 'cpu')

print("Trained model predictions:")
print("--------------------------")
for text in text_list:
    inputs = tokenizer.encode(text, return_tensors="pt").to("cpu") # moving to mps for Mac (can alternatively do 'cpu')

    logits = model(inputs).logits
    predictions = torch.max(logits,1).indices

    print(text + " - " + id2label[predictions.tolist()[0]])


