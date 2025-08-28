from dotenv import load_dotenv
import os
load_dotenv()
token = os.getenv("HUGGINGFACE_TOKEN")

import huggingface_hub
huggingface_hub.login(token)

from transformers import AutoModelForSequenceClassification
pretrained_model_name = "distilbert-base-uncased"
pretrained_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name)

from datasets import load_dataset
training_data = load_dataset("imdb",split="train")
test_data = load_dataset("imdb",split="test")
print(training_data)
print(test_data)

# Get the first row in the training data
training_text=training_data['text'][0]

# Get the first label in the training data
training_label=training_data['label'][0]

# Get the first row in the test data
test_text=test_data['text'][0]

# Get the first label in the test data
test_label=test_data['label'][0]

print("The first review in the training data:", training_text)
print("The sentiment label for the first review in the training data:", training_label)
print("The first review in the text data:", training_text)
print("The sentiment label for the first review in the test data:", test_label)

from transformers import AutoTokenizer

pretrained_model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

def tokenize(record):
    outputs = tokenizer(record['text'], truncation=True, padding="max_length", max_length=512)
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer)

tokenized_training_data = training_data.map(tokenize, batched=True)
tokenized_test_data = test_data.map(tokenize, batched=True)

print("Tokenized training data:",tokenized_training_data)
print("Tokenized test data:",tokenized_test_data)


# Get the tokenized_version of the first row in the training data
training_input_id=tokenized_training_data['input_ids'][0]

# Get the attention mask of the first row in the training data
training_attention_mask=tokenized_training_data['attention_mask'][0]

# Get the tokenized_version of the first row in the text data
test_input_id=tokenized_test_data['input_ids'][0]

# Get the attention mask of the first row in the text data
test_attention_mask=tokenized_test_data['attention_mask'][0]

print("The tokenized version of the first review in the training data:", training_input_id)
print("The attention mask for the first review in the training data:", training_attention_mask)
print("The tokenized version of the first review in the test data:", test_input_id)
print("The attention mask for the first review in the test data:", test_attention_mask)

import numpy as np
import evaluate

def compute_metrics(predictions_and_labels):
    metric = evaluate.load("accuracy")
    logits, labels = predictions_and_labels
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

from transformers import TrainingArguments  
from trl import SFTTrainer  
  
# Create training arguments  
training_args = TrainingArguments(num_train_epochs=10,  
                                  output_dir="text-classifier-supervised-codecademy",  
                                  per_device_train_batch_size=16,  
                                  per_device_eval_batch_size=16,  
                                  eval_strategy="epoch")  
  
# Create a supervised model trainer for fine-tuning the LLM model  
trainer = SFTTrainer(model=pretrained_model,   
                     processing_class=tokenizer,  
                     data_collator=data_collator,  
                     args=training_args,  
                     train_dataset=tokenized_training_data,  
                     eval_dataset=tokenized_test_data,   
                     compute_metrics=compute_metrics)  

trainer.train()

file_path_to_save_the_model =  '/home/a/andrewsq/codes/text-classifier-supervised-codecademy'
trainer.save_model(file_path_to_save_the_model)