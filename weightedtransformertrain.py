print("Starting Notebook")

try:
  import datasets, evaluate, transformers
except:
  !pip install transformers datasets evaluate transformers[torch] > /dev/null 2>$1
  import datasets, evaluate, transformers
  print("Successfully installed libraries")

import os
import matplotlib.pyplot as plt

#Data Distribution for Training
PATH = r"/content/Data"

import numpy as np

y_axis = np.array(y_axis)
weights = 1./y_axis

import datasets

#If training GAN
ds = datasets.load_dataset("imagefolder", data_dir=r"/mydata/Synthetic_25")
ds = ds["train"]
ds_train = ds["train"]

#If training Real Data
ds_real = datasets.load_dataset("imagefolder", data_dir=PATH)
ds_train = ds_real["train"].shuffle(seed=1)
ds_train = ds_real["train"].train_test_split(test_size=0.2)
ds_test = ds_real["test"]

checkpoint = "facebook/deit-base-distilled-patch16-224"

image_processor = transformers.AutoImageProcessor.from_pretrained(checkpoint, return_tensors="pt")

len(ds_train)

import torch
import torchvision.transforms as T
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms.autoaugment import AutoAugmentPolicy

train_tfs = Compose([
      T.ElasticTransform(),
      Resize(size=(224,224)),
      ToTensor(),
      Normalize(mean=[0.5],
                std=[0.5])
    ]
    )

test_tfs = Compose(
    [
      Resize(size=(224,224)),
      ToTensor(),
      Normalize(mean=[0.5],
                std=[0.5])
    ]
)

def train_transforms(example):
  example["pixel_values"] = [train_tfs(image.convert("RGB")) for image in example["image"]]
  del example["image"]
  return example

def test_transforms(example):
  example["pixel_values"] = [test_tfs(image.convert("RGB")) for image in example["image"]]
  del example["image"]
  return example

ds_train = ds_train.with_transform(train_transforms)
ds_test = ds_test.with_transform(test_transforms)

import evaluate
import numpy as np

f1 = evaluate.load("f1")
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": np.mean(predictions == labels)}

labels = ds_train.features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

data_collator = transformers.DefaultDataCollator(return_tensors="pt")

model = transformers.DeiTForImageClassification.from_pretrained(checkpoint,
                               num_labels=len(labels),
                               id2label=id2label,
                               label2id=label2id,
                               ignore_mismatched_sizes=True,
                               )

model.config = transformers.DeiTConfig(hidden_dropout_prob=0.1, attention_probs_dropout_prob = 0.1)

model = model.to("cuda")

def compute_class_weights(class_counts):
  updated_counts = []
  total = sum(y_axis)
  for count in class_counts:
    updated_counts.append(1-(count/total))

  return np.array(updated_counts)

class_weights = torch.tensor(compute_class_weights(y_axis),dtype=torch.float)

import torch.nn as nn

class WeightedTransformer(transformers.Trainer):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=class_weights.to("cuda" if torch.cuda.is_available() else "cpu"))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

args = transformers.TrainingArguments(
    output_dir="alzheimer_model_aug_deit60",
    overwrite_output_dir=True,
    remove_unused_columns=False,


    weight_decay=0.01,
    warmup_steps=6,

    logging_steps=1800//20,
    num_train_epochs=100,
    evaluation_strategy="epoch",
    per_device_eval_batch_size=32,
    per_device_train_batch_size=32,
    metric_for_best_model="accuracy",
    gradient_accumulation_steps=4,
    eval_accumulation_steps=3,


    load_best_model_at_end=True,
    greater_is_better=True,
    save_strategy="epoch",
    seed=1234,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers.optimization import Adafactor, AdafactorSchedule

optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
lr_scheduler = AdafactorSchedule(optimizer)

trainer = WeightedTransformer(
    model=model,
    args=args,
    data_collator=data_collator,
    train_dataset=ds_train,
    eval_dataset=ds_test,
    tokenizer=image_processor,
    compute_metrics=compute_metrics, #If numbers are dissapointing delete this line of code
)

trainer.train()



