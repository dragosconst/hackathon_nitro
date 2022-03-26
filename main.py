import roner
import fasttext # Static word embeddings
import fasttext.util
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import torch # Neural Networks
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel, AdamW, TrainingArguments, Trainer, pipeline # Transformers
from datasets import load_metric
from sklearn.manifold import TSNE # Data projection
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from skmultilearn.model_selection import iterative_train_test_split, IterativeStratification

from tqdm import tqdm # Progress bar
import seaborn as sns
import os
import random
from IPython import display
import json
import time
import csv

sns.set_style("darkgrid")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

seed = 31
torch.manual_seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = "/content/drive/MyDrive/nitro-lang-processing-1.zip"

train_json = None
test_json = None
with open("train.json", "r") as f:
  train_json = json.load(f)
with open("test.json", "r") as f:
  test_json = json.load(f)
PAD = 'PAD'
# train_json[idx] e de forma Dict("ner_ids": [...], "ner_tags": [...], "space_after": [...], "tokens": [...])
tag_to_id = {"O": 0, "PERSON": 1, "QUANTITY": 12, "NUMERIC": 13, "NAT_REL_POL": 5, "GPE": 3, "DATETIME": 9, "ORG": 2, "PERIOD": 10, "EVENT": 6, "FACILITY": 15, "ORDINAL": 14, "LOC": 4, "MONEY": 11, "WORK_OF_ART": 8, "LANGUAGE": 7}

from typing import Iterable


class HackathonDataset(Dataset):
  def __init__(self, json_f):
    self.json_f = json_f
    self.tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-ner")

  def __len__(self):
    return len(self.json_f)

  def __getitem__(self, idx):
    if not isinstance(idx, Iterable):
      idx = [idx]
    data = []
    targets = []
    for id in idx:
      json_item = self.json_f[id]
      ids = json_item["ner_ids"][:120]
      # if len(ids) < 100:
      #   ids.extend([0 for i in range(100 - len(ids))])
      # data.append(tokens)
      # targets.append(torch.as_tensor([0] + ids + [0]))

      # text = ""
      # for token, space_req in zip(json_item["tokens"], json_item["space_after"]):
      #   text += token
      #   if space_req:
      #     text += " "
      # Remove cedilla diacritics as suggested in
      # https://huggingface.co/dumitrescustefan/bert-base-romanian-uncased-v1
      filtered_tokens = [
        text.replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș").replace("\n", "NTOK").
          replace(' ', "WS").replace("\xa0", "WS") for text in
                         json_item["tokens"]]

      # Tokenize text and return the tensor
      # Maximum length set at 64 tokens for efficiency reasons. The maximum sequence length for BERT is 512.
      tokenized_bert = self.tokenizer(filtered_tokens, add_special_tokens=True, max_length=120, padding='max_length',
                                      return_tensors='pt', truncation=True, is_split_into_words=True)
      word_ids = tokenized_bert.word_ids()
      data.append(tokenized_bert["input_ids"].squeeze(0))
      aligned_ids = [0 if tok is None else ids[tok] for tok in word_ids]
      targets.append(torch.as_tensor(aligned_ids).squeeze(0))
    return torch.stack(data), torch.stack(targets)

  def targets(self):
    targets = []
    for idx in range(len(self.json_f)):
      ids = self.json_f[idx]["ner_ids"][:100]
      if len(ids) < 100:
        ids.extend([0 for i in range(100 - len(ids))])

      targets.append(ids)
    return targets


class HackathonTestDataset(Dataset):
  def __init__(self, json_f):
    self.json_f = json_f
    self.tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-ner")

  def __len__(self):
    return len(self.json_f)

  def __getitem__(self, idx):
    if not isinstance(idx, Iterable):
      idx = [idx]
    data = []
    for id in idx:
      json_item = self.json_f[id]

      # text = ""
      # for token, space_req in zip(json_item["tokens"], json_item["space_after"]):
      #   text += token
      #   if space_req:
      #     text += " "
      # Remove cedilla diacritics as suggested in
      # https://huggingface.co/dumitrescustefan/bert-base-romanian-uncased-v1
      # text = text.replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș")
      filtered_tokens = [
        text.replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș").replace("\n", "NTOK").
          replace(' ', "WS").replace("\xa0", "WS") for text in
                         json_item["tokens"]]

      # Tokenize text and return the tensor
      # Maximum length set at 64 tokens for efficiency reasons. The maximum sequence length for BERT is 512.
      tokenized_bert = self.tokenizer(filtered_tokens, add_special_tokens=True, max_length=120, padding='max_length',
                                      return_tensors='pt', truncation=True, is_split_into_words=True)
      data.append(tokenized_bert["input_ids"].squeeze(0))
    return torch.stack(data)

ds_train = HackathonDataset(train_json)
indices = np.asarray([x for x in range(len(ds_train))])
indices = indices[..., np.newaxis] # N x len(ds_train) -> 1 x len(ds_train)
targets = ds_train.targets()
# print(np.average([len(t) for t in targets]))
# max_target = np.max([len(t) for t in targets])
# pad_targets(targets, max_target)
train_idx, _, val_idx, _ = iterative_train_test_split(indices, np.asarray(targets), test_size=0.2)
train_set = torch.utils.data.Subset(ds_train, train_idx)
val_set = torch.utils.data.Subset(ds_train, val_idx)

ds_test = HackathonTestDataset(test_json)

BATCH_SIZE=32
dl_train = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True
)
dl_valid = DataLoader(
  val_set, batch_size=BATCH_SIZE, shuffle=False
)
dl_test = DataLoader(
    ds_test, batch_size=BATCH_SIZE, shuffle=False
)

class TransformerModel(nn.Module):
  def __init__(self, num_classes=16):
      super(TransformerModel, self).__init__()
      # Get the romanian Transformer from the huggingface server
      self.transformer = AutoModelForTokenClassification.from_pretrained("dumitrescustefan/bert-base-romanian-ner")
      self.fc = nn.Linear(31, num_classes)
  def forward(self, x):
      out = x.squeeze(1)
      # Get output from Transformer.
      out = self.transformer(out)[0]
      # for batch_out in out:
      #   for idw, word in enumerate(batch_out):
      #     new_word = [word[0]]
      #     for idp, pred in enumerate(word[1:]):
      #       new_word.append(max(word[idp], word[idp + 1]))
      #       idp += 2
      #     batch_out[idw] = torch.as_tensor(new_word)
    #   # We usually add dropout before the final classification layer when using a Transformer
    #   out = F.dropout(out, p=0.1)
      out = self.fc(out)
      return out

# Instantiate our model and move it to GPU
# model = TransformerModel().to(device)
model = TransformerModel().to(device)

# Freeze all the Transformer parameters
for p in model.transformer.parameters():
  p.requires_grad = False
# ... except for the bias terms
trainable_params_transformer = [p for (n, p) in model.transformer.named_parameters() if "bias" in n]
for p in trainable_params_transformer:
  p.requires_grad = True

# We'll train the final layer and the bias terms
trainable_params = list(model.fc.parameters())
# trainable_params = []
trainable_params.extend(list(trainable_params_transformer))

# Define our loss and optimizer
#                         [0, 1,  2,  3,  4,   5,    6,  7,   8,   9,  10,  11,  12,  13,  14,  15]
# weights = torch.as_tensor([1, 10, 20, 30, 120, 120, 150, 400, 100, 15, 150, 140, 170, 100, 170, 150 ]).float().to("cuda")
weights = torch.as_tensor([1, 32, 62, 100, 227, 250, 357, 2000, 181, 58, 357, 352, 476, 158, 416, 384 ]).float().to("cuda")
loss_fn = nn.CrossEntropyLoss(weight=weights)
# We'll use AdamW for the Transformer
optim = AdamW(trainable_params)

no_epochs = 5
best_val_loss = None


def train_epoch(model, optim, loss_fn, dataloader, epoch_idx):
  """ Trains the model for one epoch and returns the loss together with a classification report
  """

  epoch_loss = 0
  # Put the model in training mode
  model.train()
  preds = []
  gt = []

  times = []
  for idx, batch in enumerate(dataloader):
    beg = time.time()
    # Reset gradients
    optim.zero_grad()

    inputs, labels = batch
    # Move data to GPU
    inputs = inputs.to(device)
    labels = labels.to(device)
    labels = labels.squeeze(1)

    output = model(inputs.to(device))
    output = output.reshape(-1, 16)
    labels = labels.reshape(-1)
    # Calculate the loss and backpropagate
    loss = loss_fn(output, labels)
    loss.backward()
    # Update weights
    optim.step()

    final = time.time()
    times.append(final - beg)
    if idx % 10 == 0:
      print(
        f"Loss is {loss}, time left = {(sum(times) / len(times) * (len(dataloader.dataset) - idx * BATCH_SIZE - BATCH_SIZE)) / 60} minutes.")

    epoch_loss += loss.item()

    probs = F.softmax(output, dim=-1)
    batch_preds = torch.argmax(probs, dim=1)

    preds.append(batch_preds.cpu().numpy())
    gt.append(labels.cpu().numpy())

  # Average the epoch losses
  epoch_loss /= len(dataloader)

  preds = np.concatenate(preds)
  gt = np.concatenate(gt)

  print(f"Acc on train: {accuracy_score(gt, preds) * 100}%")
  # Get an epoch classification report
  clf_report = classification_report(gt, preds, output_dict=True)

  return epoch_loss, clf_report


@torch.inference_mode()
def test(model, loss_fn, dataloader):
  """ Computes and returns the loss and classification report for a testing dataset.
  """

  test_loss = 0
  # put model in evaluation mode
  model.eval()
  preds = []
  gt = []

  # Tell PyTorch that we won't be computing gradients
  for idx, batch in enumerate(dataloader):
    inputs, labels = batch

    inputs = inputs.to(device)
    labels = labels.to(device)

    output = model(inputs)

    output = output.reshape(-1, 16)
    labels = labels.reshape(-1)
    test_loss += loss_fn(output, labels).item()

    probs = F.softmax(output, dim=-1)
    batch_preds = torch.argmax(probs, dim=1)

    preds.append(batch_preds.cpu().numpy())
    gt.append(labels.cpu().numpy())

  test_loss /= len(dataloader)
  preds = np.concatenate(preds)
  gt = np.concatenate(gt)

  print(f"Acc on val: {accuracy_score(gt, preds) * 100}%")
  # Get a classification report
  clf_report = classification_report(gt, preds, output_dict=True)
  clf_report_text = classification_report(gt, preds)

  return test_loss, clf_report, clf_report_text


pbar = tqdm(range(no_epochs))

train_losses = []
val_losses = []

train_accs = []
val_accs = []

for e in pbar:
  train_loss, clf_report_train = train_epoch(model, optim, loss_fn, dl_train, e)
  val_loss, clf_report_val, clf_text = test(model, loss_fn, dl_valid)
  print(clf_text)

  train_acc = clf_report_train['accuracy'] * 100
  val_acc = clf_report_val['accuracy'] * 100

  train_losses.append(train_loss)
  val_losses.append(val_loss)

  train_accs.append(train_acc)
  val_accs.append(val_acc)

  pbar.set_description(
    "Epoch: %s, t_loss: %.3f, v_loss: %.3f, t_acc: %.2f, v_acc: %.2f" % (
    e + 1, train_loss, val_loss, train_acc, val_acc))

  if best_val_loss is None or val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), './best_transformer_model.pt')

