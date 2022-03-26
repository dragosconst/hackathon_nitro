from typing import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import csv
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel, AdamW, TrainingArguments, Trainer, pipeline # Transformers
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel, AdamW, TrainingArguments, Trainer, pipeline # Transformers

class TestHelper():
    def __init__(self, js_test):
        self.js_test = js_test
    def __getitem__(self, idx):
        i = 0
        for js_dict in self.js_test:
            if i + len(js_dict["tokens"]) > idx:
                return(js_dict["tokens"][idx - i])
            else:
                i += len(js_dict["tokens"])
    def __len__(self):
        return sum([len(js_dict["tokens"]) for js_dict in self.js_test])
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
          (text.replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș").replace("\n", "NTOK")).
          replace(' ', "WS").replace("\xa0", "WS") for text in
          json_item["tokens"]]

      # Tokenize text and return the tensor
      # Maximum length set at 64 tokens for efficiency reasons. The maximum sequence length for BERT is 512.
      tokenized_bert = self.tokenizer(filtered_tokens, add_special_tokens=True, max_length=120, padding='max_length',
                                      return_tensors='pt', truncation=True, is_split_into_words=True)
      data.append((tokenized_bert["input_ids"].squeeze(0), tokenized_bert.word_ids(), filtered_tokens))
    return data

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

BATCH_SIZE=32
ds_test = HackathonTestDataset(test_json)

def collate_fn(data):
    tokens = []
    word_inds = []
    words_original = []
    for idx in range(len(data)):
        tokens.append(data[idx][0][0])
        word_inds.append(data[idx][0][1])
        words_original.append(data[idx][0][2])
    return tokens, word_inds, words_original
dl_test = DataLoader(
    ds_test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)

model = TransformerModel()
model.load_state_dict(torch.load("best_transformer_model_v5.pt"))
model.to("cuda")

@torch.inference_mode()
def write_submission(model, dl_test):
    model.eval()

    helper = TestHelper(test_json)
    with open("submission.csv", "w") as f:
        writer = csv.writer(f)
        header = ['Id', 'ner_label']
        writer.writerow(header)

    print(len(helper))
    id = 0
    for batch in tqdm(dl_test):

        inputs, indices, original_words = batch
        inputs = torch.stack(inputs)
        inputs = inputs.to("cuda")
        # print(inputs)

        outputs = model(inputs)
        outputs = outputs.reshape(-1, 16)

        probs = F.softmax(outputs, dim=-1)
        input = inputs.to("cpu").reshape(-1)
        indices = sum(indices, []) # trick to get flatten the list
        batch_preds = torch.argmax(probs, dim=1)
        for idx, (pred, token, idw) in enumerate(zip(batch_preds, input, indices)):
            if idw is None:
                continue
            if idx > 0  and indices[idx - 1] == idw:
                continue
            with open("submission.csv", "a", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow((id, pred.item()))
            id += 1

write_submission(model, dl_test)