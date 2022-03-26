import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel, AdamW, TrainingArguments, Trainer, pipeline # Transformers

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


model = TransformerModel()
model.load_state_dict(torch.load("best_transformer_model_v2.pt"))


total = 0
for tokens in tqdm(test_json):
    for word in tokens["tokens"]:
        if word == ' ' or word=='\xa0':
            total += 1
print(total)
model.to("cuda")
for tokens in tqdm(test_json):
    tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-ner")
    filtered_tokens = [text.replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș").replace("\n", "NTOK") for text in
        tokens["tokens"]]
    tokenized_input = tokenizer(filtered_tokens, add_special_tokens=True, max_length=100, padding='max_length', return_tensors='pt', truncation=True, is_split_into_words=True)
    # ner_model = pipeline('ner', model=model, tokenizer=tokenizer)
    indices = tokenized_input.word_ids()
    outputs = model(tokenized_input["input_ids"].to("cuda"))
    probs = F.softmax(outputs.to("cpu")[0], dim=-1)
    batch_preds = torch.argmax(probs, dim=1)
    # print(batch_preds)
    # print(tokens["tokens"])
    id = 0
    for idx, (pred, token, idw) in enumerate(zip(batch_preds, tokenized_input["input_ids"].squeeze(0), indices)):
        if idw is None:
            continue
        if idx > 0 and indices[idx - 1] is not None and indices[idx - 1] == idw:
            continue
        # print(id, pred.item())
        # with open("submission.csv", "a", encoding="utf-8") as f:
        #     # writer = csv.writer(f)
        #     writer.writerow((id, pred.item(), helper[id]))
        id += 1
    if id != len(tokens["tokens"]):
        print(tokens["tokens"])
        print(tokenized_input.word_ids())