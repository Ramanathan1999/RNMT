from transformers import AutoTokenizer
from datasets import load_dataset, load_metric
from tqdm import tqdm
raw_datasets = load_dataset("wmt16", "de-en")
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM
from evaluate import load
from transformers import DataCollatorForSeq2Seq
import os
import torch
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from torch import nn
from torch.distributions import Categorical
import sys
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

bleu_metric = load_metric("sacrebleu")
meteor_metric = load("meteor")
bert_metric = load("bertscore")
tokenizer = AutoTokenizer.from_pretrained("t5-small")
prefix = "translate English to German: "
max_input_length = 128
max_target_length = 128
source_lang = "en"
target_lang = "de"

def preprocess_function(examples):
    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = load_from_disk('./tokenizer')


model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
model.cuda()

i = 0
for name, param in model.named_parameters():
  if i == 0:
    i += 1
    continue
  split_name = name.split(".")
  if split_name[2] == '0' or split_name[2] == '1' or split_name[2] == '2':
    param.requires_grad = False

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
os.environ['WANDB_ENTITY']='rl4nmt'
os.environ['WANDB_PROJECT']='rl4nmt'

def compute_metrics(pred):
  labels_ids = pred.label_ids
  pred_ids = pred.predictions[0]

  pred_ids[pred_ids == -100] = tokenizer.pad_token_id
  pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
  labels_ids[labels_ids == -100] = tokenizer.pad_token_id
  label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

  bert_score = 0
  bert_results = bert_metric.compute(predictions=pred_str, references=label_str, model_type="distilbert-base-uncased")
  bert_score += bert_results['f1'][0]

  label_str_list = [[i] for i in label_str]
  bleu_score = 0
  bleu_results = bleu_metric.compute(predictions=pred_str, references=label_str_list)
  bleu_score += bleu_results['score']

  meteor_score = 0
  meteor_results = meteor_metric.compute(predictions=pred_str, references=label_str_list)
  meteor_score += meteor_results['meteor']

  out_dict = {'bert_score': bert_score, 'bleu_score': bleu_score, 'meteor_score': meteor_score}

  return out_dict


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels

metric = bleu_metric



class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bleu = []
        self.distributions = []
        self.counter = 0 

    def compute_loss(self, model, inputs, return_outputs=False):
        log_probs = []
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        attn_mask = torch.where(labels != -100, 1, 0)
        masked_labels = labels*attn_mask
        decoded_labels = tokenizer.batch_decode(masked_labels,skip_special_tokens=True)
        decoded_labels = [[i] for i in decoded_labels]
        
        attn_mask = torch.unsqueeze(attn_mask, 2)
        masked_logits = logits*attn_mask
        decoded_logits = []
        log_probs_batch = []
        cat = Categorical(logits=masked_logits)
        samplebatch = cat.sample()
        log_probs_batch = cat.log_prob(samplebatch)
        log_probs_batch = torch.mean(log_probs_batch,1)
        decoded_logits = tokenizer.batch_decode(samplebatch, skip_special_tokens=True)
        bleu_scores = [metric.compute(predictions=[decoded_logits[i]], references=[decoded_labels[i]])["score"] for i in range(len(decoded_labels))]
        loss = sum([-log_prob*bleu_score for log_prob, bleu_score in zip(log_probs_batch, bleu_scores)])
        return (loss, outputs) if return_outputs else loss

training_args = Seq2SeqTrainingArguments(
    output_dir="./resultsbleu",
    evaluation_strategy="steps",
    eval_steps=25000,
    save_strategy="epoch", 
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=10,
    num_train_epochs=5,
    fp16=True,
    report_to="wandb",
    run_name="T5-Small-BLEURL-Finetuning-test"
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics
)

trainer.train()