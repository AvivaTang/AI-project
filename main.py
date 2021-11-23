import numpy as np
import os
from datasets import load_dataset, load_metric
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer


raw_datasets = load_dataset("wmt16", "de-en")

print(len(raw_datasets["train"]))
print(raw_datasets["test"][0])
tokenizer = T5Tokenizer.from_pretrained('t5-base')

prefix = "translate German to English: "
source_lang = "de"
target_lang = "en"
max_input_length = 128
max_target_length = 128

def preprocess_function(examples):
    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

model = T5ForConditionalGeneration.from_pretrained('t5-base')

batch_size = 32
args = Seq2SeqTrainingArguments(
    "test-translation",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    fp16=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

metric = load_metric("sacrebleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    print(decoded_preds[0])
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"].shuffle(seed=42).select(range(100)),
    eval_dataset=tokenized_datasets["validation"].shuffle(seed=42).select(range(10)),
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

#train
train_result = trainer.train()
metrics = train_result.metrics
#print(metrics)
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

#eval
metrics = trainer.evaluate()
#print(metrics)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

#predict
predict_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(10))
predict_results = trainer.predict(predict_dataset)
metrics = predict_results.metrics
#print(metrics)
trainer.log_metrics("predict", metrics)
trainer.save_metrics("predict", metrics)

# save
print(tokenized_datasets["test"][0])
predictions = tokenizer.batch_decode(predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(predictions[0])
predictions = [pred.strip() for pred in predictions]
output_prediction_file = os.path.join("./sentences_bug", "generated_predictions2.txt")
with open(output_prediction_file, "w", encoding="utf-8") as writer:
    writer.write("\n".join(predictions))
