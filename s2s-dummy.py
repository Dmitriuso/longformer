from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MBartForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from longformer.longformer_encoder_decoder import LongformerSelfAttentionForBart

tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")

class LongformerEncoderDecoderForConditionalGeneration(MBartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        # # self.model.encoder.embed_positions =
        # for i, layer in enumerate(self.model.encoder.layers):
        #     layer.self_attn = LongformerSelfAttentionForBart(config, layer_id=i)
        # for i, layer in enumerate(self.model.decoder.layers):
        #     layer.self_attn = LongformerSelfAttentionForBart(config, layer_id=i)


model = LongformerEncoderDecoderForConditionalGeneration.from_pretrained("../LongMBART-25-2K")
#model = MBartForConditionalGeneration.from_pretrained("./model3")

train_dataset = load_dataset("scientific_papers", "pubmed", split="train", cache_dir="../datasets")
val_dataset = load_dataset("scientific_papers", "pubmed", split="validation", cache_dir="../datasets")

max_input_length = 2048
max_output_length = 256
batch_size = 1


def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = tokenizer(
        batch["article"],
        padding="max_length",
        truncation=True,
        max_length=max_input_length,
    )
    outputs = tokenizer(
        batch["abstract"],
        padding="max_length",
        truncation=True,
        max_length=max_output_length,
    )

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask

    # create 0 global_attention_mask lists
    batch["global_attention_mask"] = len(batch["input_ids"]) * [
        [0 for _ in range(len(batch["input_ids"][0]))]
    ]

    # since above lists are references, the following line changes the 0 index for all samples
    batch["global_attention_mask"][0][0] = 1
    batch["labels"] = outputs.input_ids

    # We have to make sure that the PAD token is ignored
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]

    return batch


train_dataset = train_dataset.select(range(50000))
val_dataset = val_dataset.select(range(5000))


train_dataset = train_dataset.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
    remove_columns=["article", "abstract", "section_names"],
)

val_dataset = val_dataset.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
    remove_columns=["article", "abstract", "section_names"],
)


train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)
val_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)


rouge = load_metric("rouge")


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(
        predictions=pred_str, references=label_str, rouge_types=["rouge2"]
    )["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    #no_cuda=True,  # Run on CPU
    sharded_ddp="simple",
    #tpu_num_cores=1,
    fp16=True,
    output_dir="./LongMBART-pubmed",
    overwrite_output_dir=True,
    logging_steps=5,
    save_steps=1000,
    save_total_limit=2,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)


if __name__ == '__main__':
  trainer.train()