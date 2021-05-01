import argparse
import torch
import time
import json
import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MBartForConditionalGeneration
from longformer.longformer_encoder_decoder import LongformerSelfAttentionForBart

parser = argparse.ArgumentParser(description='Necessary input data for the model')

parser.add_argument('--model_path', type=str, help='path to the model directory')
parser.add_argument('--tokenizer', type=str, default="facebook/mbart-large-cc25", help='name or path to the tokenizer')
parser.add_argument('--input_file', type=str, help='path to the input file')
parser.add_argument('--output_file', type=str, help='path for the output file')
parser.add_argument('--device', type=str, default="cpu", help='computing device')
parser.add_argument('--prefix', type=str, default="", required=False, help='prefix for the model if needed')
parser.add_argument('--eval_file', type=str, required=False, help='path for the evaluation file')
parser.add_argument('--input_max_length', type=int, default=1024, help='max length of the input sequence')
parser.add_argument('--sum_max_length', type=int, default=256, help='summary max length')
parser.add_argument('--num_beams', type=int, default=2, help='number of beams')
parser.add_argument('--length_penalty', type=float, default=1.0, help='length_penalty for abstractive summarization')
parser.add_argument('--batch_size', type=int, default=8, help='size of the batch for a summary')


args = parser.parse_args()

device = args.device if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

class LongformerEncoderDecoderForConditionalGeneration(MBartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        # # self.model.encoder.embed_positions =
        # for i, layer in enumerate(self.model.encoder.layers):
        #     layer.self_attn = LongformerSelfAttentionForBart(config, layer_id=i)
        # for i, layer in enumerate(self.model.decoder.layers):
        #     layer.self_attn = LongformerSelfAttentionForBart(config, layer_id=i)


model = LongformerEncoderDecoderForConditionalGeneration.from_pretrained(args.model_path).to(device)


src_list = open(args.input_file).readlines()

src_list_clean = []

for i in src_list:
    j = i.rstrip()
    src_list_clean.append(j)


# def generate_answer(batch, device):
#   inputs_dict = tokenizer(batch, padding="max_length", max_length=args.input_max_length, return_tensors="pt", truncation=True)
#   input_ids = inputs_dict.input_ids.to(device)
#   attention_mask = inputs_dict.attention_mask.to(device)
#   global_attention_mask = torch.zeros_like(attention_mask)
#   # put global attention on <s> token
#   global_attention_mask[:, 0] = 1
#
#   predicted_abstract_ids = model.generate(input_ids, attention_mask=attention_mask)
#   summary = tokenizer.batch_decode(predicted_abstract_ids, skip_special_tokens=True)
#   return str(summary)


if __name__ == '__main__':
    with open(args.output_file, 'w') as f:
        start_time = time.time()
        for input_string in src_list_clean[0:5]:

            # summary = generate_answer(batch=input_string, device=device)

            # input_ids = tokenizer(input_string, return_tensors="pt", max_length=args.input_max_length,
            #                       padding=True, truncation=True).input_ids.to(device)
            # global_attention_mask = torch.zeros_like(input_ids)
            # # set global_attention_mask on first token
            # global_attention_mask[:, 0] = 1
            # sequences = model.generate(input_ids, global_attention_mask=global_attention_mask, max_length=args.sum_max_length,
            #                            length_penalty=args.length_penalty, num_beams=args.num_beams, early_stopping=True).sequences
            # summary = tokenizer.batch_decode(sequences)

            inputs = tokenizer.encode(input_string, return_tensors='pt', padding="max_length",
                                      max_length=args.input_max_length, truncation=True).to(device)
            summary_ids = model.generate(inputs, max_length=args.sum_max_length, length_penalty=args.length_penalty,
                                    num_beams=args.num_beams, early_stopping=True)

            for g in summary_ids:
                summary = tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                print(summary)
                f.write(summary + "\n")
                f.flush()
        f.close()
        runtime = int(time.time() - start_time)
        n_obs = len(src_list_clean)
        print(dict(n_obs=n_obs, runtime=runtime, seconds_per_sample=round(runtime / n_obs, 4)))
        with open(args.eval_file, 'w') as jf:
            x = dict(n_obs=n_obs, runtime=runtime, seconds_per_sample=round(runtime / n_obs, 4))
            json.dump(x, jf, indent=3, ensure_ascii=False)