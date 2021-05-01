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


model = LongformerEncoderDecoderForConditionalGeneration.from_pretrained(args.model_path)


src_list = open(args.input_file).readlines()

src_list_clean = []

for i in src_list:
    j = i.rstrip()
    src_list_clean.append(j)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


if __name__ == '__main__':
    with open(args.output_file, 'a+') as f:
        start_time = time.time()
        for input_string in src_list_clean:
            inputs = tokenizer.encode(args.prefix + input_string, return_tensors='pt', max_length=args.input_max_length,
                                     padding=True, truncation=True).to(device)
            output = model.generate(inputs, max_length=args.sum_max_length, length_penalty=args.length_penalty,
                                    num_beams=args.num_beams)
            summary = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
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