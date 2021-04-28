import torch
#from pytorch_lightning import LightingModule
from transformers import AutoModelForSeq2SeqLM, MBartForConditionalGeneration

PATH = "../longformer/models/LongTinyMBART"

model = AutoModelForSeq2SeqLM.from_pretrained(PATH)


# model_torch = torch.load(PATH)
# model = model_torch

# model_lightning = MyLightingModule.load_from_checkpoint(PATH)
# model = model_lightning


print(sum([param.nelement() for param in model.parameters()]))

if __name__ == '__main__':
    with open("model_summary/LongTinyMBART.txt", 'w') as f:
        f.write(repr(model))
        f.close()

    print(repr(model))