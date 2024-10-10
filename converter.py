# from argparse import ArgumentParser

# args = ArgumentParser()
# args.add_argument('--path', type=str, required=True)
# args = args.parse_args()
from transformers import AutoModel, AutoTokenizer

model_paths = [
    "mhr2004/roberta-base-pp-500000-1e-06-128",
    "mhr2004/roberta-base-dual-1000000-1e-06-128",
    "mhr2004/roberta-large-pp-500000-1e-06-128",
    "mhr2004/roberta-large-nsp-1000000-1e-06-32",
    "mhr2004/roberta-large-dual-500000-1e-06-128",
    "mhr2004/roberta-base-nsp-1000000-1e-06-32",
]

class Args:
    def __init__(self, path):
        self.path = path

for path in model_paths:
    args = Args(path)
    AutoModel.from_pretrained(args.path).save_pretrained('./negation-and-nli/downloaded_models/mhr2004/' + args.path)
    AutoTokenizer.from_pretrained(args.path).save_pretrained('./negation-and-nli/downloaded_models/mhr2004/' + args.path)

    # import torch
    # from safetensors.torch import load_file

    # # Load the safetensors model
    # state_dict = load_file('./negation-and-nli/downloaded_models/mhr2004/' + args.path + '/model.safetensors')

    # # Save as PyTorch model
    # pytorch_model_path = './negation-and-nli/downloaded_models/mhr2004/roberta-large-dual-500000-1e-06-128/pytorch_model.bin'
    # pytorch_model_path = './negation-and-nli/downloaded_models/mhr2004/' + args.path + '/pytorch_model.bin'
    # torch.save(state_dict, pytorch_model_path)
