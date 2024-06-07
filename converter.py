from argparse import ArgumentParser
args = ArgumentParser()
args.add_argument('--path', type=str, required=True)
args = args.parse_args()

from transformers import AutoModel, AutoTokenizer

AutoModel.from_pretrained('mhr2004/' + args.path).save_pretrained('./negation-and-nli/downloaded_models/mhr2004/' + args.path)
AutoTokenizer.from_pretrained('mhr2004/' + args.path).save_pretrained('./negation-and-nli/downloaded_models/mhr2004/' + args.path)

from safetensors.torch import load_file
import torch
# Load the safetensors model
state_dict = load_file('./negation-and-nli/downloaded_models/mhr2004/' + args.path + '/model.safetensors')

# Save as PyTorch model
pytorch_model_path = './negation-and-nli/downloaded_models/mhr2004/roberta-large-dual-500000-1e-06-128/pytorch_model.bin'
pytorch_model_path = './negation-and-nli/downloaded_models/mhr2004/' + args.path + '/pytorch_model.bin'
torch.save(state_dict, pytorch_model_path)
