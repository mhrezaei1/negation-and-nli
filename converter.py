from safetensors.torch import load_file
import torch

# Load the safetensors model
safetensors_model_path = './negation-and-nli/downloaded_models/mhr2004/roberta-large-dual-500000-1e-06-128/model.safetensors'
state_dict = load_file(safetensors_model_path)

# Save as PyTorch model
pytorch_model_path = './negation-and-nli/downloaded_models/mhr2004/roberta-large-dual-500000-1e-06-128/pytorch_model.bin'
torch.save(state_dict, pytorch_model_path)
