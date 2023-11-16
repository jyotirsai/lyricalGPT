import torch
from tokenizers import Tokenizer
import torch.nn.functional as F

from model import GPTModel
from pathlib import Path

def generate(config, max_new_tokens: int = 500):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("Using device:", device)

  tokenizer = Tokenizer.from_file(str(Path('./'+config['tokenizer_file'])))

  model = GPTModel(tokenizer.get_vocab_size(), config['embed_size'], config['n_heads'], config['context_size'], config['embed_size']*4, config['n_layers'], config['dropout'])
  model.to(device)

  state = torch.load('./'+config['model_filename'])
  model.load_state_dict(state['model_state_dict'])

  model.eval()

  inputs = torch.randint(100,1000,(config['context_size'], config['context_size']), dtype=torch.long, device=device)
  with torch.no_grad():
    for _ in range(max_new_tokens):
      inputs_cropped = inputs[:, -config['context_size']:]
      logits, _ = model(inputs_cropped.to(device), None)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)

      inputs_next = torch.multinomial(probs, 1)
      inputs = torch.cat((inputs, inputs_next), dim=1)
  
  return tokenizer.decode(inputs.squeeze()[0].tolist())