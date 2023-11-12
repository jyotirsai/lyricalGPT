import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

from dataset import build_dataloader_and_tokenizers
from model import GPTModel

def get_weights_file_path(config, epochs: str):
  model_folder = config['model_folder']
  model_basename = config['model_basename']
  model_filename = f"{model_basename}{epochs}.pt"
  return str(Path('.') / model_folder / model_filename)

def train_model(config):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f'Using device {device}')

  train, val, tokenizer = build_dataloader_and_tokenizers(config)
  model = GPTModel(tokenizer.get_vocab_size(), config['embed_size'], config['n_heads'], config['context_size'], config['embed_size']*4, config['n_layers'], config['dropout'])
  model.to(device)

  optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])

  initial_epoch = 0
  global_step = 0
  if config['preload']:
    model_filename = get_weights_file_path(config, config['preload'])
    print(f'Preloading model: {model_filename}')
    state = torch.load(model_filename)
    initial_epoch = state['epoch'] + 1
    optimizer.load_state_dict(state['optimizer_state_dict'])
    global_step = state['global_step']

  for epoch in range(initial_epoch, config['epochs']):
    model.train()
    batch_iterator = tqdm(train, desc=f'Processing epoch {epoch:02d}')
    for (xb, yb) in batch_iterator:
      logits, loss = model(xb.to(device), yb.to(device))

      print("loss: ", loss)
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      global_step += 1
    
    # Save the model
    model_filename = get_weights_file_path(config, f'{epoch:02d}')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
    }, model_filename)