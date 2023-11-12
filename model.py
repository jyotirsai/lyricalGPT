import torch
import torch.nn as nn

import torch.nn.functional as F
import math

class FeedForwardBlock(nn.Module):
  def __init__(self, embed_size: int, ff_size: int, dropout: float) -> None:
    super().__init__()
    self.linear_1 = nn.Linear(embed_size, ff_size)
    self.dropout = nn.Dropout(dropout)
    self.linear_2 = nn.Linear(ff_size, embed_size)

  def forward(self, x):
    # (batch_size, context_size, embed_size) --> (batch_size, context_size, ff_size) --> (batch_size, context_size, embed_size)
    return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class Head(nn.Module):
  def __init__(self, embed_size: int, head_size: int, context_size: int, dropout: float):
    super().__init__()
    self.query = nn.Linear(embed_size, head_size)
    self.key = nn.Linear(embed_size, head_size)
    self.value = nn.Linear(embed_size, head_size)
    self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    # (batch_size, context_size, embed_size) --> (batch_size, context_size, head_size)

    # (batch_size, context_size, head_size)
    q, k, v = self.query(x), self.key(x), self.value(x)

    # (batch_size, context_size, head_size) @ (batch_size, head_size, context_size) --> (batch_size, context_size, context_size)
    attention_scores = (q @ k.transpose(-2,-1)) / math.sqrt(k.shape[-1])
    attention_scores = attention_scores.masked_fill(self.tril[:, :] == 0, float('-inf'))
    attention_scores = F.softmax(attention_scores, dim=-1)
    attention_scores = self.dropout(attention_scores)

    # (batch_size, context_size, context_size) @ (batch_size, context_size, head_size) --> (batch_size, context_size, head_size)
    out = attention_scores @ v
    return out
  
class MultiHeadAttention(nn.Module):
  def __init__(self, embed_size: int, head_size: int, n_heads: int, context_size: int, dropout: float):
    super().__init__()
    self.heads = nn.ModuleList([Head(embed_size, head_size, context_size, dropout) for _ in range(n_heads)])
    self.linear = nn.Linear(head_size * n_heads, embed_size)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(out)
    return out

class DecoderBlock(nn.Module):
  def __init__(self, embed_size: int, n_heads: int, context_size: int, ff_size: int, dropout: float) -> None:
    super().__init__()
    assert embed_size % n_heads == 0, "embed_size is not divisible by n_heads"
    head_size = embed_size // n_heads
    self.multi_head_attention = MultiHeadAttention(embed_size, head_size, n_heads, context_size, dropout)
    self.feed_forward = FeedForwardBlock(embed_size, ff_size, dropout)
    self.lnorm = nn.ModuleList([nn.LayerNorm(embed_size) for _ in range(2)])
  
  def forward(self, x):
    x = x + self.multi_head_attention(self.lnorm[0](x))
    x = x + self.feed_forward(self.lnorm[1](x))
    return x

class GPTModel(nn.Module):
  def __init__(self, vocab_size: int, embed_size: int, n_heads: int, context_size: int, ff_size: int, n_layers: int, dropout: float) -> None:
    super().__init__()
    self.embeds = nn.Embedding(vocab_size, embed_size)
    self.pos_embeds = nn.Embedding(context_size, embed_size)
    self.decoder = nn.Sequential(*[DecoderBlock(embed_size, n_heads, context_size, ff_size, dropout) for _ in range(n_layers)])
    self.fnorm = nn.LayerNorm(embed_size)
    self.linear = nn.Linear(embed_size, vocab_size)

  def forward(self, inputs, targets=None):
    batch_size, context_size = inputs.shape
    
    embeds = self.embeds(inputs)
    pos_embeds = self.pos_embeds(torch.arange(context_size).to(inputs.device))
    x = embeds + pos_embeds
    x = self.decoder(x)
    x = self.fnorm(x)
    logits = self.linear(x)

    if targets is not None:
      batch_size, context_size, embed_size = logits.shape
      logits = logits.view(batch_size*context_size, embed_size)
      targets = targets.view(batch_size*context_size)
      loss = F.cross_entropy(logits, targets)
    else:
      loss = None
    
    return logits, loss