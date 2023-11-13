## lyricalGPT

A GPT-based language model trained on the Spotify 1 Million dataset to generate lyrics. The model
consists of decoder-only blocks stacked on top of one another, inspired by the Transformer network
outlined in the paper "Attention Is All You Need" by Vaswani et. al.

The model was built from scratch in PyTorch including a from-scratch implementation of Multi-Headed
Self-Attention. It was also trained for 30 epochs on a T4 GPU via Google Colab.

Still in progress.
