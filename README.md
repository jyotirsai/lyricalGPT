# lyricalGPT

A GPT-based language model trained on an English pop songs dataset to generate new song lyrics. The model
consists of decoder-only blocks stacked on top of one another, inspired by the Transformer network
outlined in the paper "Attention Is All You Need" by Vaswani et. al.

The model was built from scratch in PyTorch including an implementation of Multi-Headed
Self-Attention. It was also trained for 20 epochs on a T4 GPU via Google Colab.

## Model

The model consists of an input and positional embedding followed by 6 decoder blocks stacked
on top of one another with a final linear layer with softmax. The table includes other parameters
of the model as well as some training info.

| Parameter      | Value |
| -------------- | ----- |
| Learning Rate  | 3e-4  |
| Optimizer      | Adam  |
| Batch size     | 256   |
| Embedding size | 512   |
| Dropout        | 0.1   |
| Heads          | 6     |
| Context size   | 64    |

## Potential Improvements

The model outputs are quite nonsensical and improvements can be made in many areas. Firstly, the model was only trained
for 20 epochs due to a lack of computing power. Training for more epochs along with learning rate scheduler so improve
the results signficantly. The dataset is a 5 GB English song lyrics dataset from <a href="https://www.kaggle.com/datasets/razauhaq/english-songs-lyrics">Kaggle</a>. To reduce the compute time, only the pop songs from 2015 onwards was chosen which
resulted in about ~450,000 song lyrics (~500 MB). Including all of the pop songs, and testing out other genres, could also
help improve results.

## Requirements

## References

[1] - razauhaq, "English Songs Lyrics," Kaggle, 2023. [Online]. Available: https://www.kaggle.com/datasets/razauhaq/english-songs-lyrics.

[2] - A. Vaswani et al., "Attention is All You Need," arXiv preprint arXiv:1706.03762, 2017. [Online]. Available: https://arxiv.org/abs/1706.03762.

[3] - Eduardo Munoz, "Attention Is All You Need: Discovering the Transformer Paper," Towards Data Science, 2023. [Online]. Available: https://towardsdatascience.com/attention-is-all-you-need-discovering-the-transformer-paper-73e5ff5e0634.

[4] - J. Alammar, "The Illustrated Transformer," 2023. [Online]. Available: https://jalammar.github.io/illustrated-transformer/.

[5] - A. Karpathy, "Let's build GPT: from scratch, in code, spelled out." YouTube, 2019. [Online]. Available: https://www.youtube.com/watch?v=kCc8FmEb1nY.

[6] - U. Jamil, "Attention is all you need (Transformer) - Model explanation (including math), Inference and Training" YouTube, 2023. [Online]. Available: https://www.youtube.com/watch?v=bCz4OMemCcA.
