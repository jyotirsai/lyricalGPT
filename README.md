# lyricalGPT

A GPT-based language model trained on an English pop songs dataset to generate new song lyrics. The model
consists of decoder-only blocks stacked on top of one another, inspired by the Transformer network
outlined in the paper "Attention Is All You Need" by Vaswani et. al. The decoder block is simplified to help reduce computation for training.

The model was built from scratch in PyTorch including an implementation of Multi-Headed
Self-Attention. It was built for educational purposes and trained for 20 epochs on a T4 GPU via Google Colab. I would like to train for more epochs as well as add a learning rate scheduler in the future.

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

A BPE tokenizer from the HuggingFace tokenizers library was also used to tokenize the initial text.

## Inference

To generate lyrics, simply pass the config dictionary to the generate function.

```python
from config import get_config
from generate import generate

output = generate(config)
print(output[:600])
```

```
babe feet wear hell coming hold head Of ocean sorry play Santa somewhere Show hands kids Get Without Mr green longer pass alive comin won d again hundred against learned mad past minute against walked floor played alive y believe second wasn hair wish ago bring break His goes beat used cool until Black cause doin Before funny Its dance took men fast pretty keep The down So by Him what arms strong me cheese my gentle or hard Speaking blurred now window or down . Mix now real me thinking talking unafraid my dies " window As jammed old hurts on it weep into take turns all dusk in hear on My Alway
```

## Training

To train for additional epochs, simply pass the config to the training function as follows.

```python
from config import get_config
from train import train_model

train_model(config)
```

After each epoch, the state will automatically be saved to the current dictionary. To load in an existing state,
specify the the 'model_filename' path in config.py.

## Potential Improvements

The model outputs are quite nonsensical and improvements can be made in many areas. Firstly, the model was only trained
for 20 epochs due to a lack of computing power. Training for more epochs along with learning rate scheduler so improve
the results signficantly. The dataset is a 5 GB English song lyrics dataset from <a href="https://www.kaggle.com/datasets/razauhaq/english-songs-lyrics">Kaggle</a>. To reduce the compute time, only the pop songs from 2015 onwards was chosen which
resulted in about ~450,000 song lyrics (~500 MB). Including all of the pop songs, and testing out other genres, could also
help improve results.

## Requirements

<ul>
    <li>pandas==1.5.2</li>
    <li>tokenizers==0.15.0</li>
    <li>torch==2.1.0</li>
    <li>tqdm==4.64.0</li>
</ul>

## References

[1] - razauhaq, "English Songs Lyrics," Kaggle, 2023. [Online]. Available: https://www.kaggle.com/datasets/razauhaq/english-songs-lyrics.

[2] - A. Vaswani et al., "Attention is All You Need," arXiv preprint arXiv:1706.03762, 2017. [Online]. Available: https://arxiv.org/abs/1706.03762.

[3] - Eduardo Munoz, "Attention Is All You Need: Discovering the Transformer Paper," Towards Data Science, 2023. [Online]. Available: https://towardsdatascience.com/attention-is-all-you-need-discovering-the-transformer-paper-73e5ff5e0634.

[4] - J. Alammar, "The Illustrated Transformer," 2023. [Online]. Available: https://jalammar.github.io/illustrated-transformer/.

[5] - A. Karpathy, "Let's build GPT: from scratch, in code, spelled out." YouTube, 2019. [Online]. Available: https://www.youtube.com/watch?v=kCc8FmEb1nY.

[6] - U. Jamil, "Attention is all you need (Transformer) - Model explanation (including math), Inference and Training" YouTube, 2023. [Online]. Available: https://www.youtube.com/watch?v=bCz4OMemCcA.
