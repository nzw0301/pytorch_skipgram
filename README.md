# pytorch_skipgram

Skip-gram implementation with PyTorch.
This repo supports two loss functions: [negative sampling](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) and [noise contrastive estimation](https://papers.nips.cc/paper/5165-learning-word-embeddings-efficiently-with-noise-contrastive-estimation.pdf).

# Requirement

This code supports pytorch v0.4.

## Parameter

- `--window`: the number of window size. Default=5.
- `--dim` : the number of vector dimensions. Default=100.
- `--min-count`: threshold value for lower frequency words. Default=5.
- `--samples`: sub-sampling parameter. Default=1e-3.
- `--noise`: parameter of noise distribution, `np.pow(word_freq, noise)` default: 0.75.
- `--negative`: the number of negative samples. Default=5.
- `--epoch`: the number of epochs. Default=7.
- `--batch`: the number of pairs of words. Default=512.
- `--lr_update_rate`: after processing every this number of words, lr is updated. Default=1000.
- `--lr` :initial learning rate. Note that `lr` multiplies `batch` in this code. Default: 0.025.
- `--input`: training corpus file name.
- `--out`: vector file name. format is word2vec's text format.
- `--loss`: loss function name: neg (negative sampling) or nce (noise contrastive estimation). Default: neg.

## Run

```bash
python -m pytorch_skipgram.explicit_main.py --input=data/text8 --epoch=1 --out=text8.vec --min-count=5 --sample=1e-5 --batch=100 --negative=10
```
