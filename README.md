# pytorch_skipgram

Skip-gram implementation with PyTorch.
This repo supports two loss functions: [negative sampling](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) and [noise contrastive estimation](https://papers.nips.cc/paper/5165-learning-word-embeddings-efficiently-with-noise-contrastive-estimation.pdf).

# Requirement

- Pytorch >= 1.0
- numpy

## Parameters

```
-h, --help            show this help message and exit
--window ws           the number of window size (default: 5)
--dim dim             the number of vector dimensions (default: 100)
--min-count min       threshold value for lower frequency words (default: 5)
--samples t           sub-sampling parameter (default: 1e-3)
--noise noise         power value of noise distribution (default: 0.75)
--negative neg        the number of negative samples (default: 5)
--epoch epochs        the number of epochs (default: 7)
--batch num_minibatches
                    the number of pairs of words (default: 512)
--lr_update_rate LR_UPDATE_RATE
                    update scheduler lr (default: 1000)
--lr starting_lr      initial learning rate (default: 0.025*num_minibatch)
--input fname         training corpus file name
--out outfname        vector file name
--loss LOSS           loss function name: neg (negative sampling) or nce
                    (noise contrastive estimation)
--gpu-id gpuid        gput id (default: -1, aka CPU)
--seed SEED           random seed value for numpy and pytorch.
```

## Run

### Download two data sets: `ptb` and `text8`
```bash
sh getdata.sh
```

```bash
python -m pytorch_skipgram.main --input=data/text8 --epoch=1 --out=text8.vec --min-count=5 --sample=1e-5 --batch=100 --negative=10 --gpu-id -1
```
