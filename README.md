# pytorch_skipgram

Skip-gram implementation with PyTorch.
This repo supports two loss functions: [negative sampling](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) and [noise contrastive estimation](https://papers.nips.cc/paper/5165-learning-word-embeddings-efficiently-with-noise-contrastive-estimation.pdf).

# Requirement

- Pytorch >= 1.0
- numpy

## Parameters

```
-h, --help            show this help message and exit
--window ws           the number of windows (default: 5)
--dim dim             the number of vector dimensions (default: 100)
--min-count min       threshold value for lower frequency words (default: 5)
--samples t           sub-sampling parameter (default: 1e-3)
--noise noise         power value of noise distribution (default: 0.75)
--negative neg        the number of negative samples (default: 5)
--epochs epochs       the number of epochs (default: 7)
--batch num_minibatches
                    the number of pairs of words (default: 512)
--lr_update_rate LR_UPDATE_RATE
                    update scheduler lr (default: 1000)
--lr starting_lr      initial learning rate (default: 0.025) internal
                    learning rate is `lr * num_minibatch`
--input fname         training corpus file name
--out outfname        vector file name
--loss LOSS           loss function name: neg (negative sampling) or nce
                    (noise contrastive estimation) (default: neg)
--gpu-id gpuid        gpu id (default: -1, aka CPU)
--seed SEED           random seed value for numpy and pytorch. (default: 7)
```

## Run

### Download two data sets: `ptb` and `text8`
```bash
sh getdata.sh
```

```bash
python -m pytorch_skipgram.main --input=data/text8 --dim=128 --epoch=5 --out=text8.vec --min-count=5 --sample=1e-4 --batch=16 --negative=15 --gpu-id -1
```

### Similarity task

```python
for w, s in model.most_similar(positive=["king"], topn=10):
    print(w, s)

canute 0.7516068816184998
sweyn 0.7161520719528198
haakon 0.715397298336029
plantagenet 0.7071711421012878
kings 0.7037447094917297
valdemar 0.703365683555603
omri 0.699432373046875
capet 0.6928986310958862
conqueror 0.6921138763427734
eochaid 0.690447986125946
```


### Analogical task

```python
for w, s in model.most_similar(positive=["king", "woman"], negative=["man"], topn=10):
    print(w, s)

queen 0.649447500705719
daughter 0.6051150560379028
anjou 0.6023151874542236
consort 0.595568060874939
son 0.5846152305603027
marries 0.5731959342956543
aquitaine 0.5700898170471191
isabella 0.568467378616333
infanta 0.5641375780105591
princess 0.5628763437271118
```
