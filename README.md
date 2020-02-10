# pytorch_skipgram

Skip-gram implementation with PyTorch.
This repo supports two loss functions: [negative sampling](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) and [noise contrastive estimation](https://papers.nips.cc/paper/5165-learning-word-embeddings-efficiently-with-noise-contrastive-estimation.pdf).

# Requirement

- PyTorch >= 1.0
- numpy
- hydra

## Parameters

See [`conf/config.yaml`](./conf/config.yaml).

Default parameters are as follows:

```bash
$ python -m pytorch_skipgram.main --cfg job                                                                              [16:46:18]
dataset:
  input_path: ../../../data/text8
  outout_file_name: text8.vec
experiments:
  gpu_id: -1
  seed: 7
parameters:
  batch: 512
  dim: 100
  epochs: 7
  loss: neg
  lr: 0.025
  lr_update_rate: 1000
  min_count: 5
  negative: 5
  noise: 0.75
  samples: 0.001
  window: 5
```

## Run

### Download two data sets: `text8` and `ptb` 

```bash
sh getdata.sh
```

```bash
python -m pytorch_skipgram.main # train on text8
python -m pytorch_skipgram.main dataset=ptb # train on penn treebank
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
