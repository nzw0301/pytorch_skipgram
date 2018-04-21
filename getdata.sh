#!/usr/bin/env bash
# https://raw.githubusercontent.com/salesforce/awd-lstm-lm/bf0742cab41d8bf4cd817acfe7e5e0cbff4131ba/getdata.sh

echo "=== Acquiring datasets ==="
echo "---"
mkdir -p data
cd data

echo "- Downloading Penn Treebank (PTB)"
mkdir -p penn
cd penn
wget --quiet --continue http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
tar -xzf simple-examples.tgz
mv simple-examples/data/ptb.train.txt train.txt
mv simple-examples/data/ptb.test.txt test.txt
mv simple-examples/data/ptb.valid.txt valid.txt
rm -rf simple-examples.tgz
rm -rf simple-examples/
cd ..

echo "- Downloading text8"
if [ ! -e text8 ]; then
  wget http://mattmahoney.net/dc/text8.zip -O text8.gz
  gzip -d text8.gz -f # for linux
fi
