#!/bin/bash

SAVE_DIR="test/data/iwslt_envi"
mkdir -p ${SAVE_DIR}

cd ${SAVE_DIR}

# train
curl -o train.en   https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.en
curl -o train.vi   https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.vi

# dev
curl -o tst2012.en https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2012.en
curl -o tst2012.vi https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2012.vi

# test
curl -o tst2013.en https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2013.en
curl -o tst2013.vi https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2013.vi

curl -o vocab.ori.en   https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/vocab.en
curl -o vocab.ori.vi   https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/vocab.vi

echo -e "<unk>\n<pad>\n<s>\n</s>" > vocab.en
echo -e "<unk>\n<pad>\n<s>\n</s>" > vocab.vi

tail -n +4 vocab.ori.en >> vocab.en
tail -n +4 vocab.ori.vi >> vocab.vi
