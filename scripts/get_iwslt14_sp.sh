#!/usr/bin/env bash

# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh
# Adapted from https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-iwslt14.sh

git clone https://github.com/moses-smt/mosesdecoder.git

MOSES=`pwd`/mosesdecoder

SCRIPTS=${MOSES}/scripts
TOKENIZER=${SCRIPTS}/tokenizer/tokenizer.perl
LC=${SCRIPTS}/tokenizer/lowercase.perl
CLEAN=${SCRIPTS}/training/clean-corpus-n.perl
URL="http://dl.fbaipublicfiles.com/fairseq/data/iwslt14/de-en.tgz"
GZ=de-en.tgz

vocab_size=32000
src=de
tgt=en
lang=de-en
prep="../test/data/iwslt14_sp"
tmp=${prep}/tmp
orig=orig

mkdir -p ${orig} ${tmp} ${prep}

echo "Downloading data from ${URL}..."
cd ${orig}
curl -O "${URL}"

if [ -f ${GZ} ]; then
    echo "Data successfully downloaded."
else
    echo "Data not successfully downloaded."
    exit
fi

tar zxvf ${GZ}
cd ..

echo "pre-processing train data..."
for l in ${src} ${tgt}; do
    f=train.tags.$lang.$l
    tok=train.tags.$lang.tok.$l

    cat ${orig}/${lang}/${f} | \
    grep -v '<url>' | \
    grep -v '<talkid>' | \
    grep -v '<keywords>' | \
    sed -e 's/<title>//g' | \
    sed -e 's/<\/title>//g' | \
    sed -e 's/<description>//g' | \
    sed -e 's/<\/description>//g' | \
    perl ${TOKENIZER} -threads 8 -l $l > ${tmp}/${tok}
    echo ""
done
perl ${CLEAN} -ratio 1.5 ${tmp}/train.tags.${lang}.tok ${src} ${tgt} ${tmp}/train.tags.${lang}.clean 1 80
for l in ${src} ${tgt}; do
    perl ${LC} < ${tmp}/train.tags.${lang}.clean.${l} > ${tmp}/train.tags.${lang}.${l}
done

echo "pre-processing valid/test data..."
for l in ${src} ${tgt}; do
    for o in `ls ${orig}/${lang}/IWSLT14.TED*.${l}.xml`; do
    fname=${o##*/}
    f=${tmp}/${fname%.*}
    echo $o $f
    grep '<seg id' $o | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl ${TOKENIZER} -threads 8 -l ${l} | \
    perl ${LC} > ${f}
    echo ""
    done
done

echo "creating train, valid, test..."
for l in ${src} ${tgt}; do
    awk '{if (NR%23 == 0)  print $0; }' ${tmp}/train.tags.de-en.${l} > ${tmp}/valid.${l}
    awk '{if (NR%23 != 0)  print $0; }' ${tmp}/train.tags.de-en.${l} > ${tmp}/train.${l}

    cat ${tmp}/IWSLT14.TED.dev2010.de-en.${l} \
        ${tmp}/IWSLT14.TEDX.dev2012.de-en.${l} \
        ${tmp}/IWSLT14.TED.tst2010.de-en.${l} \
        ${tmp}/IWSLT14.TED.tst2011.de-en.${l} \
        ${tmp}/IWSLT14.TED.tst2012.de-en.${l} \
        > ${tmp}/test.${l}
done

echo "learning * joint * SentencePiece..."
cat "${tmp}/train.${src}" "${tmp}/train.${tgt}" | shuf > ${tmp}/train.tmp
spm_train --input="${tmp}/train.tmp" --model_prefix=spm.${vocab_size} --vocab_size=${vocab_size} \
          --character_coverage=1.0 --hard_vocab_limit=false --model_type=unigram \
          --unk_piece='<unk>' --pad_piece='<pad>' --user_defined_symbols='&apos;,&quot;,&#91;,&#93;,&amp;'
rm "${tmp}/train.tmp"

echo "applying SentencePiece..."
for l in ${src} ${tgt}; do
    for p in train valid test; do
        spm_encode --model=spm.${vocab_size}.model --output_format=piece < "${tmp}/${p}.${l}" > "${prep}/${p}.sp.${vocab_size}.${l}"
    done
done

for l in ${src} ${tgt}; do
    for p in train valid test; do
        mv ${tmp}/${p}.${l} ${prep}/
    done
done

mv "spm.${vocab_size}.model" "${prep}/"
mv "spm.${vocab_size}.vocab" "${prep}/"
rm -rf ${MOSES}
rm -rf ${orig}
rm -rf ${tmp}

echo "done."