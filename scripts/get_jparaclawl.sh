#!/usr/bin/env bash

joey_dir="${HOME}/joeynmt"
data_dir="${joey_dir}/test/data"
#data_dir="../test/data"
src="en"
trg="ja"


# Prepare Jparacrawl
echo "Prepare Jparacrawl (Train-Dev Data)"

jparacrawl_dir="${data_dir}/jparacrawl"

if [ ! -d "${jparacrawl_dir}" ]; then
  mkdir ${jparacrawl_dir}
fi
cd ${jparacrawl_dir}

jparacrawl_url="http://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/2.0/bitext/${src}-${trg}.tar.gz"

if [ ! -f "${src}-${trg}.tar.gz" ]; then
  echo -e "\tDownloading training data from ${jparacrawl_url}..."
  wget ${jparacrawl_url}
fi

if [ ! -d "${src}-${trg}" ]; then
  tar xzvf ./en-ja.tar.gz
  #rm ./en-ja.tar.gz
fi

if [ ! -f "train.en" ]; then
  echo -e "\tPreprocessing training data ..."
  python ${joey_dir}/scripts/preprocess_jparacrawl.py --data_dir="${jparacrawl_dir}" --dev_size=5000 --seed=12345
  wc -l train.* dev.*
fi

# train SentencePiece
model_type="unigram"
vocab_size=32000
character_coverage=1.0
if [ -f "train.en" ] && [ ! -f "train.sp.${vocab_size}.en" ]; then
  echo -e "\tLearning SentencePiece..."
  for l in ${src} ${trg}; do
    if [ ${l} == "ja" ]; then
      character_coverage=0.995
    fi
    spm_train --input="train.${l}" --model_prefix=spm.${l}.${vocab_size} --vocab_size=${vocab_size} \
            --character_coverage=${character_coverage} --hard_vocab_limit=false --model_type=${model_type} \
            --unk_piece='<unk>' --pad_piece='<pad>' --input_sentence_size=1000000 --shuffle_input_sentence=true

    # vocab file
    cut -f1 -d$'\t' spm.${l}.${vocab_size}.vocab > vocab.${l}

    # apply SentencePiece
    echo -e "\tApplying SentencePiece..."

    for p in train dev; do
      spm_encode --model=spm.${l}.${vocab_size}.model --output_format=piece < ${p}.${l} > ${p}.sp.${vocab_size}.${l}
    done
  done
fi

cd ${data_dir}
# Prepare IWSLT17
echo "Prepare IWSLT17 (Test Data)"

iwslt17_dir="${data_dir}/iwslt17"

if [ ! -d "${iwslt17_dir}" ]; then
  mkdir ${iwslt17_dir}
fi
cd ${iwslt17_dir}

iwslt17_url1="https://wit3.fbk.eu/archive/2017-01-ted-test/texts/${trg}/${src}/${trg}-${src}.tgz"
iwslt17_url2="https://wit3.fbk.eu/archive/2017-01-ted-test/texts/${src}/${trg}/${src}-${trg}.tgz"

## ja-en
if [ ! -f "${trg}-${src}.tgz" ]; then
  echo -e "\tDownloading test data from ${iwslt17_url1}..."
  wget ${iwslt17_url1}
fi

if [ ! -d "${trg}-${src}" ]; then
  tar xzvf ./${trg}-${src}.tgz
  #rm ./${trg}-${src}.tar.gz
fi

## en-ja
if [ ! -f "${src}-${trg}.tgz" ]; then
  echo -e "\tDownloading test data from ${iwslt17_url2}..."
  wget ${iwslt17_url2}
fi

if [ ! -d "${src}-${trg}" ]; then
  tar xzvf ./${src}-${trg}.tgz
  #rm ./${src}-${trg}.tar.gz
fi

if [ ! -f "test.en" ] && [ ! -f "test.sp.${vocab_size}.en" ]; then
  echo -e "\tPreprocessing test data..."
  for l in ${src} ${trg}; do
    lang="${src}-${trg}"
    if [ ${l} == "ja" ]; then
      lang="${trg}-${src}"
    fi
    for o in `ls ${iwslt17_dir}/${lang}/IWSLT17.TED*.${lang}.${l}.xml`; do
        fname=${o##*/}
        f=${iwslt17_dir}/${lang}/${fname%.*}
        echo $o $f
        grep '<seg id' $o | \
            sed -e 's/<seg id="[0-9]*">\s*//g' | \
            sed -e 's/\s*<\/seg>\s*//g' | \
            sed -e "s/\’/\'/g" > ${f}
        echo ""
    done
    cat ${lang}/IWSLT17.TED*.${lang}.${l} > test.${l}

    # apply SentencePiece
    echo -e "\tApplying SentencePiece..."
    spm_encode --model=${jparacrawl_dir}/spm.${l}.${vocab_size}.model --output_format=piece < test.${l} > test.sp.${vocab_size}.${l}
  done
fi


cd ${data_dir}
# Prepare KFTT
echo "Prepare KFTT (Fine Tuning)"

kftt_dir="${data_dir}/kftt"

if [ ! -d "${kftt_dir}" ]; then
  mkdir ${kftt_dir}
fi
cd ${kftt_dir}

kftt_url=http://www.phontron.com/kftt/download/kftt-data-1.0.tar.gz
if [ ! -f "kftt-data-1.0.tar.gz" ]; then
  echo -e "\tDownloading kftt data from ${kftt_url}..."
  wget ${kftt_url}
fi

if [ ! -d "kftt-data-1.0" ]; then
  tar zxvf kftt-data-1.0.tar.gz
  #rm kftt-data-1.0.tar.gz
fi

# apply SentencePiece
if [ ! -f "test.sp.${vocab_size}.en" ]; then
  echo -e "\tApplying SentencePiece..."
  for l in ${src} ${trg}; do
    for p in train dev test tune; do
      spm_encode --model=${jparacrawl_dir}/spm.${l}.${vocab_size}.model --output_format=piece < "kftt-data-1.0/data/orig/kyoto-${p}.${l}" > ${p}.sp.${vocab_size}.${l}
    done
  done
fi


echo "done."


