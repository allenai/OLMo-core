PATH=/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/ml/data/1124_provenance.csv
URL_HEADER=https://olmo-data.org
TOKENIZER=allenai/dolma2-tokenizer

arxiv,https://olmo-data.org/preprocessed/proof-pile-2/v0_decontaminated/arxiv/train/allenai/dolma2-tokenizer/part-15-00000.csv.gz
dclm,https://olmo-data.org/preprocessed/dclm/text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/allenai/dolma2-tokenizer/part-054-00002.csv.gz
open-web-math,https://olmo-data.org/preprocessed/proof-pile-2/v0_decontaminated/open-web-math/train/allenai/dolma2-tokenizer/part-05-00000.csv.gz
pes2o,https://olmo-data.org/preprocessed/pes2o/allenai/dolma2-tokenizer/part-03-00000.csv.gz
starcoder,https://olmo-data.org/preprocessed/starcoder/v1-decon-100_to_20k-2star-top_token_030/allenai/dolma2-tokenizer/part-048-00000.csv.gz
wiki,https://olmo-data.org/preprocessed/olmo-mix/danyh-compiled-v1_7/documents/wiki/allenai/dolma2-tokenizer/part-1-00000.csv.gz

PREPROCESSED_DATA_DIR=/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/ml/data/OLMoE-mix-1124
# URL_PREFIX=${URL_HEADER}/preprocessed/proof-pile-2/v0_decontaminated/algebraic-stack/train/allenai/dolma2-tokenizer


DATASET=algebraic-stack
mkdir -p $PREPROCESSED_DATA_DIR/${DATASET}
for i in $( seq -w 00 15 ); do
    filename=preprocessed/proof-pile-2/v0_decontaminated/algebraic-stack/train/${TOKENIZER}/part-${s}-00000.csv.gz
    url=${URL_HEADER}/${filename}
    if wget -q --method=HEAD ${url} ;
    then
        wget -O $PREPROCESSED_DATA_DIR/${filename} -c ${url} ;
    fi
done


DATASET=dclm
mkdir -p $PREPROCESSED_DATA_DIR/${DATASET}
for i in $( seq -w 00 15 ); do
    for j in $ ( seq -w 00 15 ); do
        filename=preprocessed/dclm/text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/${TOKENIZER}/part-${i}-${j}.csv.gz
        url=${URL_HEADER}/${filename}
        if wget -q --method=HEAD ${url} ;
        then
            wget -O $PREPROCESSED_DATA_DIR/${filename} -c ${url} ;
        fi
    done
done

DATASET=open-web-math
mkdir -p $PREPROCESSED_DATA_DIR/${DATASET}
for i in $( seq -w 00 15 ); do
    filename=preprocessed/proof-pile-2/v0_decontaminated/open-web-math/train/${TOKENIZER}/part-${i}-00000.csv.gz
    url=${URL_HEADER}/${filename}
    if wget -q --method=HEAD ${url} ;
    then
        wget -O $PREPROCESSED_DATA_DIR/${filename} -c ${url} ;
    fi
done

DATASET=pes2o
mkdir -p $PREPROCESSED_DATA_DIR/${DATASET}
for i in $( seq -w 00 15 ); do
    filename=preprocessed/pes2o/${TOKENIZER}/part-${i}-00000.csv.gz
    url=${URL_HEADER}/${filename}
    if wget -q --method=HEAD ${url} ;
    then
        wget -O $PREPROCESSED_DATA_DIR/${filename} -c ${url} ;
    fi
done

DATASET=starcoder
mkdir -p $PREPROCESSED_DATA_DIR/${DATASET}
for i in $( seq -w 00 15 ); do
    filename=preprocessed/starcoder/v1-decon-100_to_20k-2star-top_token_030/${TOKENIZER}/part-${i}-00000.csv.gz
    url=${URL_HEADER}/${filename}
    if wget -q --method=HEAD ${url} ;
    then
        wget -O $PREPROCESSED_DATA_DIR/${filename} -c ${url} ;
    fi
done

DATASET=wiki
mkdir -p $PREPROCESSED_DATA_DIR/${DATASET}
for i in $( seq -w 00 15 ); do
    filename=preprocessed/olmo-mix/danyh-compiled-v1_7/documents/wiki/${TOKENIZER}/part-${i}-00000.csv.gz
    url=${URL_HEADER}/${filename}
    if wget -q --method=HEAD ${url} ;
    then
        wget -O $PREPROCESSED_DATA_DIR/${filename} -c ${url} ;
    fi
done
