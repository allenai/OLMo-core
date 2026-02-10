URL_BASE=https://olmo-data.org
TOKENIZER=allenai/dolma2-tokenizer

PREPROCESSED_DATA_DIR=/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/ml/data/OLMoE-mix-0824
# URL_PREFIX=${URL_HEADER}/preprocessed/proof-pile-2/v0_decontaminated/algebraic-stack/train/allenai/dolma2-tokenizer

DATASET=algebraic-stack
FOLDER=preprocessed/proof-pile-2/v0_decontaminated/algebraic-stack/train/${TOKENIZER}
URL_HEADER=${URL_BASE}/${FOLDER}
mkdir -p $PREPROCESSED_DATA_DIR/${FOLDER}
for i in $( seq -w 00 16 ); do
    filename=part-${s}-00000.npy
    url=${URL_HEADER}/${filename}
    if wget -q --method=HEAD ${url} ;
    then
        wget -O $PREPROCESSED_DATA_DIR/${filename} -c ${url} ;
    fi
done

DATASET=arxiv
FOLDER=preprocessed/proof-pile-2/v0_decontaminated/arxiv/train/${TOKENIZER}
URL_HEADER=${URL_BASE}/${FOLDER}
mkdir -p $PREPROCESSED_DATA_DIR/${FOLDER}
for i in $( seq -w 00 20 ); do
    filename=part-${s}-00000.npy
    url=${URL_HEADER}/${filename}
    if wget -q --method=HEAD ${url} ;
    then
        wget -O $PREPROCESSED_DATA_DIR/${filename} -c ${url} ;
    fi
done

DATASET=open-web-math
FOLDER=preprocessed/proof-pile-2/v0_decontaminated/open-web-math/train/${TOKENIZER}
URL_HEADER=${URL_BASE}/${FOLDER}
mkdir -p $PREPROCESSED_DATA_DIR/${FOLDER}
for i in $( seq -w 00 13 ); do
    filename=part-${s}-00000.npy
    url=${URL_HEADER}/${filename}
    if wget -q --method=HEAD ${url} ;
    then
        wget -O $PREPROCESSED_DATA_DIR/${filename} -c ${url} ;
    fi
done

DATASET=pes2o
FOLDER=preprocessed/pes2o/${TOKENIZER}
URL_HEADER=${URL_BASE}/${FOLDER}
mkdir -p $PREPROCESSED_DATA_DIR/${FOLDER}
for i in $( seq -w 00 26 ); do
    filename=part-${s}-00000.npy
    url=${URL_HEADER}/${filename}
    if wget -q --method=HEAD ${url} ;
    then
        wget -O $PREPROCESSED_DATA_DIR/${filename} -c ${url} ;
    fi
done

DATASET=starcoder
FOLDER=preprocessed/starcoder/v1-decon-100_to_20k-2star-top_token_030/${TOKENIZER}
URL_HEADER=${URL_BASE}/${FOLDER}
mkdir -p $PREPROCESSED_DATA_DIR/${FOLDER}
for i in $( seq -w 000 100 ); do
    filename=part-${s}-00000.npy
    url=${URL_HEADER}/${filename}
    if wget -q --method=HEAD ${url} ;
    then
        wget -O $PREPROCESSED_DATA_DIR/${filename} -c ${url} ;
    fi
done

DATASET=dclm
FOLDER=preprocessed/dclm/text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/${TOKENIZER}
URL_HEADER=${URL_BASE}/${FOLDER}
mkdir -p $PREPROCESSED_DATA_DIR/${FOLDER}
for i in $( seq -w 000 188 ); do
    for j in $( seq -w 00000 00005 ); do
        filename=part-${i}-${j}.npy
        url=${URL_HEADER}/${filename}
        if wget -q --method=HEAD ${url} ;
        then
            wget -O $PREPROCESSED_DATA_DIR/${filename} -c ${url} ;
        fi
    done
done

DATASET=wikipedia
FOLDER=preprocessed/olmo-mix/danyh-compiled-v1_7/documents/wiki/${TOKENIZER}
URL_HEADER=${URL_BASE}/${FOLDER}
mkdir -p $PREPROCESSED_DATA_DIR/${FOLDER}
for i in $( seq -w 0 2 ); do
    filename=part-${i}-00000.npy
    url=${URL_HEADER}/${filename}
    if wget -q --method=HEAD ${url} ;
    then
        wget -O $PREPROCESSED_DATA_DIR/${filename} -c ${url} ;
    fi
done

