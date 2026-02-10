# export HF_URL=https://huggingface.co/datasets/allenai/OLMoE-mix-0924/resolve/main/data
# 
# # algebraic-stack
# DATASET_DIR=$RAW_DATA_DIR/algebraic-stack
# mkdir -p $DATASET_DIR; cd $DATASET_DIR;
# for i in $( seq -w 0000 9999 ); do
#     url=${HF_URL}/algebraic-stack/algebraic-stack-train-${i}.json.gz
#     if wget -q --method=HEAD ${url} ; 
#     then 
#         wget -c ${url}  ; 
#     else 
#         break 
#     fi ;
# done

# # dclm
# DATASET_DIR=$RAW_DATA_DIR/dclm
# mkdir -p $DATASET_DIR; cd $DATASET_DIR;
# for i in $( seq -w 0000 9999 ); do
#     url=${HF_URL}/dclm/dclm-${i}.json.zst
#     if wget -q --method=HEAD ${url} ; 
#     then 
#         wget -c ${url}  ; 
#     else 
#         break 
#     fi ;
# done

# # open-web-math
# DATASET_DIR=$RAW_DATA_DIR/open-web-math
# mkdir -p $DATASET_DIR; cd $DATASET_DIR;
# for i in $( seq -w 0000 9999 ); do
#     url=${HF_URL}/open-web-math/open-web-math-train-${i}.json.gz
#     if wget -q --method=HEAD ${url} ; 
#     then 
#         echo "Downloading ${url}"
#         wget -O $DATASET_DIR/open-web-math-train-${i}.json.gz -c ${url}; 
#     else 
#         break 
#     fi ;
# done
# for i in $( seq -w 041 053 ); do
#     url=${HF_URL}/open-web-math/${i}.jsonl.gz
#     # page_url=${HF_PAGE_URL}/open-web-math/open-web-math-train-${i}.json.gz
#     if wget -q --method=HEAD ${url} ; 
#     then 
#         wget -O $DATASET_DIR/${i}.jsonl.gz -c ${url} ; 
#     else 
#         break 
#     fi ;
# done

# # pes2o
# DATASET_DIR=$RAW_DATA_DIR/pes2o
# mkdir -p $DATASET_DIR; cd $DATASET_DIR;
# for i in $( seq -w 0000 9999 ); do
#     url=${HF_URL}/pes2o/pes2o-${i}.json.gz
#     if wget -q --method=HEAD ${url} ; 
#     then 
#         wget -O $DATASET_DIR/pes2o-${i}.json.gz -c ${url} ; 
#     else 
#         break 
#     fi ;
# done

# # starcoder
# declare -a sc_langs=(ada agda alloy antlr applescript assembly 
# augeas awk batchfile bluescript c c-sharp cpp css cuda dart 
# dockerfile elixir elm emacs erland f-sharp fortran git-commits-cleaned 
# github-issues-filtered-structured glsl go groovy haskell html idris 
# isabelle java java-server-pages javascript json julia jupyter kotlin 
# lean literate-agda literate-coffeescript literate-haskell lua maple 
# makefile markdown mathematica matlab ocaml pascal perl php powershell 
# prolog protocol-buffer python r racket restructuredtext rmarkdown ruby 
# rust sas scala scheme shell smalltalk solidity sparql sql stan 
# standard-ml stat systemverilog tcl tcsh tex thrift typescript verilog 
# vhdl visual-basic xslt yacc yaml zig)


# DATASET_DIR=$RAW_DATA_DIR/starcoder
# mkdir -p $DATASET_DIR; cd $DATASET_DIR;
# for l in "${sc_langs[@]}"; do 
#     for i in $( seq -w 0000 9999 ); do
#         url=${HF_URL}/starcoder/${l}-${i}.json.gz
#         if wget -q --method=HEAD ${url} ; 
#         then 
#             wget -O $DATASET_DIR/${l}-${i}.json.gz -c ${url} ; 
#         else 
#             break 
#         fi ;
#     done
# done

# # wiki
# DATASET_DIR=$RAW_DATA_DIR/wiki
# mkdir -p $DATASET_DIR; cd $DATASET_DIR;
# for i in $( seq -w 0000 9999 ); do
#     url=${HF_URL}/wiki/wiki-${i}.json.gz
#     if wget -q --method=HEAD ${url} ; 
#     then 
#         wget -O $DATASET_DIR/wiki-${i}.json.gz -c ${url} ; 
#     else 
#         break 
#     fi ;
# done

PREPROCESSED_VAL_DATA_DIR=/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/ml/data/preprocessed/eval-data/perplexity/v3_small_dolma2-tokenizer

VALID_URL=https://olmo-data.org/eval-data/perplexity/v3_small_dolma2-tokenizer
declare -a eval_subsets=(c4_en dolma_books dolma_common-crawl dolma_pes2o dolma_reddit dolma_stack dolma_wiki ice m2d2_s2orc pile wikitext_103)
for s in "${eval_subsets[@]}"; do
    url=${VALID_URL}/${s}/val/part-0-00000.npy
    if wget -q --method=HEAD ${url} ;
    then
        mkdir -p $PREPROCESSED_VAL_DATA_DIR/${s}/val ;
        wget -O $PREPROCESSED_VAL_DATA_DIR/${s}/val/part-0-00000.npy -c ${url} ;
    fi
done