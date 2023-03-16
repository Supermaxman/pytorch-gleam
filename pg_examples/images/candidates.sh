#!/usr/bin/env bash
# python remove_dups.py

# clip-retrieval index \
#     --embeddings_folder /shared/aifiles/disk1/media/twitter/v10/covid19-twitter-images-dedup-emb-vit-b-32 \
#     --index_folder /shared/aifiles/disk1/media/twitter/v10/covid19-twitter-images-dedup-emb-vit-b-32-index \
#     --max_index_memory_usage "8G" \
#     --current_memory_available "10G"

# clip-retrieval index \
#     --embeddings_folder /shared/aifiles/disk1/media/twitter/v10/covid19-twitter-text-emb-vit-b-32 \
#     --index_folder /shared/aifiles/disk1/media/twitter/v10/covid19-twitter-text-emb-vit-b-32-index \
#     --max_index_memory_usage "8G" \
#     --current_memory_available "10G"



frame_path=/users/max/data/corpora/co-vax-frames/covid19/co-vax-frames.json

python pytorch_gleam/scripts/search/search_faiss_clip.py \
    --top_k 100 \
    --query_path $frame_path \
    --index_path /shared/aifiles/disk1/media/twitter/v10/covid19-twitter-images-dedup-emb-vit-b-32-index/image.index \
    --output_path /shared/aifiles/disk1/media/twitter/v10/full-image-top-100-vit-b-32.json

python pytorch_gleam/scripts/search/search_faiss_clip.py \
    --top_k 100 \
    --query_path $frame_path \
    --index_path /shared/aifiles/disk1/media/twitter/v10/covid19-twitter-joint-emb-vit-b-32-index/text.index \
    --output_path /shared/aifiles/disk1/media/twitter/v10/full-text-top-100-vit-b-32.json
