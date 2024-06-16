# get FAISS
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

# get GENRE
git clone https://github.com/facebookresearch/GENRE.git
cd GENRE/
bash scripts_genre/download_all_models.sh
git clone --branch fixing_prefix_allowed_tokens_fn https://github.com/nicola-decao/fairseq
cd fairseq
pip install --editable ./

# install local requirements
cd ..
pip install -r requirements.txt