# ERASE
Code and Data for "Language Modeling with Editable External Knowledge"

### This repository is a work in progress.

## Setup
To setup your environment, run:
```bash
conda create -n mem_rewrite PYTHON=3.11
conda activate mem_rewrite

# get pytorch with a version of CUDA compatible with your machine
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

# install other requirements
bash setup.sh
```


## Run ERASE
To run ERASE on the CLARK-News dataset, use:
```bash
python lm_eval.py \
--dataset news \
--datapath CLARK_news/ \
--model_name [mistralai/Mixtral-8x7B-Instruct-v0.1|Meta-Llama-3-8B-Instruct] \
--context_length [2048|4096] \
--save_as_facts \
--retrieve_facts similarity \
(--overwrite_facts similarity --edit_hops 1)
```

* `--context_length` sets the context window of the model
* `--save_as_facts` toggles saving the entries to the KB as facts (rather than as passages)
* `--retrieve_facts` sets how we want to retrieve KB entries. Set it to `similarity` for dense retrieval. To turn off retrieval, do not include this flag. 
* `--overwrite_facts` toggles updating existing KB entries according to new documents. Set it to `similarity` to use dense retrieval to retrieve facts to update. To turn off updating behavior, do not include this flag. 
* `--edit_hops` sets how many "hops" of retrieval we want to performing when updating existing entries. For each edit_hops > 1, the retriever will perform another round of retrieval based on similarity to the facts retrieved from the last round. This is set to 1 by default.




## Data

### CLARK-News
The CLARK-News dataset is available under `CLARK_news`.

To run our data collection process, follow:

1. Get Wikidata triples that change over time
```bash
python script/get_wikidata_triples.py
```

2. Annotate source documents, following:

Get google searches:
```bash
python script/extract_queries.py
```

Launch annotations (launch annotation interface):
```bash
python AnnotationInterface/webserver.py
```

Cleanup and check annotations
```bash
# DO NOT RUN WHILE PEOPLE ARE ANNOTATING
python AnnotationInterface/cleanup_unsubmitted.py   # un-allocate for users who have been allocated triples but haven't submitted
```

Pull sources from links:
```bash
python script/pull_external_sources.py \
--edits_file AnnotationInterface/results/property_to_results_larger_subset_links_filtered.csv \
--output_dir wikipedia-data/subset
```

Automated validation of annotations:
```bash
# check problems
python script/check_annotations.py  # display annotations in annotations.html
```

Second round of annotation:
```bash
python CheckInterface/webserver.py
```

Cleanup and check second round
```bash
python CheckInterface/cleanup_unsubmitted.py
```

3. Make questions from wikidata relations

```bash
python script/generate_wikidata_questions.py --wikidata_csv CheckInterface/results/property_to_results_larger_subset_links_filtered.csv --output_dir wikidata-data/subset
python script/generate_wikidata_questions.py --wikidata_csv CheckInterface/results/property_to_results_larger_sports_subset_links_filtered.csv --output_dir wikidata-data/sports/
python script/generate_wikidata_questions.py --wikidata_csv CheckInterface/results/property_to_results_larger_position_subset_links_filtered.csv --output_dir wikidata-data/position/
python script/generate_wikidata_questions.py --wikidata_csv CheckInterface/results/property_to_results_smaller_sports_subset_links_filtered.csv --output_dir wikidata-data/sports_subset/
```


### CLARK-Conversations
Coming soon.
