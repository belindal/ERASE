import os
from dataloaders import DATASETS
import json
from tqdm import tqdm
from pandas import DataFrame
import argparse
import torch
from models import MODELS
from models.lang_model import api_pricing
from models.kb_model import TOKENIZER
import time
from transformers import AutoTokenizer


def main(args):
    assert args.model_name is not None or args.local_model_path is not None, "Must specify a model name or local model path"
    if args.model_name is None:
        model_name = args.local_model_path
    else:
        model_name = args.model_name
    os.makedirs(f"results/{args.dataset}_{model_name.replace('/', '_')}", exist_ok=True)
    save_fn = f"results/{args.dataset}_{model_name.replace('/', '_')}/results_{'facts_' if args.save_as_facts else ''}{'overwrite_facts_'+args.overwrite_facts if args.overwrite_facts else ''}{'retrieve_'+args.retrieve_facts if args.retrieve_facts else ''}{'_'+str(args.edit_hops)+'_edithops' if args.overwrite_facts else ''}.csv"
    print(f"Saving results to {save_fn}")

    if args.logging:
        open(f"{save_fn[:-4]}_gpt_prob_outputs.txt", "w").close()
        open(f"{save_fn[:-4]}_fact_extraction.txt", "w").close()
        open(f"{save_fn[:-4]}_question_outputs.txt", "w").close()

    print("Loading tokenizer...")
    if args.local_model_path is not None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.local_model_path)
        except:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
    else:
        tokenizer = TOKENIZER.get(model_name, AutoTokenizer.from_pretrained("gpt2"))

    # load data
    print("Loading data...")
    data_kwargs = {}
    if args.tasks is None:
        if args.dataset == "news":
            data_kwargs["tasks"] = "full"
        elif args.dataset == "convo":
            data_kwargs["tasks"] = "singlehop_convos,multihop_convos"
    else:
        data_kwargs["tasks"] = args.tasks
    if args.datapath is not None:
        data_kwargs["data_dir"] = args.datapath
    data_kwargs["context_length"] = args.context_length
    assert args.context_length > 200, "Context length must be greater than 200"
    dataset = DATASETS[args.dataset](**data_kwargs, tokenizer=tokenizer)

    # load model
    print("Loading model...")
    start_time = time.time()
    if not args.no_cache:
        inference_kwargs = {
            "temperature": 0.0
        }
    else:
        inference_kwargs = {}
    os.makedirs(os.path.dirname(f"{args.lm_cache_dir}/{model_name.replace('/', '_')}"), exist_ok=True)
    model = MODELS[args.dataset](
        model_name=model_name,
        context_length=args.context_length,
        cache_dir=args.lm_cache_dir,
        use_cache=(not args.no_cache),
        inference_kwargs=inference_kwargs,
        dataset_all_possible_objects=getattr(dataset, "all_possible_objects", None),
        device="cuda" if args.use_cuda and torch.cuda.is_available() else "cpu",
        save_as_facts=args.save_as_facts,
        overwrite_facts=args.overwrite_facts,
        retrieve_facts=args.retrieve_facts,
        specify_parameters=args.specify_parameters,
        save_fn=save_fn[:-4],
        hops=args.edit_hops,
        tokenizer=tokenizer,
        logging=args.logging,
        local_model_path=args.local_model_path,
    )
    end_time = time.time()
    print(f"Loaded after {end_time - start_time:.1f}s")

    # set metric
    if args.dataset == "news":
        metric_to_display = "accuracy_per_ts"
    elif args.dataset == "convo":
        metric_to_display = "score_per_num_updates_by_task"

    # df for saving results
    results = DataFrame(columns=[
        *dataset[0].keys(), "pred_answer", "correctness", "prompt",
    ])
    final_kbs = {}

    pbar = tqdm(enumerate(dataset), total=len(dataset))
    for idx, item in pbar:
        model_response, metadata, final_kb = model.infer(entry=item)
        final_kbs[item['questions'][0]['data_id']] = final_kb
        metrics = model.get_metrics(entry=item, prediction=model_response)

        if type(model_response) == list:
            for i, response in enumerate(model_response):
                # Update DataFrame columns dynamically for new keys in metadata
                question = item['questions'][i]
                question['timestamp'] = question['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
                new_keys = [key for key in metadata[i] if key not in results.columns]
                new_keys.extend([key for key in question if key not in results.columns])
                for new_key in new_keys:
                    results[new_key] = None
                # append to results
                results.loc[len(results)] = {
                    **question,
                    "correctness": response,
                    **metadata[i],
                }
        else:
            # Update DataFrame columns dynamically for new keys in metadata
            new_keys = [key for key in metadata if key not in results.columns]
            for new_key in new_keys:
                results[new_key] = None
            # append to results
            results.loc[len(results)] = {
                **item,
                "pred_answer": model_response,
                "correctness": metrics.get("curr_entry_correct"),
                **metadata,
            }

        # update progress bar
        # TODO cleanup
        if type(metrics[metric_to_display]) == dict:
            accuracies = {}
            overall_accuracy = [0, 0]
            for key in metrics[metric_to_display]:
                if type(metrics[metric_to_display][key]) == list and args.dataset == "news":
                    accuracies[key] = round(metrics[metric_to_display][key][0] / metrics[metric_to_display][key][1], 2)
                    overall_accuracy[0] += metrics[metric_to_display][key][0]
                    overall_accuracy[1] += metrics[metric_to_display][key][1]
                elif type(metrics[metric_to_display][key]) == list:
                    accuracies[key] = round(sum(metrics[metric_to_display][key]) / len(metrics[metric_to_display][key]), 2)
                    overall_accuracy[0] += sum(metrics[metric_to_display][key])
                    overall_accuracy[1] += len(metrics[metric_to_display][key])
                elif type(metrics[metric_to_display][key]) == float:
                    accuracies[key] = round(metrics[metric_to_display][key], 2)
                else:
                    assert type(metrics[metric_to_display][key]) == dict
                    accuracies[key] = {}
                    for subkey in metrics[metric_to_display][key]:
                        if type(metrics[metric_to_display][key][subkey]) == list and args.dataset == "news":
                            accuracies[key][subkey] = round(metrics[metric_to_display][key][subkey][0] / metrics[metric_to_display][key][subkey][1], 2)
                        elif type(metrics[metric_to_display][key][subkey]) == list:
                            accuracies[key][subkey] = round(sum(metrics[metric_to_display][key][subkey]) / len(metrics[metric_to_display][key][subkey]), 2)
                        elif type(metrics[metric_to_display][key][subkey]) == float:
                            accuracies[key][subkey] = round(metrics[metric_to_display][key][subkey], 2)
                        else:
                            assert type(metrics[metric_to_display][key][subkey]) == dict
                            accuracies[key][subkey] = {}
                            for subsubkey in metrics[metric_to_display][key][subkey]:
                                accuracies[key][subkey][subsubkey] = round(sum(metrics[metric_to_display][key][subkey][subsubkey]) / len(metrics[metric_to_display][key][subkey][subsubkey]), 2)
            if type(metrics[metric_to_display][key]) == list:
                overall_accuracy = overall_accuracy[0] / overall_accuracy[1]
                accuracies["overall"] = round(overall_accuracy, 2)
            elif type(metrics[metric_to_display][key]) == float:
                overall_accuracy = sum(accuracies[key] for key in accuracies) / len(accuracies)
                accuracies["overall"] = round(overall_accuracy, 2)
        else:
            accuracies = round(metrics[metric_to_display], 2)
        desc = f"{metric_to_display}: {accuracies}"
        desc += f", Cost: ${model.get_total_cost():.2f}"
        pbar.set_description(desc)


    # print results
    metrics = model.get_metrics()
    print(model.metrics)
    print(metrics)

    # save results
    results.to_csv(save_fn, index=False)
    json.dump(final_kbs, open(f"results/{args.dataset}_{model_name.replace('/', '_')}/kb_{'facts_' if args.save_as_facts else ''}{'overwrite_facts_' if args.overwrite_facts else ''}{'retrieve_'+args.retrieve_facts if args.retrieve_facts else ''}{'_'+str(args.edit_hops)+'_edithops' if args.overwrite_facts else ''}.json", "w"), indent=4)


def add_arguments(parser):
    parser.add_argument("--datapath", type=str, default=None)  # The path to the dataset (None for default).
    parser.add_argument("--dataset", type=str, default="news")  # The data type (e.g. news, convo).
    parser.add_argument("--model_name", type=str, default=None, choices=list(api_pricing.keys()) + [None])  # The specific model name of the model to use.
    parser.add_argument("--tasks", type=str, default=None)  # The tasks under the dataset to use (e.g. singlehop_convos/multihop_convos for convo).
    parser.add_argument("--no_cache", action="store_true", default=False)  # Don't use the cache (will repeatedly query LM).
    parser.add_argument("--use_cuda", action="store_true", default=True)  # Use CUDA if available.
    parser.add_argument("--save_as_facts", action="store_true", default=False)  # Save the read articles as facts in kb (instead of original text).
    parser.add_argument("--overwrite_facts", type=str, default=None, choices=[None, "similarity", "entity_similarity"])  # Overwrite (certain) old facts rather than appending.
    parser.add_argument("--retrieve_facts", type=str, default=None, choices=[None, "similarity", "entity_similarity"]) # Selectively retrieve relevant facts from the KB rather than using the full context.
    parser.add_argument("--specify_parameters", action="store_true", default=False)  # Specify the parameters of the data generation model in the prompt.
    parser.add_argument("--context_length", type=int, default=1000)  # Specify the parameters of the data generation model in the prompt.
    parser.add_argument("--edit_hops", type=int, default=1)  # How many hops of facts to retrieve for editing.
    parser.add_argument("--logging", action="store_true", default=False)  # Log the results.
    parser.add_argument("--lm_cache_dir", type=str, default="cache")  # The directory to store cached LM responses.
    parser.add_argument("--local_model_path", type=str, default=None)  # The path to use a local model.



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
