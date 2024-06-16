import time
from typing import List, Union
from openai import OpenAI
import requests
import os
import textwrap
import json

from models.kb_model import KBModel, probability_mapping, TOKENIZER
from abc import ABC
import regex as re
import numpy as np
from tqdm import tqdm
import torch
import transformers
from transformers import AutoTokenizer


api_pricing = {
    "gpt-4-turbo-preview": {
        "prompt_tokens": 0.01 / 1000,
        "completion_tokens": 0.03 / 1000,
    },
    "gpt-4-1106-preview": {
        "prompt_tokens": 0.01 / 1000,
        "completion_tokens": 0.03 / 1000,
    },
    "gpt-4": {
        "prompt_tokens": 0.03 / 1000,
        "completion_tokens": 0.06 / 1000,
    },
    "gpt-4o": {
        "prompt_tokens": 5 / 1000000,
        "completion_tokens": 15 / 1000000,
    },
    "gpt-3.5-turbo": {
        "prompt_tokens": 0.0015 / 1000,
        "completion_tokens": 0.002 / 1000,
    },
    "mistralai/Mixtral-8x7B-Instruct-v0.1": {
        "prompt_tokens": 0.6 / 1000000,
        "completion_tokens": 0.6 / 1000000,
    },
    "mistralai/Mixtral-8x22B-Instruct-v0.1": {
        "prompt_tokens": 1.2 / 1000000,
        "completion_tokens": 1.2 / 1000000,
    },
    "Meta-Llama-3-8B-Instruct": {
        "prompt_tokens": 0.0,
        "completion_tokens": 0.0,
    },
    "Meta-Llama-3-70B-Instruct": {
        "prompt_tokens": 0.0,
        "completion_tokens": 0.0,
    },
    "meta-llama/Llama-3-70b-chat-hf": {
        "prompt_tokens": 0.9 / 1000000,
        "completion_tokens": 0.9 / 1000000,
    },
    "meta-llama/Llama-3-8b-chat-hf": {
        "prompt_tokens": 0.2 / 1000000,
        "completion_tokens": 0.2 / 1000000,
    }
}

class LangModel(ABC):
    """
    Model for querying the OpenAI API.
    """
    def __init__(self, model_name, cache_dir="openai_cache", use_cache=True, inference_kwargs={}, context_length=4000, save_fn="results/", tokenizer=None, logging=False, **kwargs):
        self.model_name = model_name
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = TOKENIZER[self.model_name]
        if "meta-llama" in self.model_name.lower():
            try:
                self.pipeline = transformers.pipeline(
                    "text-generation",
                    model=f"/raid/lingo/models/{self.model_name}",
                    model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
                )
            except:
                pass
        self.use_cache = use_cache
        if use_cache:
            self.openai_cache_file = {
                self.model_name: os.path.join(cache_dir, f"{self.model_name}.jsonl")  # The path to write the cache entries to.
            }
            self.openai_cache = {
                self.model_name: self.load_openai_cache(self.openai_cache_file[self.model_name])  # The openai cache dict. Stores the API responses to avoid duplicate queries.
            }
        else:
            self.openai_cache_file = None
            self.openai_cache = None
        self.context_length=context_length
        self.total_tokens_dict = {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
        self.inference_kwargs = inference_kwargs
        self.logging = logging
        self.save_fn = save_fn
        if "gpt" in self.model_name:
            self.client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
            )
        else:
            self.url = "https://api.together.xyz/v1/chat/completions"
            self.headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "Authorization": f"Bearer {os.environ.get('TOGETHER_API_KEY')}"
            }


    def load_openai_cache(self, openai_cache_file):
        '''Loads the openai cache file into a dict.
        
        Args:
            openai_cache_file (str): The path to the openai cache file.
            
        Returns:
            dict: The openai cache dict.
        '''
        if not openai_cache_file:
            return None
        openai_cache = {}
        if os.path.exists(openai_cache_file):
            with open(openai_cache_file) as f:
                for line in f:
                    openai_cache.update(json.loads(line))
        return openai_cache


    def save_openai_cache(self, new_entry, model_name=None):
        '''Saves the new entry to the openai cache file and updates the openai_cache dict.
        
        Args:
            new_entry (dict): The new entry to save to the cache.
        
        Returns:
            None
        '''
        if model_name is None:
            model_name = self.model_name
        if model_name not in self.openai_cache_file:
            self.openai_cache_file[model_name] = os.path.join("cache", f"{model_name}.jsonl")
        if model_name not in self.openai_cache:
            self.openai_cache[model_name] = self.load_openai_cache(self.openai_cache_file[model_name])
        if self.use_cache:
            with open(self.openai_cache_file[model_name], "a") as wf:
                wf.write(json.dumps(new_entry)+"\n")
            self.openai_cache[model_name].update(new_entry)


    # @retry(wait=wait_random_exponential(min=1, max=60))
    def query_api(self, messages, inference_kwargs=None, model_name=None, use_cache=True):
        '''Queries the OpenAI/Together API with the given messages, or runs inference if model_name = "Meta-Llama-3-8B-Instruct" (local copy of llama 3)
        
        NOTE: This function mutates the messages list to add the new_message and the response from the API.
        
        Args:
            messages (list): A list of past messages to send to the API.
        
        Returns:
            str: The response from the API.
        '''
        if inference_kwargs is None:
            inference_kwargs = self.inference_kwargs
        messages_cache_key = json.dumps(messages)
        if model_name is None:
            model_name = self.model_name
        if model_name not in self.openai_cache:
            if self.use_cache:
                self.openai_cache_file[model_name] = os.path.join("cache", f"{model_name}.jsonl")
                self.openai_cache[model_name] = self.load_openai_cache(self.openai_cache_file[model_name])
            else:
                self.openai_cache[model_name] = None
        if use_cache and self.openai_cache[model_name] and (messages_cache_key in self.openai_cache[model_name]):
            response = self.openai_cache[model_name][messages_cache_key]
        else:
            if "temperature" not in inference_kwargs:
                inference_kwargs["temperature"] = 0.0
            wait_time = 1
            while True:
                try:
                    if "gpt" in model_name:
                        if "repetition_penalty" in inference_kwargs:
                            del inference_kwargs["repetition_penalty"]
                        response = self.client.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            logprobs=True,
                            **inference_kwargs,
                        )
                        response = response.dict()
                    elif "Mixtral" in model_name or "meta-llama/Llama-3" in model_name:
                        payload = {
                            "model": model_name,
                            "messages": messages,
                            "logprobs": True,
                            "echo": True,
                            "stop": ["</s>", "<|eot_id|>", *inference_kwargs.pop("stop", [])],
                            **inference_kwargs,
                        }
                        response = requests.post(self.url, json=payload, headers=self.headers)
                        response = response.json()
                    elif model_name == "Meta-Llama-3-8B-Instruct":
                        prompt = self.pipeline.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        terminators = [
                            self.pipeline.tokenizer.eos_token_id,
                            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                        ]
                        if "max_tokens" in inference_kwargs:
                            inference_kwargs["max_new_tokens"] = inference_kwargs.pop("max_tokens")
                        if "stop" in inference_kwargs:
                            terminators.extend([self.pipeline.tokenizer.encode(stop)[0] for stop in inference_kwargs.pop("stop")])
                        if inference_kwargs.get('temperature', None) == 0:
                            inference_kwargs['do_sample'] = False
                            del inference_kwargs['temperature']
                        if not inference_kwargs.get("do_sample", True):
                            if "top_p" in inference_kwargs:
                                del inference_kwargs["top_p"]
                            if "temperature" in inference_kwargs:
                                del inference_kwargs["temperature"]
                        outputs = self.pipeline(
                            prompt,
                            eos_token_id=terminators,
                            pad_token_id=self.pipeline.tokenizer.eos_token_id,
                            **inference_kwargs,
                        )
                        response_text = outputs[0]["generated_text"][len(prompt):]
                        response = {
                            "prompt": prompt,
                            "output": response_text,
                        }
                    else:
                        raise ValueError(f"Model {model_name} not recognized.")
                    if "error" in response and "gpt" not in model_name:
                        print(response['error'])
                        if "Input validation error:" in str(response['error']["message"]):
                            print("Context length exceeded. Truncating context...")
                            messages[-1]["content"] = messages[-1]["content"].split("***BEGIN STATEMENTS***")[0] + "***BEGIN STATEMENTS***\n" + "\n*===*\n".join(messages[-1]["content"].split("\n*===*\n")[1:])
                            continue
                        print(f"Rate limited. Waiting {wait_time} seconds then retrying...")
                        time.sleep(wait_time)
                        if wait_time < 60:
                            wait_time *= 2
                        continue
                    break
                except Exception as e:
                    print(e)
                    print(response.status_code)
                    if "gpt" not in model_name and response.status_code in [504, 520, 524, 502]:
                        print(response.status_code)
                        print(f"Rate limited. Waiting {wait_time} seconds then retrying...")
                        time.sleep(wait_time)
                        if wait_time < 60:
                            wait_time *= 2
                        # fallback to 8x7b
                        if "8x22B" in model_name and response.status_code == 524:
                            print("Falling back to 7b model")
                            model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
                        continue
                    # get error message
                    if "gpt" in model_name and "context_length_exceeded" in str(e):
                        print("Context length exceeded. Truncating context...")
                        messages = messages[-1:]
                        continue
            self.save_openai_cache({messages_cache_key: response}, model_name=model_name)
        if "gpt" in model_name or "Mixtral" in model_name or "meta-llama/Llama-3" in model_name:
            for tokens_type in response["usage"]:
                self.total_tokens_dict[tokens_type] += response["usage"][tokens_type]
            response_text = response['choices'][0]['message']['content']
        else:
            response_text = response["output"]
        messages.append({'role': 'assistant', 'content': response_text})
        return response_text, response


    def get_total_cost(self):
        """Gets the total cost of queries to the model so far.
        
        Args:
            tokens (dict): The number of prompt_tokens and completion_tokens.
            
        Returns:
            float: The cost in USD.
        """
        return self.total_tokens_dict["completion_tokens"] * api_pricing[self.model_name]["completion_tokens"] + self.total_tokens_dict["prompt_tokens"] * api_pricing[self.model_name]["prompt_tokens"]


    def get_likeliest_answer_fn(self, question: str, answer_choices: List[str], supporting_facts_nl: str, question_ts: int=None, eval_as_list: bool = False):
        if eval_as_list:
            prompt = f"""Read the statements/passages below then answer the question below

***BEGIN STATEMENTS***
{supporting_facts_nl}
***END STATEMENTS***

Given the above statements are true and any prior knowledge you have, answer the following question{' at timestep ' + str(question_ts) if question_ts else ''}?:
{question}

Briefly reason then answer with a JSON list, ["item1", "item2", ...], of zero or more of the following items: {', '.join(answer_choices)}. If you include any of the above items, make sure to copy their names exactly as is from the list. Your list may be empty, [], if none of the answers are true."""
        else:
            prompt = f"""Read the statements/passages below then answer the question below

***BEGIN STATEMENTS***
{supporting_facts_nl}
***END STATEMENTS***

Given the above statements are true and any prior knowledge you have, what is the likeliest answer to the following question{' at timestep ' + str(question_ts) if question_ts else ''}?:
{question}

Briefly reason then answer with one of: {', '.join(answer_choices)}."""
        messages = [
            {"role": "user", "content": prompt},
        ]
        acceptable_answer = False
        while not acceptable_answer:
            response_text, _ = self.query_api(
                messages,
                inference_kwargs={"temperature": 0.0, "max_tokens": 250},
            )
            answer_choices_lowered = [ac.lower() for ac in answer_choices]
            if eval_as_list:
                try:
                    response_text = response_text.strip().strip("```json").strip("```").strip()
                    # split based on '[' and ']'
                    response_text = "[" + response_text.split("[")[1].split("]")[0] + "]"
                    pred_answer = json.loads(response_text)
                    # filter out empty strings
                    pred_answer = [pa for pa in pred_answer if pa]
                    acceptable_answer = all([pa.lower() in answer_choices_lowered for pa in pred_answer])
                except:
                    acceptable_answer = False
            else:
                pred_answer = response_text.split("(")[0].split(".")[0].split("\n")[0].strip().strip('"').strip('"').strip(".").lower()
                acceptable_answer = pred_answer in [ac.lower() for ac in answer_choices_lowered]
            if not acceptable_answer:
                if eval_as_list:
                    messages.append({
                        "role": "user", "content": f"Give your final answer as the JSON list only. Do not include anything else. No explanation or reasoning. Ensure any items in list are exactly taken from the following items: {', '.join(answer_choices)}"
                    })
                else:
                    messages.append({
                        "role": "user", "content": f"Give your final answer as one of {', '.join(answer_choices)}. Do not include any other text. Answer with the mostly likely choice if uncertain."
                    })
                if len(messages) > 4:
                    if eval_as_list:
                        try:
                            pred_answer = json.loads(response_text.strip().strip("```json").strip("```"))
                        except:
                            pred_answer = []
                    else:
                        pred_answer = "Unknown"
                    break

        if self.logging:
            with open(f"{self.save_fn}_gpt_prob_outputs.txt", "a") as wf:
                wf.write(f"[PROMPT:] {prompt}\n\n[PRED ANSWER:] {pred_answer}\n\n*===*\n\n")
            
        return pred_answer



class ConvoModel(LangModel):
    """
    Model for querying the OpenAI API for a response to a HMM task.
    """
    def __init__(
        self, model_name, cache_dir="openai_cache", use_cache=True, inference_kwargs={},
        device="cuda", context_length=4000,
        save_as_facts=False, overwrite_facts=False, retrieve_facts=False,
        save_fn="results/",
        hops=1,
        tokenizer=None,
        logging=False,
        **kwargs,
    ):
        super().__init__(model_name=model_name, cache_dir=cache_dir, use_cache=use_cache, inference_kwargs=inference_kwargs, context_length=context_length, save_fn=save_fn, tokenizer=tokenizer, logging=logging)
        self.metrics = {
            "accuracy": {},
            "accuracy_by_task": {},
            "accuracy_per_num_updates": {},
            "accuracy_per_num_updates_by_task": {},
            "accuracy_per_percent_total_updates": {},
            "accuracy_per_percent_total_updates_by_task": {},
            "score": {},
            "score_by_task": {},
            "score_per_num_updates": {},
            "score_per_num_updates_by_task": {},
            "score_per_percent_total_updates": {},
            "score_per_percent_total_updates_by_task": {},
        }
        self.hops = hops
        self.query_model_name = "gpt-4"
        self.save_as_facts = save_as_facts
        self.overwrite_facts = overwrite_facts
        self.retrieve_facts = retrieve_facts
        self.device = device
        self.data_model = KBModel(
            self.model_name,
            self.save_as_facts, self.overwrite_facts, self.retrieve_facts,
            get_supporting_fact_update_direction=None,
            query_lm_fn=self.query_api, device=self.device,
            context_length=self.context_length, save_fn=self.save_fn, hops=self.hops,
            tokenizer=self.tokenizer, logging=self.logging,
        )

    def score_query(self, question_entry, data_model, save_metadata: bool=False):
        """Scores the question using the KB.
        
        Args:
            question_entry (dict): The question entry.
            data_model (KBModel): The data model.
            
        Returns:
            score (float): The score of the answer.
            context (str): The prompt/context of the answer.
            metadata (dict): Dictionary of metadata. Keys include
                "context": The context of the answer.
                "retrieved_facts_nl": The retrieved facts in natural language.
                "pred_answer": The predicted answer.
        """
        metadata = {}
        query = question_entry["question"]
        possible_answers = question_entry["answer_choices"]
        true_answer = question_entry["answer"]
        # score using lm
        retrieved_facts, retrieved_facts_ts_truthscores, retrieved_fact_scores, retrieved_facts_nl = data_model.retrieve(
            query=query, ts=question_entry["timestamp"], max_sentences_to_retrieve=1000, relevance_threshold=0.7, min_sentences_to_retrieve=5,
        )
        context = data_model.convert_facts_to_nl_story(retrieved_facts, retrieved_facts_ts_truthscores, use_timestamps=True, use_likelihood=False, use_recent=(not self.retrieve_facts))  #, randomize_order=True)
        if save_metadata:
            metadata["context"] = context
            metadata["retrieved_facts_nl"] = retrieved_facts_nl

        pred_answer = self.get_likeliest_answer_fn(
            question=query, answer_choices=possible_answers, supporting_facts_nl=context, question_ts=question_entry["timestamp"], eval_as_list=type(true_answer) == list,
        )
        if type(true_answer) == list:
            try:
                iou = len(set(pred_answer).intersection(set(true_answer))) / len(set(pred_answer).union(set(true_answer))) if len(set(pred_answer).union(set(true_answer))) > 0 else 1
            except:
                iou = 0
            score = iou
        else:
            score = pred_answer.lower() == true_answer.lower()
        metadata["pred_answer"] = pred_answer

        if self.logging:
            with open(self.save_fn + "_question_outputs.txt", "a") as wf:
                wf.write(f"{question_entry['timestamp']} Question: {query}\n\tAnswer: {pred_answer} ({true_answer})\n")
        
        return score, context, metadata


    def infer(self, entry, inference_kwargs=None):
        """Query model for to answer the question after each conversation chunk.
        
        Args:
            entry (dict): The Convo entry to perform inference on.
            
        Returns:
            str: The model's response.
        """
        self.data_model.reset()
        model_scores = []
        metadatas = []
        metadata = {}

        pbar = tqdm(enumerate(entry["actions"]), total=len(entry["actions"]))
        for action_idx, action in pbar:
            if action["action_type"] == "read":
                """
                Read sources
                """
                sources = entry["conversation_per_ts"][action["conversation_chunk_idx"]]
                extracted_facts_to_score, propagated_fact_to_new_score = self.data_model.save_context_facts(
                    sources["conversation_chunk"], sources["timestamp"], sources["conversation_chunk_idx"], sources["timestamp"].strftime("%Y-%m-%d"),
                )

            elif action["action_type"] == "ask":
                question_entry = entry["questions"][action["questions_idx"]]

                """
                Post-eval
                """
                score, _, metadata = self.score_query(
                    question_entry, self.data_model, save_metadata=True)

                """
                Update metrics
                """
                model_score = score
                model_scores.append(model_score)
                self.update_metric(
                    data_id=entry["data_id"],
                    query=question_entry, score=model_score,
                    timestamp=question_entry["timestamp"],
                    num_updates=question_entry["num_updates"],
                    percent_total_updates=question_entry["percent_of_total_updates"],
                    query_type=question_entry["question_type"],
                    task=entry["task"],
                )
                metadatas.append(metadata)
            pbar.set_description(f"Cost: ${self.get_total_cost():.2f}, tokens: {self.total_tokens_dict['total_tokens']}")

        return model_scores, metadatas, self.data_model.stringify_kb()

    def update_metric(self, data_id, query, score, timestamp, num_updates, percent_total_updates, query_type, task):
        """Updates and returns the overall metrics given edit_scores on newest sample.
        
        Args:
            entry (dict): The sample entry. None to return current metrics without updating.
            score (list): The model's score. None to return current metrics without updating.
            
        Updates:
            self.metric (dict): The metrics. Keys include
                "accuracy": The overall accuracy.
                "accuracy_per_task_type": The accuracy per dataset type.
        """
        if query is not None and score is not None:
            with open("temp.txt", "a") as wf:
                wf.write(f"{data_id} {task} {query['timestamp']} Question: {query['question']}\n\tAnswer: {score}\n")
            if task not in self.metrics["accuracy_by_task"]:
                self.metrics["accuracy_by_task"][task] = {}
                self.metrics["score_by_task"][task] = {}
                self.metrics["accuracy_per_num_updates_by_task"][task] = {}
                self.metrics["score_per_num_updates_by_task"][task] = {}
                self.metrics["accuracy_per_percent_total_updates_by_task"][task] = {}
                self.metrics["score_per_percent_total_updates_by_task"][task] = {}
            if num_updates not in self.metrics["accuracy_per_num_updates"]:
                self.metrics["accuracy_per_num_updates"][num_updates] = {}
                self.metrics["score_per_num_updates"][num_updates] = {}
            if num_updates not in self.metrics["accuracy_per_num_updates_by_task"][task]:
                self.metrics["accuracy_per_num_updates_by_task"][task][num_updates] = {}
                self.metrics["score_per_num_updates_by_task"][task][num_updates] = {}
            if percent_total_updates not in self.metrics["accuracy_per_percent_total_updates"]:
                self.metrics["accuracy_per_percent_total_updates"][percent_total_updates] = {}
                self.metrics["score_per_percent_total_updates"][percent_total_updates] = {}
            if percent_total_updates not in self.metrics["accuracy_per_percent_total_updates_by_task"][task]:
                self.metrics["accuracy_per_percent_total_updates_by_task"][task][percent_total_updates] = {}
                self.metrics["score_per_percent_total_updates_by_task"][task][percent_total_updates] = {}
            if data_id not in self.metrics["accuracy"]:
                self.metrics["accuracy"][data_id] = []
                self.metrics["score"][data_id] = []
            if data_id not in self.metrics["accuracy_per_num_updates"][num_updates]:
                self.metrics["accuracy_per_num_updates"][num_updates][data_id] = []
                self.metrics["score_per_num_updates"][num_updates][data_id] = []
            if data_id not in self.metrics["accuracy_per_percent_total_updates"][percent_total_updates]:
                self.metrics["accuracy_per_percent_total_updates"][percent_total_updates][data_id] = []
                self.metrics["score_per_percent_total_updates"][percent_total_updates][data_id] = []
            if data_id not in self.metrics["accuracy_by_task"][task]:
                self.metrics["accuracy_by_task"][task][data_id] = []
                self.metrics["score_by_task"][task][data_id] = []
            if data_id not in self.metrics["accuracy_per_num_updates_by_task"][task][num_updates]:
                self.metrics["accuracy_per_num_updates_by_task"][task][num_updates][data_id] = []
                self.metrics["score_per_num_updates_by_task"][task][num_updates][data_id] = []
            if data_id not in self.metrics["accuracy_per_percent_total_updates_by_task"][task][percent_total_updates]:
                self.metrics["accuracy_per_percent_total_updates_by_task"][task][percent_total_updates][data_id] = []
                self.metrics["score_per_percent_total_updates_by_task"][task][percent_total_updates][data_id] = []

            self.metrics["accuracy"][data_id].append(score == 1)
            self.metrics["score"][data_id].append(score)
            self.metrics["accuracy_per_num_updates"][num_updates][data_id].append(score == 1)
            self.metrics["score_per_num_updates"][num_updates][data_id].append(score)
            self.metrics["accuracy_per_percent_total_updates"][percent_total_updates][data_id].append(score == 1)
            self.metrics["score_per_percent_total_updates"][percent_total_updates][data_id].append(score)

            self.metrics["accuracy_by_task"][task][data_id].append(score == 1)
            self.metrics["score_by_task"][task][data_id].append(score)

            self.metrics["accuracy_per_num_updates_by_task"][task][num_updates][data_id].append(score == 1)
            self.metrics["score_per_num_updates_by_task"][task][num_updates][data_id].append(score)
            self.metrics["accuracy_per_percent_total_updates_by_task"][task][percent_total_updates][data_id].append(score == 1)
            self.metrics["score_per_percent_total_updates_by_task"][task][percent_total_updates][data_id].append(score)


    def get_metrics(self, entry=None, prediction=None):
        """Updates and returns the overall metrics given prediction on newest sample.
        
        Args:
            entry (dict): The sample entry. None to return current metrics without updating.
            prediction (str): The model's prediction. None to return current metrics without updating.
            
        Returns:
            dict: The metrics. Keys include
                "accuracy": The overall accuracy.
                "accuracy_per_task_type": The accuracy per dataset type.
        """
        per_datum_metrics = {
            "accuracy": {},
            "score": {},
            "accuracy_by_task": {},
            "accuracy_per_num_updates": {},
            "accuracy_per_num_updates_by_task": {},
            "accuracy_per_percent_total_updates": {},
            "accuracy_per_percent_total_updates_by_task": {},
            "score_by_task": {},
            "score_per_num_updates": {},
            "score_per_num_updates_by_task": {},
            "score_per_percent_total_updates": {},
            "score_per_percent_total_updates_by_task": {},
        }
        # first aggregate within each datum
        for data_id in self.metrics["accuracy"]:
            per_datum_metrics["accuracy"][data_id] = sum(self.metrics["accuracy"][data_id]) / len(self.metrics["accuracy"][data_id])
            per_datum_metrics["score"][data_id] = sum(self.metrics["score"][data_id]) / len(self.metrics["score"][data_id])

            for num_updates in self.metrics["accuracy_per_num_updates"]:
                if data_id in self.metrics["accuracy_per_num_updates"][num_updates]:
                    if num_updates not in per_datum_metrics["accuracy_per_num_updates"]:
                        per_datum_metrics["accuracy_per_num_updates"][num_updates] = {}
                        per_datum_metrics["score_per_num_updates"][num_updates] = {}
                    per_datum_metrics["accuracy_per_num_updates"][num_updates][data_id] = sum(self.metrics["accuracy_per_num_updates"][num_updates][data_id]) / len(self.metrics["accuracy_per_num_updates"][num_updates][data_id])
                    per_datum_metrics["score_per_num_updates"][num_updates][data_id] = sum(self.metrics["score_per_num_updates"][num_updates][data_id]) / len(self.metrics["score_per_num_updates"][num_updates][data_id])

                    for task in self.metrics["accuracy_per_num_updates_by_task"]:
                        if task not in per_datum_metrics["accuracy_per_num_updates_by_task"]:
                            per_datum_metrics["accuracy_per_num_updates_by_task"][task] = {}
                            per_datum_metrics["score_per_num_updates_by_task"][task] = {}
                        if num_updates not in per_datum_metrics["accuracy_per_num_updates_by_task"][task]:
                            per_datum_metrics["accuracy_per_num_updates_by_task"][task][num_updates] = {}
                            per_datum_metrics["score_per_num_updates_by_task"][task][num_updates] = {}
                        if data_id in self.metrics["accuracy_per_num_updates_by_task"][task][num_updates]:
                            per_datum_metrics["accuracy_per_num_updates_by_task"][task][num_updates][data_id] = sum(self.metrics["accuracy_per_num_updates_by_task"][task][num_updates][data_id]) / len(self.metrics["accuracy_per_num_updates_by_task"][task][num_updates][data_id])
                            per_datum_metrics["score_per_num_updates_by_task"][task][num_updates][data_id] = sum(self.metrics["score_per_num_updates_by_task"][task][num_updates][data_id]) / len(self.metrics["score_per_num_updates_by_task"][task][num_updates][data_id])

            for percent_total_updates in self.metrics["accuracy_per_percent_total_updates"]:
                if data_id in self.metrics["accuracy_per_percent_total_updates"][percent_total_updates]:
                    if percent_total_updates not in per_datum_metrics["accuracy_per_percent_total_updates"]:
                        per_datum_metrics["accuracy_per_percent_total_updates"][percent_total_updates] = {}
                        per_datum_metrics["score_per_percent_total_updates"][percent_total_updates] = {}
                    per_datum_metrics["accuracy_per_percent_total_updates"][percent_total_updates][data_id] = sum(self.metrics["accuracy_per_percent_total_updates"][percent_total_updates][data_id]) / len(self.metrics["accuracy_per_percent_total_updates"][percent_total_updates][data_id])
                    per_datum_metrics["score_per_percent_total_updates"][percent_total_updates][data_id] = sum(self.metrics["score_per_percent_total_updates"][percent_total_updates][data_id]) / len(self.metrics["score_per_percent_total_updates"][percent_total_updates][data_id])
                    for task in self.metrics["accuracy_per_percent_total_updates_by_task"]:
                        if task not in per_datum_metrics["accuracy_per_percent_total_updates_by_task"]:
                            per_datum_metrics["accuracy_per_percent_total_updates_by_task"][task] = {}
                            per_datum_metrics["score_per_percent_total_updates_by_task"][task] = {}
                        if percent_total_updates not in per_datum_metrics["accuracy_per_percent_total_updates_by_task"][task]:
                            per_datum_metrics["accuracy_per_percent_total_updates_by_task"][task][percent_total_updates] = {}
                            per_datum_metrics["score_per_percent_total_updates_by_task"][task][percent_total_updates] = {}
                        if data_id in self.metrics["accuracy_per_percent_total_updates_by_task"][task][percent_total_updates]:
                            per_datum_metrics["accuracy_per_percent_total_updates_by_task"][task][percent_total_updates][data_id] = sum(self.metrics["accuracy_per_percent_total_updates_by_task"][task][percent_total_updates][data_id]) / len(self.metrics["accuracy_per_percent_total_updates_by_task"][task][percent_total_updates][data_id])
                            per_datum_metrics["score_per_percent_total_updates_by_task"][task][percent_total_updates][data_id] = sum(self.metrics["score_per_percent_total_updates_by_task"][task][percent_total_updates][data_id]) / len(self.metrics["score_per_percent_total_updates_by_task"][task][percent_total_updates][data_id])

        data_metrics = {
            "accuracy": [per_datum_metrics["accuracy"][data_id] for data_id in per_datum_metrics["accuracy"]],
            "score": [per_datum_metrics["score"][data_id] for data_id in per_datum_metrics["score"]],
            "accuracy_by_task": {task: [
                per_datum_metrics["accuracy_by_task"][task][data_id] for data_id in per_datum_metrics["accuracy_by_task"][task]
            ] for task in per_datum_metrics["accuracy_by_task"]},
            "score_by_task": {task: [
                per_datum_metrics["score_by_task"][task][data_id] for data_id in per_datum_metrics["score_by_task"][task]
            ] for task in per_datum_metrics["score_by_task"]},
            "accuracy_per_num_updates": {num_updates: [
                per_datum_metrics["accuracy_per_num_updates"][num_updates][data_id] for data_id in per_datum_metrics["accuracy_per_num_updates"][num_updates]
            ] for num_updates in per_datum_metrics["accuracy_per_num_updates"]},
            "score_per_num_updates": {num_updates: [
                per_datum_metrics["score_per_num_updates"][num_updates][data_id] for data_id in per_datum_metrics["score_per_num_updates"][num_updates]
            ] for num_updates in per_datum_metrics["score_per_num_updates"]},
            "accuracy_per_percent_total_updates": {percent_total_updates: [
                per_datum_metrics["accuracy_per_percent_total_updates"][percent_total_updates][data_id] for data_id in per_datum_metrics["accuracy_per_percent_total_updates"][percent_total_updates]
            ] for percent_total_updates in per_datum_metrics["accuracy_per_percent_total_updates"]},
            "score_per_percent_total_updates": {percent_total_updates: [
                per_datum_metrics["score_per_percent_total_updates"][percent_total_updates][data_id] for data_id in per_datum_metrics["score_per_percent_total_updates"][percent_total_updates]
            ] for percent_total_updates in per_datum_metrics["score_per_percent_total_updates"]},
            "accuracy_per_num_updates_by_task": {task: {num_updates: [
                per_datum_metrics["accuracy_per_num_updates_by_task"][task][num_updates][data_id] for data_id in per_datum_metrics["accuracy_per_num_updates_by_task"][task][num_updates]
            ] for num_updates in per_datum_metrics["accuracy_per_num_updates_by_task"][task]} for task in per_datum_metrics["accuracy_per_num_updates_by_task"]},
            "score_per_num_updates_by_task": {task: {num_updates: [
                per_datum_metrics["score_per_num_updates_by_task"][task][num_updates][data_id] for data_id in per_datum_metrics["score_per_num_updates_by_task"][task][num_updates]
            ] for num_updates in per_datum_metrics["score_per_num_updates_by_task"][task]} for task in per_datum_metrics["score_per_num_updates_by_task"]},
            "accuracy_per_percent_total_updates_by_task": {task: {percent_total_updates: [
                per_datum_metrics["accuracy_per_percent_total_updates_by_task"][task][percent_total_updates][data_id] for data_id in per_datum_metrics["accuracy_per_percent_total_updates_by_task"][task][percent_total_updates]
            ] for percent_total_updates in per_datum_metrics["accuracy_per_percent_total_updates_by_task"][task]} for task in per_datum_metrics["accuracy_per_percent_total_updates_by_task"]},
            "score_per_percent_total_updates_by_task": {task: {percent_total_updates: [
                per_datum_metrics["score_per_percent_total_updates_by_task"][task][percent_total_updates][data_id] for data_id in per_datum_metrics["score_per_percent_total_updates_by_task"][task][percent_total_updates]
            ] for percent_total_updates in per_datum_metrics["score_per_percent_total_updates_by_task"][task]} for task in per_datum_metrics["score_per_percent_total_updates_by_task"]},
        }

        return data_metrics


class NewsModel(LangModel):
    """
    Model for querying the OpenAI API for a response to a HMM task.
    """
    def __init__(
        self, model_name, cache_dir="openai_cache", use_cache=True, inference_kwargs={},
        device="cuda", context_length=4000,
        save_as_facts=False, overwrite_facts=False, retrieve_facts=False,
        save_fn="results/",
        hops=1,
        tokenizer=None,
        logging=False,
        **kwargs,
    ):
        super().__init__(model_name=model_name, cache_dir=cache_dir, use_cache=use_cache, inference_kwargs=inference_kwargs, context_length=context_length, save_fn=save_fn, tokenizer=tokenizer, logging=logging)
        self.metrics = {
            "accuracy": [0,0],
            "accuracy_per_ts": {},
            "accuracy_per_num_updates": {},
            "accuracy_per_percent_total_updates": {},
            "accuracy_per_ts_by_type": {},
        }
        self.hops = hops
        self.query_model_name = "gpt-4"
        self.save_as_facts = save_as_facts
        self.overwrite_facts = overwrite_facts
        self.retrieve_facts = retrieve_facts
        self.device = device
        self.model_scores_per_ts = {}
        self.model_scores_per_num_updates = {}
        self.model_scores_per_percent_total_updates = {}
        self.model_scores_per_ts_by_type = {}

    def get_supporting_fact_update_direction(self, supporting_facts_nl: str, fact_nl: str, fact_ts: int):
        """
        NOT as reliable
        """
        # score using lm
        prompt = f"{supporting_facts_nl}\n\nSuppose the above facts are true. Has the probability that \"{fact_nl}\" is true at timestep {fact_ts} gone up, down, or stayed the same? Only output \"gone up\", \"gone down\", or \"stayed the same\" and nothing else."
        messages = [{"role": "user", "content": prompt}]
        ANSWER_TO_DIRECTION = {
            "gone up": 1,
            "gone down": -1,
            "stayed the same": 0,
        }
        update_direction = None
        while update_direction is None:
            try:
                response, _ = self.query_api(messages, inference_kwargs={"temperature": 0.0})
                update_direction = ANSWER_TO_DIRECTION[response]
            except:
                messages.append({
                    "role": "user", "content": "Please output one of \"gone up\", \"gone down\", or \"stay the same\" and nothing else."
                })
        return update_direction

    def parse_model_response(self, response):
        """Parses the model's response to get the objects.
        
        Args:
            response (str): The model's response.
            
        Returns:
            set: The objects in the response.
        """
        if self.binary_answers:
            return response.strip().replace("'", "").replace('"', '').replace(".", "")
        else:
            response = response.strip().replace("'", "").replace('"', '')
            response = response.split(",")
            response = [r.strip() for r in response]
            return response

    def score_query(self, question_entry, data_model, save_metadata: bool=False):
        """Scores the question using the KB.
        
        Args:
            question_entry (dict): The question entry.
            data_model (KBModel): The data model.
            
        Returns:
            score (float): The score of the answer.
            context (str): The prompt/context of the answer.
            metadata (dict): Dictionary of metadata. Keys include
                "context": The context of the answer.
                "retrieved_facts_nl": The retrieved facts in natural language.
                "pred_answer": The predicted answer.
        """
        metadata = {}
        query = question_entry["question"]
        possible_answers = question_entry["answer_choices"]
        true_answer = question_entry["answer"]
        # score using lm
        retrieved_facts, retrieved_facts_ts_truthscores, retrieved_fact_scores, retrieved_facts_nl = data_model.retrieve(
            query=query, ts=question_entry["timestamp"], max_sentences_to_retrieve=1000, relevance_threshold=0.7, min_sentences_to_retrieve=5,
        )
        context = data_model.convert_facts_to_nl_story(retrieved_facts, retrieved_facts_ts_truthscores, use_timestamps=True, use_likelihood=False, use_recent=(not self.retrieve_facts))  #, randomize_order=True)
        if save_metadata:
            metadata["context"] = context
            metadata["retrieved_facts_nl"] = retrieved_facts_nl

        pred_answer = self.get_likeliest_answer_fn(
            question=query, answer_choices=possible_answers, supporting_facts_nl=context, question_ts=question_entry["timestamp"], eval_as_list=type(true_answer) == list,
        )
        score = pred_answer.lower() == true_answer.lower()
        metadata["pred_answer"] = pred_answer

        if self.logging:
            with open(self.save_fn + "_question_outputs.txt", "a") as wf:
                wf.write(f"{question_entry['timestamp']} Question: {query}\n\tAnswer: {pred_answer} ({true_answer})\n")
        
        return score, context, metadata

    def infer(self, entry, inference_kwargs=None):
        """Query an OpenAI model for to answer the question after each context.
        
        Args:
            entry (dict): The bAbI entry to perform inference on.
            
        Returns:
            str: The model's response.
        """
        data_model = KBModel(
            self.model_name,
            self.save_as_facts, self.overwrite_facts, self.retrieve_facts,
            get_supporting_fact_update_direction=self.get_supporting_fact_update_direction,
            query_lm_fn=self.query_api, device=self.device, context_length=self.context_length, save_fn=self.save_fn, hops=self.hops,
            tokenizer=self.tokenizer, logging=self.logging,
        )
        model_scores = []
        accuracy = 0.0
        pbar = tqdm(
            entry["actions"], desc=f"Acc: {accuracy:.1f}",
        )
        metadatas = []
        metadata = {}

        for action_idx, action in enumerate(pbar):
            if action["action_type"] == "read":
                """
                Read sources
                """
                sources = entry["sources"][action["sources_idx"]]
                extracted_facts_to_score, propagated_fact_to_new_score = data_model.save_context_facts(
                    sources["context"], sources["source_ts"], sources["chunk_idx"], sources["url"], sources["read_ts"],
                )
                accuracy = 0 if self.metrics['accuracy'][1] == 0 else self.metrics['accuracy'][0] / self.metrics['accuracy'][1] * 100
                ts_to_display = action['timestamp'].strftime("%Y-%m-%d")
                pbar.set_description(f"{action['action_type']}@{ts_to_display}, Acc: {accuracy:.1f}, Cost: ${self.get_total_cost():.2f}")
            elif action["action_type"] == "ask":
                question_entry = entry["questions"][action["questions_idx"]]
                if question_entry["timestamp"] not in self.model_scores_per_ts:
                    self.model_scores_per_ts[question_entry["timestamp"]] = []
                if question_entry["num_updates"] not in self.model_scores_per_num_updates:
                    # how many times has answer to this question changed at this point in time
                    self.model_scores_per_num_updates[question_entry["num_updates"]] = []
                if question_entry["percent_of_total_updates"] not in self.model_scores_per_percent_total_updates:
                    self.model_scores_per_percent_total_updates[question_entry["percent_of_total_updates"]] = []
                    self.model_scores_per_ts_by_type[question_entry["timestamp"]] = {}
                if question_entry["question_type"] not in self.model_scores_per_ts_by_type[question_entry["timestamp"]]:
                    self.model_scores_per_ts_by_type[question_entry["timestamp"]][question_entry["question_type"]] = []

                """
                Eval
                """
                score, _, metadata = self.score_query(
                    question_entry, data_model, save_metadata=True)

                """
                Update metrics
                """
                model_score = score
                model_scores.append(model_score)
                self.model_scores_per_ts[question_entry["timestamp"]].append(model_score)
                self.model_scores_per_num_updates[question_entry["num_updates"]].append(model_score)
                self.model_scores_per_percent_total_updates[question_entry["percent_of_total_updates"]].append(model_score)
                self.model_scores_per_ts_by_type[question_entry["timestamp"]][question_entry["question_type"]].append(model_score)
                self.update_metric(
                    query=question_entry, score=model_score,
                    timestamp=question_entry["timestamp"],
                    num_updates=question_entry["num_updates"],
                    percent_total_updates=question_entry["percent_of_total_updates"],
                    query_type=question_entry["question_type"],
                )
                ts_accuracy = 0 if self.metrics['accuracy_per_ts'][question_entry['timestamp']][1] == 0 else self.metrics['accuracy_per_ts'][question_entry['timestamp']][0] / self.metrics['accuracy_per_ts'][question_entry['timestamp']][1] * 100
                ts_to_display = action['timestamp'].strftime("%Y-%m-%d")
                pbar.set_description(f"{action['action_type']}@{ts_to_display}, Acc: {ts_accuracy:.1f}, Cost: ${self.get_total_cost():.2f}")
                metadatas.append(metadata)
                # last ask of this phase
                if action_idx + 1 == len(entry['actions']) or entry['actions'][action_idx + 1]["action_type"] == "read":
                    for ts in self.metrics['accuracy_per_ts']:
                        print(ts, f"{self.metrics['accuracy_per_ts'][ts][0] / self.metrics['accuracy_per_ts'][ts][1] * 100:.1f}")
                        for q_type in self.metrics['accuracy_per_ts_by_type'][ts]:
                            print("   ", q_type, f"{self.metrics['accuracy_per_ts_by_type'][ts][q_type][0] / self.metrics['accuracy_per_ts_by_type'][ts][q_type][1] * 100:.1f}")

        return model_scores, metadatas, data_model.stringify_kb()

    def update_metric(self, query, score, timestamp, num_updates, percent_total_updates, query_type):
        """Updates and returns the overall metrics given edit_scores on newest sample.
        
        Args:
            entry (dict): The sample entry. None to return current metrics without updating.
            score (list): The model's score. None to return current metrics without updating.
            
        Updates:
            self.metric (dict): The metrics. Keys include
                "accuracy": The overall accuracy.
                "accuracy_per_task_type": The accuracy per dataset type.
        """
        if query is not None and score is not None:
            self.metrics["accuracy"][0] += score > 0.0  #predicted_answer == gt_answer
            self.metrics["accuracy"][1] += 1
            if timestamp not in self.metrics["accuracy_per_ts"]:
                self.metrics["accuracy_per_ts"][timestamp] = [0,0]
            self.metrics["accuracy_per_ts"][timestamp][0] += score > 0.5
            self.metrics["accuracy_per_ts"][timestamp][1] += 1
            if num_updates not in self.metrics["accuracy_per_num_updates"]:
                self.metrics["accuracy_per_num_updates"][num_updates] = [0,0]
            self.metrics["accuracy_per_num_updates"][num_updates][0] += score > 0.5
            self.metrics["accuracy_per_num_updates"][num_updates][1] += 1
            if percent_total_updates not in self.metrics["accuracy_per_percent_total_updates"]:
                self.metrics["accuracy_per_percent_total_updates"][percent_total_updates] = [0,0]
            self.metrics["accuracy_per_percent_total_updates"][percent_total_updates][0] += score > 0.5
            self.metrics["accuracy_per_percent_total_updates"][percent_total_updates][1] += 1
            if timestamp not in self.metrics["accuracy_per_ts_by_type"]:
                self.metrics["accuracy_per_ts_by_type"][timestamp] = {}
            if query_type not in self.metrics["accuracy_per_ts_by_type"][timestamp]:
                self.metrics["accuracy_per_ts_by_type"][timestamp][query_type] = [0,0]
            self.metrics["accuracy_per_ts_by_type"][timestamp][query_type][0] += score > 0.5
            self.metrics["accuracy_per_ts_by_type"][timestamp][query_type][1] += 1


    def get_metrics(self, entry=None, prediction=None):
        """Updates and returns the overall metrics given prediction on newest sample.
        
        Args:
            entry (dict): The sample entry. None to return current metrics without updating.
            prediction (str): The model's prediction. None to return current metrics without updating.
            
        Returns:
            dict: The metrics. Keys include
                "accuracy": The overall accuracy.
                "accuracy_per_task_type": The accuracy per dataset type.
        """

        return self.metrics

