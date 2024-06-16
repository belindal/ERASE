from typing import Any, Dict, List
from torch.utils.data import Dataset
from glob import glob
import os
import random
import pandas as pd
import json
import ast
import nltk.data
from tqdm import tqdm
from datetime import datetime, timezone
from transformers import AutoTokenizer
from utils import TOKENIZER, count_tokens

random.seed(0)

PROMPT_TOKENS = 365

class ConvoDataset(Dataset):
    def __init__(self, data_dir="synth-convo/data", tasks="singlehop_convos,multihop_convos", context_length=1000, tokenizer=None):
        self.tasks = tasks.split(",")
        self.nltk_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = TOKENIZER[self.model_name]
            if "meta-llama" in self.model_name.lower():
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(f"/raid/lingo/models/{self.model_name}")
                except:
                    self.tokenizer = AutoTokenizer.from_pretrained(f"/raid/lingo/models/Meta-Llama-3-8B-Instruct")
        self.context_length = context_length
        self.data = self.load_data(data_dir, tasks)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def load_questions(self, questions: pd.DataFrame) -> dict:
        # load questions from the questions.csv file
        question_to_ts_to_answers = {}
        for index, question_entry in questions.iterrows():
            question = question_entry["question"]
            answer = question_entry["answer"]
            if answer != answer:
                answer = "None"
            if question not in question_to_ts_to_answers:
                question_to_ts_to_answers[question] = {}
            ts = pd.Timestamp(question_entry.get("timestamp", question_entry["timestamp"]), tzinfo=timezone.utc)
            question_to_ts_to_answers[question][ts] = {"answer": eval(answer) if answer.startswith("['") or answer.startswith("[]") else answer}
            answer_choices = question_entry["potential_answers"]
            assert answer_choices == answer_choices  # not nan
            question_to_ts_to_answers[question][ts]["answer_choices"] = sorted(eval(answer_choices))
        return question_to_ts_to_answers

    def get_num_answer_updates_from_timestamp(self, timestamp: pd.Timestamp, question_to_ts_to_answers: dict):
        # get the number of times the answer to the question has been updated up to the given timestamp
        num_updates = 0
        sorted_qa = sorted(question_to_ts_to_answers)
        num_updates = sorted_qa.index(timestamp)
        return num_updates

    def divide_chunks(self, turns, ts, n=500):
        turns = [f"[CONVERSATION AT {ts}]", *turns, "[END CONVERSATION]"]
        chunks = [""]
        for turn in turns:
            if type(turn) == dict:
                turn_nl = f"{turn['role']}: {turn['content']}"
            else:
                turn_nl = turn
            if count_tokens(self.tokenizer, chunks[-1] + "\n" + turn_nl) >= n:
                chunks[-1] = chunks[-1].strip()
                # add paragraph to context
                if count_tokens(self.tokenizer, chunks[-1]) > n:
                    chunks = chunks[:-1]
                chunks.append("")
            chunks[-1] += "\n" + turn_nl
            
        for chunk in chunks:
            assert count_tokens(self.tokenizer, chunk) < self.context_length - PROMPT_TOKENS
        return chunks

    def parse_convo(self, conversation: List[Dict[str, Any]], questions: pd.DataFrame, question_to_ts_to_answers: dict = None, conv_id: int = 0, task: str = "singlehop_convos"):
        """
        [sources to read (up to ts)] --> [questions to ask]
        [sources to read (up to ts)] --> [questions to ask]
        [sources to read (up to ts)] --> [questions to ask]
        Returns
        {
            sources (List[Dict[str, str]]):
                [
                    {"url": url, "timestamp": ts, "context": context, "context_sentences": context_sentences},
                    ...
                ]
            questions:
                [
                    {"question": q, "answer": a, "timestamp": ts, "source_cutoff": source_cutoff_idx},
                    ...
                ]
        }
        """
        data_entry = {
            "conversation_per_ts": [],
            "questions": [],
            "actions": [],
            "task": task,
        }

        last_added_source_index = 0

        percent_of_total_updates = []

        ts_to_conv_turns = {}
        for turn in conversation:
            ts = pd.Timestamp(datetime.strptime(turn["ts"], "%Y-%m-%d"), tzinfo=timezone.utc)
            if ts not in ts_to_conv_turns:
                ts_to_conv_turns[ts] = []
            ts_to_conv_turns[ts].append(turn)
        sorted_convchunks_ts = sorted(list(ts_to_conv_turns.keys()))

        total_updates_overall = 0
        for question in question_to_ts_to_answers:
            total_updates_overall += self.get_num_answer_updates_from_timestamp(sorted_convchunks_ts[-1], question_to_ts_to_answers[question])
        ts_to_percent_of_total_updates = {}
        for timestamp in sorted_convchunks_ts:
            total_updates_at_ts = 0.0
            for question in question_to_ts_to_answers:
                total_updates_at_ts += self.get_num_answer_updates_from_timestamp(timestamp, question_to_ts_to_answers[question])
            ts_to_percent_of_total_updates[timestamp] = total_updates_at_ts / total_updates_overall

        question_to_last_answer = {}
        token_counts = []
        all_contexts = set()

        assert len(sorted_convchunks_ts) == 12
        for s_idx, timestamp in enumerate(sorted_convchunks_ts):
            chunks = self.divide_chunks(ts_to_conv_turns[timestamp], timestamp, n=int(self.context_length / 2))
            for chunk_idx, chunk in enumerate(chunks):
                data_entry["conversation_per_ts"].append({
                    "conversation_chunk": chunk,
                    "conversation_chunk_idx": chunk_idx,
                    "timestamp": timestamp,
                    "data_id": conv_id,
                })
                data_entry["actions"].append({
                    "action_type": "read",
                    "timestamp": timestamp,
                    "conversation_chunk_idx": len(data_entry["conversation_per_ts"]) - 1,
                    "data_id": conv_id,
                })
                last_added_source_index += 1

            if (s_idx == len(sorted_convchunks_ts)-1) or (sorted_convchunks_ts[s_idx+1] != timestamp):
                if s_idx == 0 or s_idx == 2 or s_idx == 4 or s_idx == 7 or s_idx == 9 or s_idx == 11:
                    # last timestamp of question increment
                    total_updates_at_ts = 0.0
                    for question in question_to_ts_to_answers:
                        total_updates_at_ts += self.get_num_answer_updates_from_timestamp(timestamp, question_to_ts_to_answers[question])
                    percent_of_total_updates.append(total_updates_at_ts / total_updates_overall)
                    question_to_answer = {question: question_to_ts_to_answers[question][timestamp] for question in question_to_ts_to_answers}
                    changed_questions = {question for question in question_to_ts_to_answers if question_to_last_answer.get(question) != question_to_answer[question] and question_to_answer[question] is not None}
                    # changed_questions + some sample of unchanged questions
                    unchanged_questions = [question for question in question_to_ts_to_answers if question not in changed_questions and question_to_answer.get(question) is not None]
                    if timestamp not in self.timestamp_to_questions:
                        questions_to_ask = list(changed_questions) + random.sample(
                            unchanged_questions, min(len(changed_questions), len(unchanged_questions)))
                        self.timestamp_to_questions[timestamp] = questions_to_ask
                    else:
                        questions_to_ask = self.timestamp_to_questions[timestamp]
                    for question in question_to_ts_to_answers:
                        answer_at_timestamp = question_to_ts_to_answers[question][timestamp]
                        if question not in questions_to_ask:
                            continue
                        question_to_last_answer[question] = answer_at_timestamp
                        if answer_at_timestamp is not None:
                            answer_choices = answer_at_timestamp["answer_choices"]
                            if type(answer_choices) == str:
                                answer_choices = ast.literal_eval(answer_choices)
                            if len(answer_choices) == 1:
                                continue
                            answer = answer_at_timestamp["answer"]
                            if type(answer) == str:
                                assert answer in answer_choices
                            else:
                                for a in answer:
                                    assert a in answer_choices
                            convchunk_nl = {timestamp: "\n".join(f"{turn['role']}: {turn['content']}" for turn in ts_to_conv_turns[timestamp]) for timestamp in ts_to_conv_turns}
                            data_entry["questions"].append({
                                "question": question,
                                "answer": answer,
                                "answer_choices": sorted(answer_choices),
                                "timestamp": timestamp,
                                "num_updates": self.get_num_answer_updates_from_timestamp(timestamp, question_to_ts_to_answers[question]),
                                "percent_of_total_updates": total_updates_at_ts / total_updates_overall,
                                # all previous chunks
                                "context": "\n*===*\n".join([
                                    f"[CONVERSATION AT {ts}]\n{convchunk_nl[ts]}\n[END CONVERSATION]" for ts in sorted_convchunks_ts[:s_idx+1]
                                ]),
                                "question_type": "TODO",
                                "data_id": conv_id,
                                "task": task,
                            })
                            data_entry["actions"].append({
                                "action_type": "ask",
                                "timestamp": timestamp,
                                "questions_idx": len(data_entry["questions"]) - 1,
                                "data_id": conv_id,
                            })
                            token_counts.append(count_tokens(self.tokenizer, data_entry["questions"][-1]["context"]))
        data_entry["actions"] = sorted(data_entry["actions"], key=lambda x: (x["timestamp"], x["action_type"] == "ask", x["conversation_chunk_idx"] if x["action_type"] == "read" else x["questions_idx"]))
        data_entry["data_id"] = conv_id
        return data_entry, token_counts[-1]

    def load_data(self, data_dir, tasks):
        """
        Loads the wikipedia dataset from the given directory pattern.

        Args:
            data_dir (str): directory to load data from
            stories (list[int]): list of stories to load
            
        Returns:
            data_entries (list):
                list of dicts with the following keys:
                    context (str): full story
                    context_sentences (list[str]): list of sentences in the story
                    question (str): question about the story
                    answer (str): answer to the question
                    answer_vocab (set[str]): set of all possible answers for questions about the story
                    context_id (int): id of the context
                    story_id (str): id of the story
                    task (int): task type
        """
        data_entries = []
        for task in self.tasks:
            token_counts = []
            all_conversation_files = set(glob(os.path.join(data_dir, task, "conversation_*.json")))
            for conversation_file in tqdm(all_conversation_files):
                conv_id = int(conversation_file.split("_")[-1].replace(".json", ""))
                if os.path.exists(os.path.join(data_dir, str(conv_id), "timestamp_to_questions.json")):
                    timestamp_to_questions_dict = json.load(open(os.path.join(data_dir, str(conv_id), "timestamp_to_questions.json")))
                    self.timestamp_to_questions = {
                        pd.Timestamp(key, tzinfo=timezone.utc): value for key, value in timestamp_to_questions_dict.items()
                    }
                else:
                    self.timestamp_to_questions = {}
                conversation = json.load(open(conversation_file))
                # load questions
                questions = pd.read_csv(os.path.join(data_dir, "questions", f"conversation_question_answers_{conv_id}.csv"))
                question_to_ts_to_answers = self.load_questions(questions)

                data_entry, conv_token_counts = self.parse_convo(conversation, questions, question_to_ts_to_answers, conv_id, task)
                data_entries.append(data_entry)
                token_counts.append(conv_token_counts)
                os.makedirs(os.path.join(data_dir, str(conv_id)), exist_ok=True)
                with open(os.path.join(data_dir, str(conv_id), "timestamp_to_questions.json"), "w") as f:
                    timestamp_to_questions_dict = {
                        key.strftime("%Y-%m-%dT%H:%M:%SZ"): value for key, value in self.timestamp_to_questions.items()
                    }
                    json.dump(timestamp_to_questions_dict, f, indent=4)
            print(task, "token counts:", token_counts)
            if len(token_counts) > 0:
                print("Average context tokens:", sum(token_counts) / len(token_counts))
            print("Loaded", len(data_entries), "entries for tasks", tasks)
        data_entries = sorted(data_entries, key=lambda x: x["questions"][0]["data_id"])
        return data_entries
