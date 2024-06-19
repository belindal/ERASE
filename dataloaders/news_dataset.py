from torch.utils.data import Dataset
from glob import glob
import os
import random
import pandas as pd
import json
import ast
import nltk.data
from tqdm import tqdm
from datetime import timezone
from transformers import AutoTokenizer
from utils import TOKENIZER, count_tokens


random.seed(0)

PROMPT_TOKENS = 365

class NewsDataset(Dataset):
    def __init__(self, data_dir="CLARK_news/", tasks="full", context_length=1000, tokenizer=None):
        self.nltk_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.data = self.load_data(data_dir, tasks)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def extract_article_sentences(self, article):
        # split into sentences
        title = article.split("\n\n")[0]
        article_body = article[len(title):].lstrip()
        sentences = [title, *self.nltk_tokenizer.tokenize(article_body)]
        return sentences

    def load_questions(self, questions: pd.DataFrame) -> dict:
        # load questions from the questions.csv file
        question_to_ts_to_answers = {}
        for index, question_entry in questions.iterrows():
            question = question_entry["Question"]
            answer = question_entry["Answer"]
            if answer != answer:
                answer = "None"
            if question not in question_to_ts_to_answers:
                question_to_ts_to_answers[question] = {}
            start_ts = pd.Timestamp(question_entry.get("Known start timestamp", question_entry["Start timestamp"]), tzinfo=timezone.utc)
            end_ts = pd.Timestamp(question_entry.get("Known end timestamp", question_entry["End timestamp"]), tzinfo=timezone.utc)
            question_to_ts_to_answers[question][(start_ts, end_ts)] = {"answer": answer}
            answer_choices = question_entry["Answer choices"]
            if answer_choices == answer_choices:  # not nan
                question_to_ts_to_answers[question][(start_ts, end_ts)]["answer_choices"] = sorted(eval(answer_choices))
        return question_to_ts_to_answers

    def find_timestamp_answer(self, timestamp: pd.Timestamp, question_to_ts_to_answers: dict):
        # find the answer to the question at the given timestamp
        for ts_range in question_to_ts_to_answers:
            if (timestamp >= ts_range[0] or ts_range[0] != ts_range[0]) and (timestamp < ts_range[1] or ts_range[1] != ts_range[1]):
                return question_to_ts_to_answers[ts_range]
        return None

    def get_num_answer_updates_from_timestamp(self, timestamp: pd.Timestamp, question_to_ts_to_answers: dict):
        # get the number of times the answer to the question has been updated up to the given timestamp
        num_updates = 0
        beginning_of_time = pd.Timestamp("1970-01-01T00:00:00Z", tzinfo=timezone.utc)
        sorted_qa = sorted(question_to_ts_to_answers, key=lambda x: beginning_of_time if x[0] != x[0] else x[0])
        if timestamp < sorted_qa[0][0]:
            return 0
        for ts_range in sorted_qa:
            if (timestamp >= ts_range[0] or ts_range[0] != ts_range[0]) and (timestamp < ts_range[1] or ts_range[1] != ts_range[1]):
                return num_updates
            num_updates += 1
        return num_updates

    def divide_chunks(self, text, n=500):
        # divide text into chunks of n characters
        text = text.split("\n\n")
        paragraphs = []
        for paragraph in text:
            # further subdivide if necessary
            if count_tokens(self.tokenizer, paragraph) > self.context_length - PROMPT_TOKENS:
                sentences = self.nltk_tokenizer.tokenize(paragraph)
                paragraph_chunk = ""
                for sentence in sentences:
                    if count_tokens(self.tokenizer, paragraph_chunk + " " + sentence) >= self.context_length - PROMPT_TOKENS:
                        paragraphs.append(paragraph_chunk)
                        paragraph_chunk = ""
                    else:
                        paragraph_chunk += " " + sentence
                paragraphs.append(paragraph_chunk)
            else:
                paragraphs.append(paragraph)
        chunks = [""]
        for paragraph in paragraphs:
            if count_tokens(self.tokenizer, chunks[-1] + "\n\n" + paragraph) >= n:
                chunks[-1] = chunks[-1].strip()
                # add paragraph to context
                if count_tokens(self.tokenizer, chunks[-1]) > n:
                    chunks = chunks[:-1]
                chunks.append("")
            chunks[-1] += "\n\n" + paragraph
        chunks[-1] = chunks[-1].strip()
        for paragraph in paragraphs:
            assert count_tokens(self.tokenizer, paragraph) < self.context_length - PROMPT_TOKENS
        return chunks

    def parse_dataset(self, task: str, dataset: pd.DataFrame, external_sources_mapping: dict, question_to_ts_to_answers: dict = None, data_id: int = 0):
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
            "sources": [],
            "source_sentences": [],
            "source_timestamps": [],
            "questions": [],
            "actions": [],
            "task": task,
        }

        last_added_source_index = 0
        added_urls = set()

        sorted_source_ts = sorted([
            (url, pd.Timestamp(ts, tzinfo=timezone.utc)) for url in external_sources_mapping for ts in external_sources_mapping[url]
        ], key=lambda x: x[1])
        question_freq = len(sorted_source_ts) // 5  # 5 questions
        question_rate = 0.2  # every 0.2 updates
        last_question_percent = 0.0
        question_increments = []
        percent_of_total_updates = []
        total_updates_overall = 0
        for question in question_to_ts_to_answers:
            total_updates_overall += self.get_num_answer_updates_from_timestamp(sorted_source_ts[-1][1], question_to_ts_to_answers[question])
        ts_to_percent_of_total_updates = {}
        prev_timestamp = sorted_source_ts[0][1]
        for url, timestamp in sorted_source_ts:
            total_updates_at_ts = 0.0
            for question in question_to_ts_to_answers:
                total_updates_at_ts += self.get_num_answer_updates_from_timestamp(timestamp, question_to_ts_to_answers[question])
            ts_to_percent_of_total_updates[timestamp] = total_updates_at_ts / total_updates_overall
            if total_updates_at_ts / total_updates_overall > last_question_percent:
                last_question_percent += question_rate
                question_increments.append(prev_timestamp)
            prev_timestamp = timestamp
        question_increments.append(prev_timestamp)

        question_to_last_answer = {}

        for s_idx, (url, timestamp) in tqdm(enumerate(sorted_source_ts), total=len(sorted_source_ts), desc="Loading data"):
            article = external_sources_mapping[url][timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")]
            if article["archive_url"] in added_urls:
                continue
            added_urls.add(article["archive_url"])
            chunks = self.divide_chunks(article["source_text"], n=int(self.context_length / 2))
            for chunk_idx, chunk in enumerate(chunks):
                data_entry["sources"].append({
                    "url": url,
                    "read_ts": timestamp,  # (/read ts)
                    "context": chunk,
                    "context_sentences": self.extract_article_sentences(chunk),
                    "chunk_idx": chunk_idx,
                    "source_ts": pd.Timestamp(article["source_timestamp"], tzinfo=timezone.utc),  # (/write_ts)
                    "data_id": data_id,
                })
                data_entry["actions"].append({
                    "action_type": "read",
                    "timestamp": timestamp,
                    "sources_idx": len(data_entry["sources"]) - 1,
                    "data_id": data_id,
                })
                last_added_source_index += 1

            if (s_idx == len(sorted_source_ts)-1) or (timestamp in question_increments and sorted_source_ts[s_idx+1][1] != timestamp):
                # last timestamp of question increment
                total_updates_at_ts = 0.0
                for question in question_to_ts_to_answers:
                    total_updates_at_ts += self.get_num_answer_updates_from_timestamp(timestamp, question_to_ts_to_answers[question])
                percent_of_total_updates.append(total_updates_at_ts / total_updates_overall)
                question_to_answer = {question: self.find_timestamp_answer(timestamp, question_to_ts_to_answers[question]) for question in question_to_ts_to_answers}
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
                    answer_at_timestamp = self.find_timestamp_answer(timestamp, question_to_ts_to_answers[question])
                    if question not in questions_to_ask:
                        continue
                    question_to_last_answer[question] = answer_at_timestamp
                    if answer_at_timestamp is not None:
                        answer_choices = answer_at_timestamp.get("answer_choices", list(set([answer_dict["answer"] for answer_dict in question_to_ts_to_answers[question].values()])))
                        if type(answer_choices) == str:
                            answer_choices = ast.literal_eval(answer_choices)
                        if len(answer_choices) == 1:
                            continue
                        answer = answer_at_timestamp["answer"]
                        assert answer in answer_choices
                        data_entry["questions"].append({
                            "question": question,
                            "answer": answer,
                            "answer_choices": sorted(answer_choices),
                            "timestamp": timestamp,
                            "num_updates": self.get_num_answer_updates_from_timestamp(timestamp, question_to_ts_to_answers[question]),
                            "percent_of_total_updates": total_updates_at_ts / total_updates_overall,
                            "edit_type": "question",
                            "question_type": dataset[dataset["Question"] == question]["Question type"].values[0],
                            # all previous articles
                            "context": "\n\n===\n\n".join([source["context"] for source in data_entry["sources"]]),
                            "data_id": data_id,
                        })
                        data_entry["actions"].append({
                            "action_type": "ask",
                            "timestamp": timestamp,
                            "questions_idx": len(data_entry["questions"]) - 1,
                            "data_id": data_id,
                        })
        data_entry["actions"] = sorted(data_entry["actions"], key=lambda x: (x["timestamp"], x["action_type"] == "ask", x["sources_idx"] if x["action_type"] == "read" else x["questions_idx"]))
        data_entry["context_sentences"] = [source["context_sentences"] for source in data_entry["sources"]]
        data_entry["read_ts"] = [source["read_ts"] for source in data_entry["sources"]]
        return data_entry

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
        for task_dir in glob(os.path.join(data_dir, tasks)):
            if os.path.exists(os.path.join(task_dir, "timestamp_to_questions.json")):
                timestamp_to_questions_dict = json.load(open(os.path.join(task_dir, "timestamp_to_questions.json")))
                self.timestamp_to_questions = {
                    pd.Timestamp(key, tzinfo=timezone.utc): value for key, value in timestamp_to_questions_dict.items()
                }
            else:
                self.timestamp_to_questions = {}
            task = task_dir.replace(f"{data_dir}", "").strip("/")
            if "questions.csv" in os.listdir(task_dir):
                # load questions
                questions = pd.read_csv(os.path.join(task_dir, "questions.csv"))
                question_to_ts_to_answers = self.load_questions(questions)
            external_sources_mapping = json.load(open(os.path.join(data_dir, task, "external_sources.json")))

            data_entries.append(self.parse_dataset(task, questions, external_sources_mapping, question_to_ts_to_answers))
            with open(os.path.join(task_dir, "timestamp_to_questions.json"), "w") as f:
                timestamp_to_questions_dict = {
                    key.strftime("%Y-%m-%dT%H:%M:%SZ"): value for key, value in self.timestamp_to_questions.items()
                }
                json.dump(timestamp_to_questions_dict, f, indent=4)
        print("Loaded", len(data_entries), "entries for tasks", tasks)
        random.shuffle(data_entries)
        return data_entries

