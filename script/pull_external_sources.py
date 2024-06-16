from typing import List, Optional, Union
import pandas as pd
import requests
from datetime import datetime
import argparse
from typing import Tuple
from tqdm import tqdm
import backoff
import os
from newspaper import Article
import newspaper
import json
from typing import Dict
import ast
from datetime import datetime as Date
import numpy as np
import sys
from termcolor import colored
# from datetime
import webbrowser
import regex as re
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from openai import OpenAI
import tiktoken


DOWNLOAD_LINK = "https://txtify.it/"
FIREFOX_DRIVER_PATH = "/Users/belindali/geckodriver"

enc = tiktoken.get_encoding("cl100k_base")
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)
openai_cache_file = "cache/gpt-4.jsonl"
openai_cache = {}
if os.path.exists(openai_cache_file):
    with open(openai_cache_file) as f:
        for line in f:
            try:
                openai_cache.update(json.loads(line))
            except:
                breakpoint()

def get_completion_openai(
    prompts: Union[List, str],
    model_name: str = "gpt-4",
    temp: float = 0.7,
    logprobs: bool = False,
    max_tokens: Optional[int] = None,
) -> Tuple[str, float, float]:
    key = json.dumps(prompts)
    if key in openai_cache:
        return openai_cache[key]
    if len(enc.encode(prompts[0]["content"])) > 4000:
        # cutoff
        prompts[0]["content"] = enc.decode(enc.encode(prompts[0]["content"])[:4000])
    completion = client.chat.completions.create(
        model=model_name,
        messages=prompts,
        temperature=temp,
        logprobs=logprobs,
        max_tokens=max_tokens,
    ).dict()
    with open(openai_cache_file, "a") as wf:
        wf.write(json.dumps({key: completion})+"\n")
    openai_cache.update({key: completion})

    return completion


times_after_edit = 0

class RateLimitException(Exception):
    """Exception raised when a 429 Rate Limit Exceeded status code is received."""
    pass

def in_timestamp_range(timestamp: datetime.timestamp, timestamp_range: List[str]) -> bool:
    after_start = timestamp_range[0] != timestamp_range[0] or timestamp >= convert_timestamp(timestamp_range[0])
    before_end = timestamp_range[1] != timestamp_range[1] or timestamp < convert_timestamp(timestamp_range[1]) + pd.Timedelta("1 day")
    if timestamp_range[0] == timestamp_range[0] and convert_timestamp(timestamp_range[0]) > Date.now():
        # starts in the future
        after_start = timestamp > convert_timestamp("2024-02-28T00:00:00Z")

    return after_start and before_end

# TODO: I'm getting the rate limit way more than expected. They say it's 5 requests/min, but I repeatedly hit the limit. Need to investigate
@backoff.on_exception(backoff.expo, (requests.exceptions.RequestException, requests.exceptions.HTTPError), max_tries=10, factor=5)
def get_closest_archived_url(url: str, timestamp: str, timestamp_range: List[str] = None) -> Tuple[str, str]:
    if url is None or url == "":
        return None, None
    return url, timestamp


def convert_timestamp(timestamp: str) -> str:
    if timestamp.count("/") == 2:
        return datetime.strptime(timestamp, "%m/%d/%Y")
    if timestamp.count("/") == 1:
        return datetime.strptime(timestamp, "%m/%Y")
    try:
        return datetime.strptime(timestamp, "%Y%m%d%H%M%S")
    except:
        if "T" in timestamp:
            return datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%SZ')
        else:
            return datetime.strptime(timestamp, '%Y-%m-%d')


@backoff.on_exception(backoff.constant, (requests.exceptions.RequestException, requests.exceptions.HTTPError, requests.exceptions.ConnectionError, newspaper.article.ArticleException), interval=5, max_tries=25)
def extract_main_content(url: str, article_download: bool=False, driver=None) -> str:
    if url is None:
        return None
    try:
        article = Article(url)
        article.download()
        article.parse()
        source_text = article.title + "\n\n" + article.text

    except RateLimitException:
        print("Rate limited, sleeping for 5s")
        raise
    except requests.exceptions.HTTPError as e:
        print(f"An HTTP error occurred: {e}, sleeping for 5s")
        raise
    except requests.exceptions.ConnectionError as e:
        print(f"A connection error occurred: {e}, sleeping for 5s")
        raise
    except newspaper.article.ArticleException as e:
        print(f"An article error occurred: {e}")
        source_text = ""
    except Exception as e:
        print(f"An error occurred: {e}")
        source_text = ""
    if len(source_text.strip()) == 0:
        source_text = extract_main_content_selenium(url, driver)
    return source_text

def extract_main_content_selenium(url: str, driver) -> str:
    driver.set_page_load_timeout(10)
    driver.get(f"{DOWNLOAD_LINK}/{url}")
    print(f"Got {url}")
    # tag is "pre"
    try:
        element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, 'pre'))
        )
    except selenium.common.exceptions.TimeoutException:
        print(f"Timed out for {url}")
        # try again
        element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, 'pre'))
        )
    print(f"Got element {element}")
    source_text = element.text
    print(f"Got source text {source_text}")
    if times_after_edit > 5 or len(source_text.strip()) == 0:
        source_text = extract_main_content_manual(url)
    return source_text

def extract_main_content_manual(url: str) -> str:
    print(f"{DOWNLOAD_LINK}/{url}")
    webbrowser.open(f"{DOWNLOAD_LINK}/{url}")
    print("Enter article text (press Ctrl-D to save):")
    article = sys.stdin.read()
    return article

def match_text(span, source_text):
    # Replace spaces, newlines, and tabs with regex that matches any number of whitespaces
    try:
        span = re.sub(r'\s+', ' ', span)
        source_text = re.sub(r'\s+', ' ', source_text)
    except:
        breakpoint()
    # return span.lower() in source_text.lower()
    return span.lower().replace(' ', '').replace("\n", "") == source_text.lower().replace(" ", "").replace("\n", "")

def has_fact(row, all_rows, source_text):
    user_prompt = """Please read the article below and verify whether these conditions are met:\n1. The article must imply the fact: {fact}\n2. The article must NOT imply the fact: {next_fact}\n3. The article must NOT state that {fact} is about to become false\n4. The article must be in English.\n\nArticle text (written at {source_date}):\n{article_text}\n\n*===*\n\nPlease state 'valid' if the article meets these conditions, otherwise state 'invalid'."""
    
    # TODO get the next fact
    next_triple = all_rows[(
        all_rows["subjectLabel"] == row["subjectLabel"]
    ) & (
        all_rows["propertyLabel"] == row["propertyLabel"]
    ) & (
        all_rows["objectLabel"] != row["objectLabel"]
    ) & (
        all_rows["startDate"] > row["startDate"]
    )]
    if next_triple.shape[0] > 0:
        next_triple = next_triple[next_triple["startDate"] == next_triple["startDate"].min()]
        next_triple = tuple(next_triple[["subjectLabel", "propertyLabel", "objectLabel"]].iloc[0].to_list())
        next_fact_str  = f"{next_triple[0]} - {next_triple[1]} - {next_triple[2]}"
    else:
        next_triple = None
        next_fact_str = ""
    fact = (row['subjectLabel'], row['propertyLabel'], row['objectLabel'], row['startDate'])
    source_date = row['source_date']
    fact_str = f"{fact[0]} - {fact[1]} - {fact[2]}"

    openai_response = get_completion_openai(
        [
            {"role": "user", "content": user_prompt.format(fact=fact_str, next_fact=next_fact_str, source_date=source_date, article_text=source_text)},
        ],
        model_name="gpt-4",
        temp=0.0,
        logprobs=False,
    )

    try:
        openai_pred = "invalid" not in openai_response['choices'][0]['message']['content'].lower()
    except:
        breakpoint()
    return fact, openai_pred

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--url_file", required=True)
    # parser.add_argument("--timestamp_file", required=True)
    parser.add_argument("--edits_file", required=True)
    parser.add_argument("--output_dir", default="wikipedia-data/")
    # parser.add_argument("--url_key", default="referenceURL")  # url
    parser.add_argument("--url_key", default="selected_link")  # url
    # parser.add_argument("--ts_start_key", default="startDate")  # start
    parser.add_argument("--ts_start_key", default="source_date")  # start
    parser.add_argument("--ts_end_key", default="endDate") # end
    args = parser.parse_args()

    driver = webdriver.Firefox()  #executable_path=FIREFOX_DRIVER_PATH)

    os.makedirs(args.output_dir, exist_ok=True)
    # urls_file = os.path.join(args.output_dir, "urls.jsonl")
    archive_urls_file = os.path.join(args.output_dir, "url_timestamp_to_archive_url.jsonl")
    sources_cache_file = os.path.join(args.output_dir, "external_sources_cache.jsonl")
    outputs_file = os.path.join(args.output_dir, "external_sources.json")

    # (url, timestamp) -> (archived_url (str), content (str))
    url_timestamp_to_archive_url = {}
    archived_url_to_content = {}
    

    all_edits = pd.read_csv(args.edits_file)

    if os.path.exists(archive_urls_file):
        with open(archive_urls_file, "r") as f:
            for line in f.readlines():
                try:
                    data = json.loads(line)
                except:
                    breakpoint()
                # get existing cached data
                url_timestamp_to_archive_url[(data["url"], data["timestamp"])] = {
                    "archive_url": data["archive_url"],
                    "source_timestamp": data["source_timestamp"],
                }
    if os.path.exists(sources_cache_file):
        with open(sources_cache_file, "r") as f:
            for line in f.readlines():
                try:
                    data = json.loads(line)
                except:
                    breakpoint()
                # keeping these separate as sometimes multiple (url, timestamp) correspond to sample archive_url
                archived_url_to_content[data["archive_url"]] = {
                    "source_text": data["source_text"],
                    "corrected": data.get("corrected", "n"),
                }

    print("Getting closest versions of urls and corresponding website texts...")
    outputs = {}
    for i, edit in tqdm(all_edits.iterrows(), total=len(all_edits)):
        if edit[args.url_key] != edit[args.url_key] or edit[args.url_key].strip("-") == "":
            continue
        # edit_urls, edit_timestamp = json.loads(edit[args.url_key]), edit[args.ts_key]
        try:
            edit_urls = ast.literal_eval(edit[args.url_key].replace("\n", ","))
        except:
            edit_urls = [edit[args.url_key]]
        if edit[args.ts_start_key] == edit[args.ts_start_key] and edit[args.ts_start_key].startswith("http://www.wikidata.org/.well-known"):
            edit[args.ts_start_key] = np.nan
        if edit[args.ts_end_key] == edit[args.ts_end_key] and edit[args.ts_end_key].startswith("http://www.wikidata.org/.well-known"):
            edit[args.ts_end_key] = np.nan
        timestamp_range = [edit[args.ts_start_key], edit[args.ts_end_key]]
        edit_timestamp = edit[args.ts_start_key] if edit[args.ts_start_key] == edit[args.ts_start_key] else "1970-01-01T00:00:00Z"
        if edit[args.ts_start_key] != edit[args.ts_start_key]:
            print(edit["selected_link"])
            # breakpoint()
        edit_timestamp = convert_timestamp(edit_timestamp).strftime("%Y-%m-%dT%H:%M:%SZ")
        # print(edit_timestamp)
        filtered_urls = []
        filtered_archive_urls = []
        filtered_source_timestamps = []
        filtered_contents = []
        for url in edit_urls:
            if not (url, edit_timestamp) in url_timestamp_to_archive_url:
                archive_url, source_timestamp = get_closest_archived_url(url, edit_timestamp, timestamp_range=timestamp_range)
                with open(archive_urls_file, "a") as wf:
                    wf.write(json.dumps({
                        "url": url, "timestamp": edit_timestamp,
                        "archive_url": archive_url, "source_timestamp": source_timestamp,
                    }) + "\n")
            else:
                archive_url = url_timestamp_to_archive_url[(url, edit_timestamp)]["archive_url"]
                source_timestamp = url_timestamp_to_archive_url[(url, edit_timestamp)]["source_timestamp"]
                # if source_timestamp is None and not in_timestamp_range(Date.now(), timestamp_range):
                #     print(source_timestamp, timestamp_range, url)
                #     # breakpoint()
                # elif source_timestamp is not None and not in_timestamp_range(convert_timestamp(source_timestamp), timestamp_range):
                #     print(source_timestamp, timestamp_range, url)
                #     # breakpoint()
            if archive_url is None:
                continue
            if archive_url not in archived_url_to_content:
                source_text = extract_main_content(archive_url, driver=driver)
                archived_url_to_content[archive_url] = {
                    "source_text": source_text,
                }
                with open(sources_cache_file, "a") as wf:
                    wf.write(json.dumps({
                        "archive_url": archive_url,
                        "source_text": archived_url_to_content[archive_url]["source_text"],
                    }) + "\n")
            else:
                source_text = archived_url_to_content[archive_url]["source_text"]
            # flag potentially problematic sources
            if source_text is None:
                print(f"None source text: {archive_url}")
                source_text = extract_main_content(archive_url, driver=driver)
            elif len(source_text) < 100:
                print(f"Short source text: {archive_url}, {source_text}")
                source_text = extract_main_content(archive_url, driver=driver)
                with open(sources_cache_file, "a") as wf:
                    wf.write(json.dumps({
                        "archive_url": archive_url,
                        "source_text": source_text,
                    }) + "\n")
            span = all_edits.loc[i, "span"]
            if span is not None and span == span:
                # placeholder_span = span.strip().replace(r'\s+', "<SPACE>")
                # escaped_span = re.escape(placeholder_span)
                # span_re = escaped_span.replace('<SPACE>', r'\\s*')
                # pattern = re.compile(f'({escaped_span})', re.UNICODE)
                # if not match_text(span, source_text) and (archived_url_to_content[archive_url].get("corrected", "n") != "y"):
                fact, has_fact_gpt = has_fact(edit, all_edits, source_text)
                if not has_fact_gpt and (archived_url_to_content[archive_url].get("corrected", "n") != "y"):
                    """
                    """
                    print("*==*")
                    print(f"{colored(source_text, 'red')}\n\n*==*\n\nFact\n{colored(fact, 'green')}\nnot in above source text of {archive_url}")
                    print("Is source text correct? (y/n)")
                    webbrowser.open(f"{DOWNLOAD_LINK}/{archive_url}")
                    correct = input()
                    if correct == "n":

                        source_text = extract_main_content_manual(archive_url)
                        with open(sources_cache_file, "a") as wf:
                            wf.write(json.dumps({
                                "archive_url": archive_url,
                                "source_text": source_text,
                                "corrected": "y",
                            }) + "\n")
                    else:
                        # # confirmed correct, mark as such
                        # print("Marking as correct")
                        with open(sources_cache_file, "a") as wf:
                            wf.write(json.dumps({
                                "archive_url": archive_url,
                                "source_text": source_text,
                                "corrected": "y",
                            }) + "\n")
                    # """

            if url not in outputs:
                outputs[url] = {}
            outputs[url][edit_timestamp] = {
                "archive_url": archive_url,
                "source_timestamp": source_timestamp,
                "source_text": source_text,
            }
        if len(filtered_urls) == 0:
            continue
    #     # append to df (note not necessarily at i as some entries are skipped)
    #     df.loc[len(df.index)] = {
    #         "edit_id": len(df),  # int
    #         "edit": all_edits[i],  # str
    #         "url": filtered_urls,   # list[str]
    #         "timestamp": edit_timestamp,   # time
    #         "archive_url": filtered_archive_urls,  # list[str]
    #         "source_timestamp": filtered_source_timestamps,  # list[time]
    #         "source_text": filtered_contents,  # list[str]
    #     }

    # df.to_csv(outputs_file, index=False)
    with open(outputs_file, "w") as wf:
        json.dump(outputs, wf, indent=4)

    # # split into train and test
    # df_train = all_edits.sample(frac=0.2)
    # df_test = all_edits.drop(df_train.index)
    # df_train.to_csv(os.path.join(args.output_dir, "train.csv"), index=False)
    # df_test.to_csv(os.path.join(args.output_dir, "test.csv"), index=False)

    driver.quit()