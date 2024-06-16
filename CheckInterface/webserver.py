import copy
from flask import Flask, jsonify, redirect, render_template, request, url_for
import json
import os
import pandas as pd
import random
from datetime import datetime, timedelta
import re
import time


subset = ""
orig_file = f"notebooks/annotation_splits/property_to_results_larger{subset}_subset_links_filtered.csv"
source_file = f"AnnotationInterface/results/property_to_results_larger{subset}_subset_links_filtered.csv"
target_file = f"CheckInterface/results/property_to_results_larger{subset}_subset_links.csv"
ORIGIN_DATE = datetime.strptime("2024-04-11", "%Y-%m-%d")

if subset == "":
    URL_TO_TEXT_FILE = "wikidata-data/subset/external_sources.json"
else:
    URL_TO_TEXT_FILE = f"wikidata-data/{subset.strip('_')}/external_sources.json"
URL_TO_TEXT_FILE = f"wikidata-data/full/external_sources.json"

with open(URL_TO_TEXT_FILE) as f:
    URL_TO_TEXT = json.load(f)

app = Flask(__name__)


os.makedirs("CheckInterface/users", exist_ok=True)
os.makedirs("CheckInterface/results", exist_ok=True)
# triple_to_user_file = "CheckInterface/users/triple_to_user.json"

# user_to_triple = {}
# user_to_unannotated_triple = {}  # current set of annotations for user
# NUM_QUESTIONS = 8
MAX_ANNOTATIONS_AT_A_TIME = 1
MIN_ANNOTATIONS = 4
NUM_MINUTES = 12

user_to_triple = {}
user_to_triple_file = "CheckInterface/users/user_to_triple.json"
if os.path.exists(user_to_triple_file):
    with open(user_to_triple_file, 'r') as f:
        user_to_triple = json.load(f)
        for user in user_to_triple:
            user_to_triple[user] = [tuple(triple) for triple in user_to_triple[user]]

annotated_triples = []
annotated_triples_file = "CheckInterface/users/annotated_triples.json"
if os.path.exists(annotated_triples_file):
    with open(annotated_triples_file, 'r') as f:
        annotated_triples = json.load(f)
        annotated_triples = [tuple(triple) for triple in annotated_triples]


data = pd.read_csv(source_file)
# unique-ify data
data = data.drop_duplicates(subset=["subjectLabel", "propertyLabel", "objectLabel", "startDate"])
orig_data = pd.read_csv(orig_file)
all_triples = []
triples_and_sources = []
# [(subj, relation, obj) -> {"sources": [sources], "next_triple": (subj, relation, obj2)}]
for i, row in data.iterrows():
    if row["objectLabel"] != row["objectLabel"]:
        continue
    next_triple = data[(
        data["subjectLabel"] == row["subjectLabel"]
    ) & (
        data["propertyLabel"] == row["propertyLabel"]
    ) & (
        data["objectLabel"] != row["objectLabel"]
    ) & (
        data["startDate"] > row["startDate"]
    )]
    if next_triple.shape[0] > 0:
        next_triple = next_triple[next_triple["startDate"] == next_triple["startDate"].min()]
        next_triple = tuple(next_triple[["subjectLabel", "propertyLabel", "objectLabel"]].iloc[0].to_list())
    else:
        next_triple = None
    triple = tuple(row[["subjectLabel", "propertyLabel", "objectLabel", "startDate"]].to_list())
    if row["selected_link"] != row["selected_link"]:
        continue
    source_link = row["selected_link"]
    source_date = row["source_date"]
    if "T" in source_date:
        source_date = datetime.strptime(source_date, "%Y-%m-%dT%H:%M:%SZ")
    elif source_date.count("/") == 2:
        source_date = datetime.strptime(source_date, "%m/%d/%Y")
    elif source_date.count("/") == 1:
        source_date = datetime.strptime(source_date, "%m/%Y")
    else:
        source_date = datetime.strptime(source_date, "%Y-%m-%d")
    try:
        source_text = URL_TO_TEXT[source_link][source_date.strftime("%Y-%m-%dT%H:%M:%SZ")]["source_text"]
    except:
        print(source_link, source_date, source_date.strftime("%Y-%m-%dT%H:%M:%SZ"))
        continue
    source = {"link": row["selected_link"], "date": row["source_date"], "text": source_text}
    start_date = None
    if row["startDate"] == row["startDate"] and not row["startDate"].startswith("http://www.wikidata.org/.well-known"):
        start_date = datetime.strptime(row["startDate"], "%Y-%m-%dT%H:%M:%SZ").strftime("%m/%Y")
    end_date = None
    if row["endDate"] == row["endDate"] and not row["endDate"].startswith("http://www.wikidata.org/.well-known"):
        end_date = datetime.strptime(row["endDate"], "%Y-%m-%dT%H:%M:%SZ").strftime("%m/%Y")
    searchURL = orig_data[(
        orig_data["subjectLabel"] == row["subjectLabel"]
    ) & (
        orig_data["propertyLabel"] == row["propertyLabel"]
    ) & (
        orig_data["objectLabel"] == row["objectLabel"]
    ) & (
        orig_data["startDate"] == row["startDate"]
    )]["searchURL"].iloc[0]
    triples_and_sources.append({
        "triple": triple,
        "startDate": start_date,
        "endDate": end_date,
        "source": source,
        "source_text": source_text,
        "google_link": searchURL,
        "next_triple": next_triple,
    })
    if triple not in all_triples:
        all_triples.append(triple)

user_to_curr_annotating_triple_and_sources = {user: [
    triple for triple in triples_and_sources if triple['triple'] in user_to_triple[user] and triple['triple'] not in annotated_triples  # currently annotated triples
] for user in user_to_triple}
annotating_triples = []
for user in user_to_curr_annotating_triple_and_sources:
    for triple in user_to_curr_annotating_triple_and_sources[user]:
        annotating_triples.append(triple['triple'])
# user_to_unannotated_triple = {user: [triple for triple in triples_and_sources if triple not in annotated_triples] for user, triples in user_to_triple.items()}
user_start_time = {}

def distribute_triples(username, num_annotations=MAX_ANNOTATIONS_AT_A_TIME):
    # distribute new
    unannotated_triples = [triple for triple in all_triples if triple not in annotated_triples]  # hasn't been annotated
    not_annotating_triples = [triple for triple in all_triples if triple not in annotating_triples and triple not in annotated_triples]  # not currently being annotated

    user_triples = [triple['triple'] for triple in user_to_curr_annotating_triple_and_sources[username]]
    if len(not_annotating_triples) > 0:
        i = 0
        new_triples = []
        while len(user_triples) < num_annotations and i < len(not_annotating_triples):
            user_triples.append(not_annotating_triples[i])
            new_triples.append(not_annotating_triples[i])
            i += 1
        user_to_triple[username].extend(new_triples)
        user_to_curr_annotating_triple_and_sources[username].extend([
            triple for triple in triples_and_sources if triple['triple'] in new_triples
        ])
        annotating_triples.extend(new_triples)
        json.dump(user_to_triple, open(user_to_triple_file, 'w'), indent=4)
    elif len(unannotated_triples) > 0:
        i = 0
        new_triples = []
        # sort by fewest count in annotating_triples, preserving order otherwise
        annotating_triples_count = {triple: annotating_triples.count(triple) for triple in annotating_triples}
        unannotated_triples = sorted(unannotated_triples, key=lambda x: annotating_triples_count.get(x, 0))
        while len(user_triples) < num_annotations and i < len(unannotated_triples):
            if unannotated_triples[i] not in user_triples:
                # don't present the same triple to the same user
                user_triples.append(unannotated_triples[i])
                new_triples.append(unannotated_triples[i])
            i += 1
        user_to_triple[username].extend(new_triples)
        user_to_curr_annotating_triple_and_sources[username].extend([
            triple for triple in triples_and_sources if triple['triple'] in new_triples
        ])
        annotating_triples.extend(new_triples)
        json.dump(user_to_triple, open(user_to_triple_file, 'w'), indent=4)
    else:
        return 'No more unannotated facts available.'

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        user_start_time[username] = time.time()
        if username not in user_to_triple:
            user_to_triple[username] = []
            user_to_curr_annotating_triple_and_sources[username] = []
        return redirect(url_for('next_question', username=username))
    else:
        return render_template('login.html')


@app.route('/next_question', methods=['GET', 'POST'])
def next_question():
    if request.method == 'GET':
        username = request.args.get('username')
        distribute_triples(username, num_annotations=MIN_ANNOTATIONS)
        return render_template('questions.html', data=user_to_curr_annotating_triple_and_sources.get(username, []), username=username, time_left=NUM_MINUTES*60)
    else:
        if os.path.exists(target_file):
            annotated_data = pd.read_csv(target_file)
        else:
            annotated_data = pd.DataFrame(columns=["subjectLabel", "propertyLabel", "objectLabel", "startDate", "endDate", "selected_link", "source_date", "span", "annotator"])

        # append to annotations
        annotations = request.get_json()
        username = annotations.get('username')
        triples = []
        for question in annotations['responses']:
            triple_index = int(question['questionId'].split('_')[-1])
            try:
                fact_triple = user_to_curr_annotating_triple_and_sources[username][triple_index]['triple']
            except:
                print(len(user_to_curr_annotating_triple_and_sources[username]), triple_index)
            if fact_triple in annotating_triples:
                annotating_triples.remove(fact_triple)
                annotated_triples.append(fact_triple)
            source = user_to_curr_annotating_triple_and_sources[username][triple_index]['source']
            if question['selectedOption'] == "source_alt":
                selected_link = question["link"]
                source_date = question["date"]
            elif question['selectedOption'] == "source_cannot_find":
                selected_link = None
                source_date = None
            else:
                selected_link = source["link"]
                source_date = source["date"]
            span = question.get("span")
            notes = question.get("notes")
            # source_date = question.get("date")
            triples.append({
                'subject': fact_triple[0],
                'relation': fact_triple[1],
                'object': fact_triple[2],
                'startDate': user_to_curr_annotating_triple_and_sources[username][triple_index]['startDate'],
                'selected_link': selected_link,
                'source_date': source_date,
                'span': span,
                'notes': notes,
            })
            data_row = data[(
                data["subjectLabel"] == fact_triple[0]
            ) & (
                data["propertyLabel"] == fact_triple[1]
            ) & (
                data["objectLabel"] == fact_triple[2]
            )]
            if data_row.shape[0] > 1:
                data_row = data_row[(
                    data_row["startDate"].apply(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ").strftime("%m/%Y")) == user_to_curr_annotating_triple_and_sources[username][triple_index]['startDate']
                )]
            annotated_row = {"subjectLabel": data_row["subjectLabel"], "propertyLabel": data_row["propertyLabel"], "objectLabel": data_row["objectLabel"]}
            annotated_row["startDate"] = data_row["startDate"]
            annotated_row["endDate"] = data_row["endDate"]
            annotated_row["selected_link"] = selected_link
            annotated_row["source_date"] = source_date
            annotated_row["span"] = span
            annotated_row["notes"] = notes
            annotated_row["annotator"] = username
            annotated_row = pd.DataFrame(annotated_row)
            annotated_data = pd.concat([annotated_data, annotated_row])
        
        annotated_data.to_csv(target_file, index=False)
        json.dump(annotated_triples, open(annotated_triples_file, 'w'), indent=4)

        user_to_curr_annotating_triple_and_sources[username] = []
        # BAN PEOPLE SPAMMING NO SOURCE BUTTON
        user_annotations = annotated_data[annotated_data["annotator"] == username]
        if (user_annotations["selected_link"].isnull().sum() >= 10):
            return jsonify({
                'redirect': url_for('feedback', username=username),
            })
        if time.time() - user_start_time[username] > NUM_MINUTES*60:
            return jsonify({
                'redirect': url_for('feedback', username=username),
            })
        else:
            distribute_triples(username)
            if len(user_to_curr_annotating_triple_and_sources.get(username, [])) == 0:
                return jsonify({
                    'redirect': url_for('feedback', username=username),
                })
            return jsonify({
                'data': user_to_curr_annotating_triple_and_sources.get(username, []),
                'time_left': NUM_MINUTES*60 - (time.time() - user_start_time[username]),
                'start_time': user_start_time[username]
            })
            
        # continue annotating


@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'GET':
        username = request.args.get('username')
        return render_template('feedback.html', username=username)
    else:
        # append to annotations
        annotations = request.get_json()
        username = annotations.get('username')
        feedback = annotations.get('feedback')
        if not os.path.exists(f"CheckInterface/results/feedback.jsonl"):
            with open("CheckInterface/results/feedback.jsonl", "w") as f:
                pass
        with open("CheckInterface/results/feedback.jsonl", "a") as f:
            f.write(json.dumps({
                "username": username,
                "feedback": feedback
            }) + "\n")
        user_to_curr_annotating_triple_and_sources[username] = []
        # SUCCESS
        return jsonify({'message': 'Form submitted successfully.'})


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
