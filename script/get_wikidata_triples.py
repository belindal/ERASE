import requests
import json
from tqdm import tqdm
import os
import pandas as pd

# Define the URL of the Wikibase SPARQL endpoint
url = 'https://query.wikidata.org/sparql'

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
prop_to_subjs_file = f"{DATA_DIR}/prop_to_subj.json"
property_to_results_file = f"{DATA_DIR}/property_to_results.json"
dataframe_file = f"{DATA_DIR}/property_to_results.csv"
query_to_results_cache_file = f"{DATA_DIR}/query_to_results_cache.json"

properties = [
    "P169", "P35", "P488",
    "P39",
    "P54", "P102", "P26", "P451", "P551", "P108",
    "P159",
    "P263",
    "P463",
    "P607",
    "P647",
    "P664",
    "P822",
    "P1307",
    "P1448",
]
properties_nl = {
    "P169": "chief executive officer",
    "P35": "head of state",
    "P488": "chairperson",
    "P39": "position held",
    "P54": "member of sports team",
    "P102": "member of political party",
    "P26": "spouse",
    "P451": "unmarried partner",
    "P551": "residence",
    "P108": "employer",
    "P159": "headquarters location",
    "P263": "official residence",
    "P286": "head coach",
    "P463": "member of",
    "P607": "conflict",
    "P647": "drafted by",
    "P664": "organizer",
    "P822": "mascot",
    "P1307": "director/manager",
    "P1308": "relative",
    "P1448": "official name",
}

subsets = {
    "P169": [
        "?object wdt:P27 wd:Q30.",
        "?object wdt:P27 wd:Q16.",
        "?object wdt:P27 wd:Q145.",
        "?object wdt:P27 wd:Q408.",
    ],
    "P35": [
        "?object wdt:P27 wd:Q30.",
        "?object wdt:P27 wd:Q16.",
        "?object wdt:P27 wd:Q145.",
        "?object wdt:P27 wd:Q408.",
    ],
    "P488": [
        "?object wdt:P27 wd:Q30.",
        "?object wdt:P27 wd:Q16.",
        "?object wdt:P27 wd:Q145.",
        "?object wdt:P27 wd:Q408.",
    ],
    "P39": [
        "?subject wdt:P27 wd:Q30.",
        "?subject wdt:P27 wd:Q16.",
        "?subject wdt:P27 wd:Q145.",
        "?subject wdt:P27 wd:Q408.",
    ],
    "P54": [
        "?subject wdt:P27 wd:Q30.",
        "?subject wdt:P27 wd:Q16.",
        "?subject wdt:P27 wd:Q145.",
        "?subject wdt:P27 wd:Q408.",
    ],
    "P102": [
        "?subject wdt:P27 wd:Q30.",
        "?subject wdt:P27 wd:Q16.",
        "?subject wdt:P27 wd:Q145.",
        "?subject wdt:P27 wd:Q408.",
    ],
    "P26": [
        "?subject wdt:P27 wd:Q30.",
        "?subject wdt:P27 wd:Q16.",
        "?subject wdt:P27 wd:Q145.",
        "?subject wdt:P27 wd:Q408.",
    ],
    "P451": [
        "?subject wdt:P27 wd:Q30.",
        "?subject wdt:P27 wd:Q16.",
        "?subject wdt:P27 wd:Q145.",
        "?subject wdt:P27 wd:Q408.",
    ],
    "P551": [
        "?subject wdt:P27 wd:Q30.",
        "?subject wdt:P27 wd:Q16.",
        "?subject wdt:P27 wd:Q145.",
        "?subject wdt:P27 wd:Q408.",
    ],
    "P108": [
        "?subject wdt:P27 wd:Q30.",
        "?subject wdt:P27 wd:Q16.",
        "?subject wdt:P27 wd:Q145.",
        "?subject wdt:P27 wd:Q408.",
    ],
    "P159": [
        "?object wdt:P17 wd:Q30.",
        "?object wdt:P17 wd:Q16.",
        "?object wdt:P17 wd:Q145.",
        "?object wdt:P17 wd:Q408.",
    ],
    "P263": [""],
    "P286": [
        "?object wdt:P27 wd:Q30.",
        "?object wdt:P27 wd:Q16.",
        "?object wdt:P27 wd:Q145.",
        "?object wdt:P27 wd:Q408.",
    ],
    "P463": [
        "?subject wdt:P27 wd:Q30.",
        "?subject wdt:P27 wd:Q16.",
        "?subject wdt:P27 wd:Q145.",
        "?subject wdt:P27 wd:Q408.",
    ],
    "P607": [
        "?subject wdt:P27 wd:Q30.",
        "?subject wdt:P27 wd:Q16.",
        "?subject wdt:P27 wd:Q145.",
        "?subject wdt:P27 wd:Q408.",
    ],
    "P647": [
        "?subject wdt:P27 wd:Q30.",
        "?subject wdt:P27 wd:Q16.",
        "?subject wdt:P27 wd:Q145.",
        "?subject wdt:P27 wd:Q408.",
    ],
    "P664": [""],
    "P822": [""],
    "P1307": [
        "?object wdt:P17 wd:Q30.",
        "?object wdt:P17 wd:Q16.",
        "?object wdt:P17 wd:Q145.",
        "?object wdt:P17 wd:Q408.",
    ],
    "P1448": [
        "?subject wdt:P17 wd:Q30.",
        "?subject wdt:P17 wd:Q16.",
        "?subject wdt:P17 wd:Q145.",
        "?subject wdt:P17 wd:Q408.",
        "?subject wdt:P27 wd:Q30.",
        "?subject wdt:P27 wd:Q16.",
        "?subject wdt:P27 wd:Q145.",
        "?subject wdt:P27 wd:Q408.",
     ],
}


import re
from datetime import datetime, timezone
def datetime_transform(x):
    if x != x or "http" in x:
        return pd.to_datetime("2024-03-13T00:00:00Z")
    match = re.match(r'(\d{8})-01-01T00:00:00Z', x)
    if match:
        return pd.to_datetime(match.group(1), format='%Y%m%d', utc=True)
    match = re.match(r'(\d{6})-01-01T00:00:00Z', x)
    if match:
        return pd.to_datetime(match.group(1) + "01", format='%Y%m%d', utc=True)
    try:
        return pd.to_datetime(x)
    except:
        print(x)


if os.path.exists(prop_to_subjs_file):
    property_to_subjects = json.load(open(prop_to_subjs_file))
else:
    property_to_subjects = {
        prop: [] for prop in properties}

    for prop in properties:
        print(prop)
        if len(property_to_subjects[prop]) > 0:
            continue
        for filter in tqdm(subsets[prop]):
            query = f"""
SELECT ?subject ?subjectLabel
WITH {{
  SELECT ?subject ?subjectLabel
  WHERE
  {{
    SELECT ?subject (COUNT(DISTINCT(?object)) AS ?objectCount) (MAX(?startDate) AS ?lastReplacementDate)
    WHERE {{
      ?subject p:{prop} ?statement.
      ?statement ps:{prop} ?object.
      {filter}
      OPTIONAL {{ ?statement pq:P580 ?startDate. hint:Prior hint:rangeSafe true. }}
    }}
    GROUP BY ?subject
    HAVING (?objectCount > 1 && ?lastReplacementDate > "2021-11-01T00:00:00Z"^^xsd:dateTime)
  }}
  LIMIT 1000
}} AS %i
WHERE {{
  INCLUDE %i
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" . }}
}}
"""
            # Send the HTTP request and get the response
            response = requests.get(url, params={'format': 'json', 'query': query})
            try:
                # Parse the response as JSON
                data = response.json()
            except:
                print(query)
                print(response.text)
        
            for item in data['results']['bindings']:
                property_to_subjects[prop].append(item)

    for property in property_to_subjects:
        print(property)
        print([subj['subjectLabel']['value'] for subj in property_to_subjects[property]])

    json.dump(property_to_subjects, open(prop_to_subjs_file, "w"), indent=4)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# get results incrementally
query_to_results_cache = json.load(open(query_to_results_cache_file))
print(len(query_to_results_cache))

if os.path.exists(property_to_results_file):
    property_to_results = json.load(open(property_to_results_file))
else:
    property_to_results = {}
    max_subjects = 100
    for prop in property_to_subjects:
        print(prop)
        if prop not in property_to_results:
            property_to_results[prop] = []
        for result in property_to_results[prop]:
            if type(result['subject']) == dict:
                result['subject'] = result['subject']['value']
        existing_subjects = {result['subject'].split("/")[-1] for result in property_to_results[prop]}
        all_subjects = {subj["subject"]["value"].split("/")[-1] for subj in property_to_subjects[prop]}
        remaining_subjects = all_subjects - existing_subjects
        subjects_list = []
        # divide into units of up to max_subjects
        for subj_id in remaining_subjects:
            subjects_list.append(f"wd:{subj_id}")
        print("Excess subjects:", existing_subjects - all_subjects)
        new_results = []
        for result in property_to_results[prop]:
            if result['subject'].split("/")[-1] in all_subjects:
                new_results.append(result)
        property_to_results[prop] = new_results
        if len(subjects_list) == 0:
            continue
        for subject_chunk in tqdm(chunks(subjects_list, max_subjects)):
            subjects_chunk_nl = " ".join(subject_chunk)
            query = f"""SELECT ?subject ?subjectLabel ?object ?objectLabel ?startDate ?endDate ?of ?ofLabel ?referenceURL
WITH {{
    SELECT ?subject ?subjectLabel ?object ?objectLabel ?startDate ?endDate ?of ?ofLabel ?referenceURL
    WHERE {{
      VALUES ?subject {{ {subjects_chunk_nl} }}
      ?subject p:{prop} ?statement.
      ?statement ps:{prop} ?object.
      OPTIONAL {{ ?statement pq:P31 ?of. }}
      OPTIONAL {{ ?statement pq:P276 ?of. }}
      OPTIONAL {{ ?statement pq:P108 ?of. }}
      OPTIONAL {{ ?statement pq:P708 ?of. }}
      OPTIONAL {{ ?statement pq:P642 ?of. }}
      OPTIONAL {{ ?statement pq:P580 ?startDate. }}
      OPTIONAL {{ ?statement pq:P582 ?endDate. }}
      OPTIONAL {{ ?statement prov:wasDerivedFrom ?referenceNode. 
                 ?referenceNode pr:P854 ?referenceURL. }}
    }}
    LIMIT 1000
}} AS %i
WHERE {{
  INCLUDE %i
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" . }}
}}"""
            # Send the HTTP request and get the response
            if query in query_to_results_cache:
                data = {'results': {'bindings': query_to_results_cache[query]}}
                print("Cache hit")
            else:
                print(query)
                print(json.dumps(query))    
            for item in data['results']['bindings']:
                property_to_results[prop].append(item)

    with open(query_to_results_cache_file, "w") as wf:
        json.dump(query_to_results_cache, wf, indent=4)

    json.dump(property_to_results, open(property_to_results_file, "w"))

# linearize
property_to_results_entries = []
for prop in property_to_results:
    for entry in property_to_results[prop]:
        for k in entry:
            if type(entry[k]) == dict:
                entry[k] = entry[k]["value"]
        entry["property"] = prop
        entry["propertyLabel"] = properties_nl[prop]
        property_to_results_entries.append(entry)


property_to_results_entries = pd.DataFrame.from_dict(property_to_results_entries)
property_to_results_entries["objectLabel"] = property_to_results_entries.apply(lambda x: x["objectLabel"] + " " + x["ofLabel"] if x["of"] == x["of"] else x["objectLabel"], axis=1)
property_to_results_entries[property_to_results_entries["of"] == property_to_results_entries["of"]]
property_to_results_entries_grouped = property_to_results_entries.groupby(["subject", "property", "object", "startDate"]).agg({
    "subjectLabel": "first","propertyLabel": "first", "objectLabel": "first",
    "endDate": "first",
})
# sort by subject, then relation, then date
property_to_results_entries_grouped = property_to_results_entries_grouped.sort_values(by = ['subject', 'property', 'startDate'], ascending = [True, True, True])
property_to_results_entries_grouped.to_csv(dataframe_file)
