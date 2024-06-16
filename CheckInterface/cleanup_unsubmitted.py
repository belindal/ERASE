"""
1. Creates annotation set for round 2 (annotations where 1 person has annotated and could not find a source)
2. Drops duplicate annotations (keeps one with source, or one with earlier source_date if no source)
3. Removes unsubmitted annotations
4. Once round 2 is over, removes (subj, prop) pairs where < 2 objs are annotated with sources
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime


subsets = ["_sports", "_position", ""]
# subsets = ["_position"]

with open("CheckInterface/users/user_to_triple.json") as f:
    user_to_triples = json.load(f)

with open("CheckInterface/users/annotated_triples.json") as f:
    annotated_triples = json.load(f)

new_annotated_triples = []
# disagree_triples = []
for subset in subsets:
    if not os.path.exists(f"CheckInterface/results/property_to_results_larger{subset}_subset_links.csv"):
        continue
    if os.path.exists(f"CheckInterface/results/property_to_results_larger{subset}_subset_links_filtered.csv"):
        with open(f"CheckInterface/disagree_triples{subset}.json") as f:
            disagree_triples = json.load(f)
    else:
        disagree_triples = []
    disagree_triple_map = {tuple(triple["triple"]): {
        "source1": triple["source1"],
        "source1_span": triple["source1_span"],
        "source2": triple["source2"],
        "source2_span": triple["source2_span"],
        "valid_source": triple.get("valid_source")
    } for triple in disagree_triples}
    disagree_triples = []
    with open(f"CheckInterface/results/property_to_results_larger{subset}_subset_links.csv") as f:
        df = pd.read_csv(f)
        distinct_annotated_triples = df[["subjectLabel", "propertyLabel", "objectLabel", "startDate"]].drop_duplicates().values

    # remove users / annotations for which the user responded with empty links for everything
    for user in df["annotator"].unique():
        user_annotations = df[df["annotator"] == user]
        if user_annotations["selected_link"].isnull().all():
            df = df[df["annotator"] != user]
    
    with open(f"CheckInterface/results/property_to_results_larger{subset}_subset_links.csv", "w") as f:
        df.to_csv(f, index=False)

    with open(f"AnnotationInterface/results/property_to_results_larger{subset}_subset_links_filtered.csv") as f:
        round1 = pd.read_csv(f)
        # drop ones without sources
        round1 = round1[round1["selected_link"] == round1["selected_link"]]
        distinct_round1_triples = round1[["subjectLabel", "propertyLabel", "objectLabel", "startDate"]].drop_duplicates().values
    
    rows_to_keep = pd.DataFrame()
    # cleanup duplicates and check inter-annotator agreement
    for triple in distinct_annotated_triples:
        if triple[2] == triple[2]:
            triple_annotations = df[(
                df["subjectLabel"] == triple[0]
            ) & (
                df["propertyLabel"] == triple[1]
            ) & (
                df["objectLabel"] == triple[2]
            ) & (df["startDate"] == triple[3])]
            prev_triple = round1[(
                round1["subjectLabel"] == triple[0]
            ) & (round1["propertyLabel"] == triple[1]) & (
                round1["objectLabel"] == triple[2]
            ) & (round1["startDate"] == triple[3])]
        else:
            triple_annotations = df[(
                df["subjectLabel"] == triple[0]
            ) & (
                df["propertyLabel"] == triple[1]
            ) & (
                df["startDate"] == triple[3]
            ) & (df["objectLabel"] != df["objectLabel"])]
            prev_triple = round1[(
                round1["subjectLabel"] == triple[0]
            ) & (round1["propertyLabel"] == triple[1]) & (
                round1["startDate"] == triple[3]
            ) & (round1["objectLabel"] != round1["objectLabel"])]
        if triple_annotations["selected_link"].notnull().any():
            try:
                selected_link = triple_annotations["selected_link"][triple_annotations["selected_link"].notnull()].iloc[0]
            except:
                breakpoint()
            selected_row = triple_annotations[triple_annotations["selected_link"] == selected_link].iloc[0]
            selected_row = selected_row.to_frame().T
        else:
            selected_link = triple_annotations["selected_link"].iloc[0]
            selected_row = triple_annotations.iloc[0].to_frame().T
        # breakpoint()
        prev_selected_link = prev_triple["selected_link"].iloc[0]
        if prev_selected_link == selected_link:
            # good source
            rows_to_keep = pd.concat([rows_to_keep, selected_row])
            # if rows_to_keep[(rows_to_keep["subjectLabel"] == triple_annotations["subjectLabel"].iloc[0]) & (rows_to_keep["propertyLabel"] == triple_annotations["propertyLabel"].iloc[0]) & (rows_to_keep["objectLabel"] == triple_annotations["objectLabel"].iloc[0])].shape[0] > 1:
            #     breakpoint()
        # elif tuple(triple.tolist()) not in disagree_triple_map:
        elif tuple(triple.tolist()) in disagree_triple_map and disagree_triple_map[tuple(triple.tolist())].get("valid_source") is not None:
            if disagree_triple_map[tuple(triple.tolist())]["valid_source"] == "source1" and prev_triple["selected_link"].iloc[0] == selected_link:
                # keep source1
                rows_to_keep = pd.concat([rows_to_keep, prev_triple])
            elif disagree_triple_map[tuple(triple.tolist())]["valid_source"] == "source2" and selected_row["selected_link"].iloc[0] == selected_link:
                # keep source2
                rows_to_keep = pd.concat([rows_to_keep, selected_row])
            disagree_triples.append({
                "triple": triple.tolist(),
                "source1": prev_selected_link,
                "source1_span": prev_triple["span"].iloc[0],
                "source2": selected_link,
                "source2_span": selected_row["span"].iloc[0],
                "valid_source": disagree_triple_map[tuple(triple.tolist())]["valid_source"],
            })
        else:
            # print(f"Disagreement for {triple}")
            disagree_triples.append({
                "triple": triple.tolist(),
                "source1": prev_selected_link,
                "source1_span": prev_triple["span"].iloc[0],
                "source2": selected_link,
                "source2_span": selected_row["span"].iloc[0],
            })
            # breakpoint()

    for triple in distinct_annotated_triples:
        new_annotated_triples.append(triple.tolist())

    remaining_triples_count = len(distinct_round1_triples) - len(distinct_annotated_triples)
    remaining_triples = set(tuple(triple) for triple in distinct_round1_triples) - set(tuple(triple) for triple in distinct_annotated_triples)
    print(f"{subset} remaining triples:", remaining_triples_count)

    inter_annotator_agreement = 1 - len(disagree_triples) / len(distinct_annotated_triples)

    print(f"{subset} inter-annotator agreement: {len(distinct_annotated_triples) - len(disagree_triples)} / {len(distinct_annotated_triples)} = ", inter_annotator_agreement)

    # clean up rows where (subj, rel) have < 2 objs with sources
    rows_to_keep = rows_to_keep.groupby(["subjectLabel", "propertyLabel"]).filter(lambda x: len(x) > 1)
    with open(f"CheckInterface/results/property_to_results_larger{subset}_subset_links_filtered.csv", "w") as f:
        rows_to_keep.to_csv(f, index=False)
    print(f"{subset} remaining triples after filtering:", len(rows_to_keep))

    with open(f"CheckInterface/disagree_triples{subset}.json", "w") as f:
        json.dump(disagree_triples, f, indent=4)

    print(f"CheckInterface/disagree_triples{subset}.json")


with open("CheckInterface/users/annotated_triples.json", "w") as f:
    json.dump(new_annotated_triples, f, indent=4)

annotated_triples = [tuple(triple) for triple in new_annotated_triples]
annotated_triples_count = {
    triple: annotated_triples.count(triple) for triple in annotated_triples
}

new_user_to_triples = {}
for user in user_to_triples:
    for triple in user_to_triples[user]:
        if tuple(triple) in annotated_triples:
            if user not in new_user_to_triples:
                new_user_to_triples[user] = []
            new_user_to_triples[user].append(triple)

with open("CheckInterface/users/user_to_triple.json", "w") as f:
    json.dump(new_user_to_triples, f, indent=4)
