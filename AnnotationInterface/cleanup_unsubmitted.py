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

with open("AnnotationInterface/users/user_to_triple.json") as f:
    user_to_triples = json.load(f)

with open("AnnotationInterface/users/annotated_triples.json") as f:
    annotated_triples = json.load(f)


new_annotated_triples = []
for subset in subsets:
    if not os.path.exists(f"AnnotationInterface/results/property_to_results_larger{subset}_subset_links_raw.csv"):
        continue
    with open(f"AnnotationInterface/results/property_to_results_larger{subset}_subset_links_raw.csv") as f:
        df = pd.read_csv(f)
        distinct_annotated_triples = df[["subjectLabel", "propertyLabel", "objectLabel", "startDate"]].drop_duplicates().values

    with open(f"notebooks/annotation_splits/property_to_results_larger{subset}_subset_links_filtered.csv") as f:
        source_df = pd.read_csv(f)
        distinct_source_triples = source_df[["subjectLabel", "propertyLabel", "objectLabel", "startDate"]].drop_duplicates().values
    
    # cleanup duplicates and check inter-annotator agreement
    for triple in distinct_annotated_triples:
        triple_annotations = df[(df["subjectLabel"] == triple[0]) & (df["propertyLabel"] == triple[1]) & (df["objectLabel"] == triple[2]) & (df["startDate"] == triple[3])]
        if not triple_annotations["selected_link"].isnull().all():
            # chose one with a source, otherwise get one with earlier source_date
            if triple_annotations.shape[0] > 1 and triple_annotations["selected_link"].isnull().any() and not triple_annotations["selected_link"].isnull().all():
                df = df.drop(triple_annotations[triple_annotations["selected_link"] != triple_annotations["selected_link"]].index)
                triple_annotations = triple_annotations[triple_annotations["selected_link"] == triple_annotations["selected_link"]]
            if triple_annotations.shape[0] > 1:
                min_source_date = pd.to_datetime(triple_annotations["source_date"].iloc[0])
                for i in range(1, triple_annotations.shape[0]):
                    source_date = pd.to_datetime(triple_annotations["source_date"].iloc[i])
                    if source_date < min_source_date:
                        min_source_date = source_date
                for i in range(triple_annotations.shape[0]):
                    if pd.to_datetime(triple_annotations["source_date"].iloc[i]) != min_source_date:
                        df = df.drop(triple_annotations.index[i])

    # rewrite
    df.to_csv(f"AnnotationInterface/results/property_to_results_larger{subset}_subset_links_filtered.csv", index=False)

    # prepare round 2 of annotation / get unannotated triples
    if len(distinct_annotated_triples) == len(distinct_source_triples):
        # all triples are annotated -- get ones where source was not found
        distinct_annotated_triples = df[df["selected_link"] == df["selected_link"]][["subjectLabel", "propertyLabel", "objectLabel", "startDate"]].drop_duplicates().values
        distinct_annotated_triples_no_sources = df[df["selected_link"] != df["selected_link"]][["subjectLabel", "propertyLabel", "objectLabel", "startDate"]]
        # drop if count of triple is > 1
        distinct_annotated_triples_no_sources_counts = distinct_annotated_triples_no_sources.value_counts()
        for triple in distinct_annotated_triples_no_sources_counts.index:
            # add to distinct_annotated_triples only if count is > 1
            if distinct_annotated_triples_no_sources_counts[triple] > 1:
                distinct_annotated_triples = np.concatenate([distinct_annotated_triples, np.array([triple])], axis=0)
    else:
        try:
            assert len(distinct_annotated_triples) < len(distinct_source_triples)
        except:
            breakpoint()

    for triple in distinct_annotated_triples:
        # triple = tuple(triple)
        # if distinct_annotated_triples[
        #     (distinct_annotated_triples["subjectLabel"] == triple[0]) & (distinct_annotated_triples["propertyLabel"] == triple[1]) & (distinct_annotated_triples["objectLabel"] == triple[2])
        # ].shape[0] == 0:
        #     continue
        new_annotated_triples.append(triple.tolist())

    remaining_triples_count = len(distinct_source_triples) - len(distinct_annotated_triples)
    print(f"{subset} remaining triples:", remaining_triples_count)

    if remaining_triples_count == 0:
        # filter out subjs, props for which < 2 objs are annotated with sources
        for subj, prop in df[["subjectLabel", "propertyLabel"]].drop_duplicates().values:
            prop_df = df[(df["subjectLabel"] == subj) & (df["propertyLabel"] == prop)]
            if prop_df[prop_df["selected_link"] == prop_df["selected_link"]].shape[0] < 2:
                df = df.drop(prop_df.index)
        with open(f"AnnotationInterface/results/property_to_results_larger{subset}_subset_links_filtered.csv", "w") as f:
            df.to_csv(f, index=False)


with open("AnnotationInterface/users/annotated_triples.json", "w") as f:
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

with open("AnnotationInterface/users/user_to_triple.json", "w") as f:
    json.dump(new_user_to_triples, f, indent=4)
