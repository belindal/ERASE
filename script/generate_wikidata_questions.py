import pandas as pd
import numpy as np
import os
import argparse
from wikipedia.pull_external_sources import convert_timestamp
from datetime import datetime

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
    #"P263": "official residence",
    #"P286": "head coach",
    #"P463": "member of",
    #"P607": "conflict",
    # "P641": "sport",
    #"P647": "drafted by",
    #"P664": "organizer",
    #"P822": "mascot",
    #"P1307": "director/manager",
    #"P1308": "relative",
    "P1448": "official name",
    # "P27": "country of citizenship",
}

opposite_properties_nl = {
    # "P169": "
}

property_nl_to_property = {
    properties_nl[property]: property for property in properties_nl
}

properties_to_questions = {
    "P108": {
        "subject_q": ("Who is the employer of {subject}?", "{object}"),
        "binary_q": ("Is {subject} an employee of {object}?", "yes"),  # TODO get distractors
    },
    "P169": {
        "subject_q": ("Who is the CEO of {subject}?", "{object}"),
        "object_q": ("What company is {object} the CEO of?", "{subject}"),
        "binary_q": ("Is {object} the CEO of {subject}?", "yes"),  # TODO get distractors
    },
    "P488": {
        "subject_q": ("Who is the chairperson of {subject}?", "{object}"),
        "object_q": ("What organization is {object} the chairperson of?", "{subject}"),
        "binary_q": ("Is {object} the chairperson of {subject}?", "yes"),  # TODO get distractors
    },
    "P35": {
        "subject_q": ("Who is the head of state of {subject}?", "{object}"),
        "object_q": ("Where is {object} the head of state of?", "{subject}"),
        "binary_q": ("Is {object} the head of state of {subject}?", "yes"),  # TODO get distractors
    },
    "P39": {
        "subject_q": ("What government position does {subject} hold?", "{object}"),
        "binary_q": ("Does {subject} hold government position {object}?", "yes"),
    },
    "P54": {
        "subject_q": ("What sports team is {subject} a member of?", "{object}"),
        "binary_q": ("Is {subject} a member of {object}?", "yes"),
    },
    "P451": {
        "subject_q": ("Who is the unmarried partner of {subject}?", "{object}"),
        "object_q": ("Who is the unmarried partner of {object}?", "{subject}"),
        "binary_q": ("Is {object} the unmarried partner of {subject}?", "yes"),  # TODO get distractors
    },
    "P551": {
        "subject_q": ("Where does {subject} reside?", "{object}"),
        "binary_q": ("Does {subject} reside in {object}?", "yes"),
    },
    "P159": {
        "subject_q": ("Where is the headquarters location of {subject}?", "{object}"),
        "binary_q": ("Is the headquarters location of {subject} in {object}?", "yes"),
    },
    "P463": {
        "subject_q": ("What organization is {subject} a member of?", "{object}"),
        "binary_q": ("Is {subject} a member of {object}?", "yes"),
    },
    "P102": {
        "subject_q": ("What political party is {subject} a member of?", "{object}"),
        "binary_q": ("Is {subject} a member of {object}?", "yes"),
    }
}


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--wikidata_csv", type=str, default="./notebooks/property_to_results copy.csv")
    argparser.add_argument("--output_dir", type=str, default="./wikidata-data/starter")
    args = argparser.parse_args()

    wikidata_df = pd.read_csv(args.wikidata_csv)
    # breakpoint()
    # wikidata_df = wikidata_df.groupby(["property"]).agg({
    #     "startDate": "first", 
    #     "subjectLabel": "first","propertyLabel": "first", "objectLabel": "first",
    #     "endDate": "first", "referenceURL": lambda x: ','.join(x.unique())
    # })
    all_questions = []
    all_answers = []
    all_start_ts = []
    all_end_ts = []
    known_start_ts = []
    known_end_ts = []
    question_types = []

    wikidata_df["startDate"] = wikidata_df["startDate"].apply(
        lambda x: convert_timestamp("01/1970") if x != x or x.startswith("http://www.wikidata.org/.well-known") else convert_timestamp(x))
    wikidata_df["endDate"] = wikidata_df["endDate"].apply(
        lambda x: datetime.now() if x != x or x.startswith("http://www.wikidata.org/.well-known") else convert_timestamp(x))
    # consistencize timestamp formats
    wikidata_df["source_date"] = wikidata_df["source_date"].apply(lambda x: convert_timestamp(x))

    # breakpoint()
    for property_nl in wikidata_df["propertyLabel"].unique():
        property = property_nl_to_property[property_nl]
        if property not in properties_to_questions:
            continue
        property_df = wikidata_df[wikidata_df["propertyLabel"] == property_nl]
        # property_df = property_df[["subjectLabel", "objectLabel", "referenceURL", "startDate", "endDate"]]
        property_questions = []
        for i, row in property_df.iterrows():
            if row["objectLabel"] != row["objectLabel"]:
                row["objectLabel"] = "None"
            questions = [properties_to_questions[property][qa_pair][0].format(
                subject=row["subjectLabel"], object=row["objectLabel"], timestamp=row["startDate"]
            ) for qa_pair in properties_to_questions[property]]
            answer = [properties_to_questions[property][qa_pair][1].format(
                subject=row["subjectLabel"], object=row["objectLabel"], timestamp=row["startDate"]
            ) for qa_pair in properties_to_questions[property]]
            all_questions.extend(questions)
            question_types.extend([property_nl for _ in range(len(questions))])
            all_answers.extend(answer)
            all_start_ts.extend([row["startDate"].strftime("%Y-%m-%dT%H:%M:%SZ") for _ in range(len(questions))])
            all_end_ts.extend([row["endDate"].strftime("%Y-%m-%dT%H:%M:%SZ") for _ in range(len(questions))])
            known_start_ts.extend([row["source_date"].strftime("%Y-%m-%dT%H:%M:%SZ") for _ in range(len(questions))])

            has_previous_relation = ((property_df["subjectLabel"] == row["subjectLabel"]) & (property_df["startDate"] < row["startDate"])).any() and row["startDate"] == row["startDate"]
            has_next_relation = ((property_df["subjectLabel"] == row["subjectLabel"]) & (property_df["startDate"] > row["startDate"])).any() and row["endDate"] == row["endDate"]
            source_date_of_next_relation = None
            if has_next_relation:
                source_date_of_next_relation = property_df[(property_df["subjectLabel"] == row["subjectLabel"]) & (property_df["startDate"] > row["startDate"])]["source_date"]
                # parse strings into dates
                source_date_of_next_relation = pd.to_datetime(source_date_of_next_relation).min()
                try:
                    assert source_date_of_next_relation > row["source_date"]
                except:
                    print(row["subjectLabel"], row["propertyLabel"], row["objectLabel"], "current:", row["source_date"], "next:", source_date_of_next_relation)
                known_end_ts.extend([source_date_of_next_relation.strftime("%Y-%m-%dT%H:%M:%SZ") for _ in range(len(questions))])
            else:
                known_end_ts.extend([None for _ in range(len(questions))])

            if "binary_q" in properties_to_questions[property] and (has_previous_relation or has_next_relation):
                # There is a defined end date (this relationship becomes untrue at some point)
                binary_q = properties_to_questions[property]["binary_q"][0].format(
                    subject=row["subjectLabel"], object=row["objectLabel"], timestamp=row["startDate"]
                )
                binary_a = "no"
                all_questions.append(binary_q)
                question_types.append(property_nl)
                all_answers.append(binary_a)
                if has_previous_relation:
                    # This relationship is false before current timespan
                    assert row["startDate"] == row["startDate"]  # defined start date
                    all_start_ts.append(None)
                    all_end_ts.append(row["startDate"].strftime("%Y-%m-%dT%H:%M:%SZ"))
                    known_start_ts.append(None)
                    known_end_ts.append(row["source_date"].strftime("%Y-%m-%dT%H:%M:%SZ"))
                elif has_next_relation:
                    # This relationship is false after current timespan
                    try:
                        assert row["endDate"] == row["endDate"]  # defined end date
                    except:
                        breakpoint()
                    all_start_ts.append(row["endDate"].strftime("%Y-%m-%dT%H:%M:%SZ"))
                    all_end_ts.append(None)
                    assert source_date_of_next_relation is not None
                    known_start_ts.append(source_date_of_next_relation)
                    known_end_ts.append(None)
    
    # Question,Answer,Answer choices,Start timestamp,End timestamp
    df_qs = pd.DataFrame({
        "Question": all_questions,
        "Answer": all_answers,
        "Answer choices": None,
        "Start timestamp": all_start_ts,
        "End timestamp": all_end_ts,
        "Known start timestamp": known_start_ts, 
        "Known end timestamp": known_end_ts,
        "Question type": question_types,
    })
    df_qs.to_csv(os.path.join(args.output_dir, "questions.csv"), index=False)
    

