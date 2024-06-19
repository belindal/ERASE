"""
Abstract class with methods specific to HMM dataset
"""
from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
import math
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSeq2SeqLM, pipeline
import faiss
from pandas import Timestamp
from sentence_transformers import SentenceTransformer
import json
import copy
from difflib import SequenceMatcher
import os
from utils import count_tokens, TOKENIZER
from GENRE.genre.trie import Trie


FEWSHOT_EXAMPLES = True
probability_mapping = {
    "true": 1.0,
    "almost certain": 0.9999,
    "very likely": 0.9,
    "likely": 0.75,
    "somewhat likely": 0.6,
    "uncertain": 0.5,
    "somewhat unlikely": 0.4,
    "unlikely": 0.25,
    "very unlikely": 0.1,
    "almost impossible": 1e-4,
    "false": 0.0,
}
probability_units = [1e-3, 0.999]


class Entity: pass


class Fact:
    sentence: str
    entities: List[Entity]  # entities mentioned in fact

    def __init__(self, sentence, entities=None):
        self.sentence = sentence
        self.entities = entities

    def __eq__(self, other):
        return self.sentence == other.sentence
    
    def __hash__(self):
        return hash(self.sentence)
    
    def __str__(self):
        return self.sentence

class Entity:
    name: str  # entity name
    facts: List[Fact]  # facts mentioning entity

    def __init__(self, name, facts=None):
        self.name = name
        if facts is None:
            self.facts = []
        else:
            self.facts = facts

    def add_fact(self, fact):
        self.facts.append(fact)
    
    def __eq__(self, other):
        return self.name == other.name
    
    def __hash__(self):
        return hash(self.name)
    
    def __str__(self):
        return self.name
    



class KBModel:
    def __init__(
        self, model_name, save_as_facts, overwrite_facts, retrieve_facts,
        get_supporting_fact_update_direction,
        query_lm_fn,
        device, context_length=4000, save_fn=None, hops=1, tokenizer=None,
        logging=False,
        **kwargs,
    ):
        self.model_name = model_name
        self.save_as_facts = save_as_facts
        self.overwrite_facts = overwrite_facts
        self.retrieve_facts = retrieve_facts
        self.update_mode = None
        self.query_lm_fn = query_lm_fn
        self.get_supporting_fact_update_direction = get_supporting_fact_update_direction
        self.fact_noise = 1e-3
        self.device = device
        self.hops = hops
        self.kb = {}  # Fact -> [{"ts": ts, "probability": prob, "source_ids": source_ids, "connected_facts": connected_facts, "contexts": contexts}]
        self.all_entities = {}  # entity name -> Entity
        self.all_entity_tokens = []
        self.all_facts = []
        self.cache_file = {}

        self.retrieval_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/gtr-t5-large')
        self.retrieval_model = SentenceTransformer('sentence-transformers/gtr-t5-large')
        self.model_output_dim = self.retrieval_model.get_sentence_embedding_dimension()
        # if self.retrieve_facts == "instruct_similarity":
        #     # self.retrieval_tokenizer = EncT5Tokenizer.from_pretrained("facebook/tart-full-flan-t5-xl")
        #     # self.retrieval_model = EncT5ForSequenceClassification.from_pretrained("facebook/tart-full-flan-t5-xl")
        #     self.retrieval_tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")
        #     self.retrieval_model = AutoModel.from_pretrained("facebook/contriever-msmarco")
        #     state_dict = torch.load("tart-dual-contriever-msmarco/checkpoint.pth")
        #     self.retrieval_model.pooler = None
        #     self.retrieval_model.load_state_dict(state_dict['model'])
        #     self.model_output_dim = self.retrieval_model.config.hidden_size

        self.relevance_threshold = 0.7

        if self.retrieve_facts:
            self.all_embeddings = np.empty((0,self.model_output_dim))
            self.index = faiss.IndexFlatIP(self.model_output_dim)  # 768
            self.retrieval_model.to(self.device)
            self.retrieval_model.eval()
        self.tokenizer = tokenizer
        self.num_sentences_to_retrieve = 5
        self.context_length = context_length

        # load entity extraction model
        ner_tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        ner_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER").eval()
        self.ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer)
        # load entity linking model
        self.entity_linker_tokenizer = AutoTokenizer.from_pretrained("facebook/genre-linking-aidayago2")
        self.entity_linker = AutoModelForSeq2SeqLM.from_pretrained("facebook/genre-linking-aidayago2").eval()
        self.entity_linker.to(self.device)
        self.cache_file[self.entity_linker] = f"cache/facebook/genre-linking-aidayago2/retrieval_cache.txt"
        self.entity_trie = Trie()  # contains entities in all_entities
        self.cache = self.load_cache()
        self.save_fn = save_fn
        self.logging = logging

        # self.fact_extractor_tokenizer = AutoTokenizer.from_pretrained("chentong00/propositionizer-wiki-flan-t5-large")
        # self.fact_extractor_model =  AutoModelForSeq2SeqLM.from_pretrained("chentong00/propositionizer-wiki-flan-t5-large").to(self.device)
    
    def reset(self):
        self.kb = {}
        self.all_entities = {}
        self.all_entity_tokens = []
        self.all_facts = []
        if self.retrieve_facts:
            self.all_embeddings = np.empty((0,self.model_output_dim))
            self.index.reset()
        self.entity_trie = Trie()
        
    
    def load_cache(self):
        """
        load cache from file
        """
        cache = {}
        for model in self.cache_file:
            cache[model] = {}
            if not os.path.exists(os.path.dirname(self.cache_file[model])):
                os.makedirs(os.path.dirname(self.cache_file[model]))
            if not os.path.exists(self.cache_file[model]):
                continue
            with open(self.cache_file[model], "r") as rf:
                for line in rf:
                    line = json.loads(line)
                    cache[model].update(line)
        return cache
    
    def embed_facts(self, facts: List[Fact]):
        """
        embed facts, return a np array of embeddings
        """
        facts_nl = [fact.sentence for fact in facts]
        if self.retrieve_facts == "similarity" or self.retrieve_facts == "entity_similarity":
            all_embeddings = self.retrieval_model.encode(facts_nl)
        elif self.retrieve_facts == "instruct_similarity":
            # Apply tokenizer
            inputs = self.retrieval_tokenizer(facts_nl, padding=True, truncation=True, return_tensors='pt').to(self.device)
            # Compute token embeddings
            outputs = self.retrieval_model(**inputs)
            # Mean pooling
            def mean_pooling(token_embeddings, mask):
                token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
                sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
                return sentence_embeddings
            all_embeddings = mean_pooling(outputs[0], inputs['attention_mask']).cpu().detach().numpy()

        return all_embeddings

    def add_fact_to_kb(self, facts: List[Fact]):
        """
        embed facts and add to index
        """
        self.all_facts.extend(facts)
        if self.retrieve_facts:
            embeddings = self.embed_facts(facts)
            self.index.add(embeddings)
            self.all_embeddings = np.concatenate([self.all_embeddings, embeddings], axis=0)
            assert len(self.all_facts) == self.index.ntotal
            assert len(self.all_embeddings) == self.index.ntotal
    
    def remove_fact_from_kb(self, fact):
        """
        remove fact from kb
        """
        fact_index = self.all_facts.index(fact)
        self.all_facts.pop(fact_index)
        self.all_embeddings = np.delete(self.all_embeddings, fact_index, axis=0)
        self.index.remove_ids(faiss.IDSelectorArray([fact_index]))
        assert len(self.all_facts) == self.index.ntotal
        assert len(self.all_embeddings) == self.index.ntotal
 
    def save_context_facts(self, context, ts, chunk_idx, source_id=None, read_ts=None):
        """
        update KB according to context
        """
        fact_to_score = {}
        fact_to_propagated_fact_to_new_score = {}
        fact_to_score[ts] = []
        if read_ts is None:
            read_ts = ts
        fact_to_propagated_fact_to_new_score[ts] = []

        if chunk_idx == 0:
            # assume sequential presentation to model
            self.source_id_to_initial_chunk_facts = {source_id: []}

        operations, _, _ = self.get_operations(
            context, ts, source_id,
            chunk_idx, initial_chunk_facts=self.source_id_to_initial_chunk_facts.get(source_id, []))

        fact_to_propagated_fact_to_new_score[ts].append([])
        for operation in operations:
            fact = operation["fact"]  # old fact for rewriting
            source_ids = operation["new_source_ids"]
            connected_facts = operation["new_connected_facts"]
            contexts = operation["new_contexts"]

            if operation["operation"] == "update_fact":
                assert fact in self.kb
                likelihood_values = sorted(probability_units)
                curr_likelihood = self.kb[fact][-1]["probability"]
                if operation["update_direction"] == "increase":
                    new_probability = likelihood_values[min(len(likelihood_values)-1, likelihood_values.index(curr_likelihood) + 1)]
                elif operation["update_direction"] == "decrease":
                    new_probability = likelihood_values[max(0, likelihood_values.index(curr_likelihood) - 1)]
                elif operation["update_direction"] == "unknown":
                    new_probability = 0.5
                else:
                    new_probability = curr_likelihood
                if self.kb[fact][-1]["ts"] != ts:
                    self.kb[fact].append({
                        "ts": ts, "probability": new_probability,
                        "source_ids": source_ids,
                        "connected_facts": connected_facts,
                        "contexts": contexts,
                        "read_ts": read_ts,
                    })
                else:
                    self.kb[fact][-1]["probability"] = new_probability
                    self.kb[fact][-1]["source_ids"] = source_ids
                    self.kb[fact][-1]["connected_facts"] = connected_facts
                    self.kb[fact][-1]["contexts"] = contexts
                    self.kb[fact][-1]["read_ts"] = read_ts

            elif operation["operation"] == "rewrite_fact":
                assert fact in self.kb
                new_fact = operation["new_fact"]
                if new_fact == fact: continue
                if new_fact not in self.kb:
                    self.kb[new_fact] = [{
                        "ts": ts, "probability": 1 - self.fact_noise,
                        "source_ids": source_ids,
                        "connected_facts": connected_facts,
                        "contexts": contexts,
                        "read_ts": read_ts,
                    }]
                    self.add_fact_to_kb([new_fact])
                # add info from old fact to new fact
                old_fact_entries = self.kb[fact]
                for fact_entry in old_fact_entries:
                    if "fact" not in fact_entry:
                        # save version of fact before rewrite (if not already saved as something else, indicating a prior rewrite)
                        fact_entry["fact"] = fact.sentence
                self.kb[new_fact].extend(old_fact_entries)
                self.kb[new_fact].sort(key=lambda x: x["ts"])
                self.remove_fact_from_kb(fact)
                if chunk_idx== 0:
                    self.source_id_to_initial_chunk_facts[source_id].append(new_fact)

            elif operation["operation"] == "add_fact":
                truthfulness_score = 1 - self.fact_noise
                fact_to_score[ts].append({"fact": fact.sentence, "score": truthfulness_score})
                self.add_fact(
                    fact, truthfulness_score, ts,
                    source_ids=operation["new_source_ids"],
                    connected_facts=operation["new_connected_facts"],
                    contexts=operation["new_contexts"],
                    read_ts=read_ts,
                )
                if chunk_idx== 0:
                    self.source_id_to_initial_chunk_facts[source_id].append(fact)
            else:
                print(f"[ERROR] Unrecognized operation: {operation['operation']}.")

        return fact_to_score, fact_to_propagated_fact_to_new_score

    def find_fact(self, fact, facts: List[Fact]):
        """
        find lexically closest fact in facts
        """
        max_lexical_similarity = 0
        max_lexical_similarity_fact = None
        for f in facts:
            similarity = SequenceMatcher(None, f.sentence, fact).ratio()
            if similarity > max_lexical_similarity:
                max_lexical_similarity = similarity
                max_lexical_similarity_fact = f
        if max_lexical_similarity > 0.9:
            return max_lexical_similarity_fact
        return None
    
    def cached_generate(self, model, tokenizer, inputs, **kwargs):
        """
        generate with caching
        """
        del kwargs['return_dict_in_generate']
        del kwargs['output_scores']
        kwargs_key = {key: value for key, value in kwargs.items() if key not in ["return_dict_in_generate", "output_scores"]}
        if "prefix_allowed_tokens_fn" not in kwargs and "constraints" not in kwargs:
            # don't know what's in the KB...
            key = str({
                "inputs": inputs,
                "kwargs": kwargs_key,
            })
            if key in self.cache[model]:
                return self.cache[model][key]
        input_ids = tokenizer(
            inputs, return_tensors="pt",
            padding=True, truncation=True, max_length=1024
        ).to(self.device)
        try:
            outputs = model.generate(**input_ids, **kwargs, return_dict_in_generate=True, output_scores=True)
        except ValueError as e:
            if "constraint is unsatisfiable." in str(e) or "max() arg is an empty sequence" in str(e):
                # no valid outputs
                return {
                    "sequences": [None],
                    "sequences_scores": [float("-inf")],
                }

        output_sequences = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        outputs = {
            "output_tokens": outputs.sequences.tolist(),
            "sequences": output_sequences,
            "sequences_scores": outputs.sequences_scores.tolist(),
        }
        if "prefix_allowed_tokens_fn" not in kwargs:
            self.cache[key] = outputs
            with open(self.cache_file[model], "a") as wf:
                wf.write(json.dumps({key: outputs}) + "\n")
        return outputs

    def ner(self, sentence):
        """
        extract entities from sentence
        """
        if len(sentence) == 0:
            return []
        messages = [{
            "role": "system",
            "content": f"""You will be given some sentence(s). Extract all named entities from the given sentence. Output your answer as a list ["entity", "entity", ...]. Copy and paste named entities exactly as they appear in the sentence. Only output a list and nothing else. Output an empty list if there are no named entities."""
        },
        {
            "role": "user",
            "content": f"""On November 17, 2023, the board removed Altman as CEO, while Brockman was removed as chairman and then resigned as president."""
        },
        {
            "role": "assistant",
            "content": f"""["November 17, 2023", "Altman", "CEO", "Brockman", "chairman", "president"]"""
        },
        {
            "role": "user",
            "content": f"""Several issues with glitches, design flaws and security vulnerabilities were cited"""
        },
        {
            "role": "assistant",
            "content": f"""[]"""
        },
        {
            "role": "user",
            "content": sentence
        }]
        output_text, _ = self.query_lm_fn(
            messages,
            inference_kwargs={"temperature": 0.0, "max_tokens": min(len(sentence), self.context_length), "stop": ["\n"]},
        )
        try:
            output_text = json.loads(output_text.split("\n")[0].strip())
            entities = []
            for entry in output_text:
                entities.append({
                    "text": entry,
                    "start": sentence.find(entry),
                    "end": sentence.find(entry) + len(entry),
                })
        except:
            tags = self.ner_pipeline(sentence)
            entities = []
            for i, tag in enumerate(tags):
                if tag["entity"].startswith("B"):
                    entity = {"text": sentence[tag["start"]:tag["end"]], "start": tag["start"], "end": tag["end"]}
                    for j in range(i+1, len(tags)):
                        if tags[j]["entity"].startswith("I"):
                            entity["end"] = tags[j]["end"]
                            entity["text"] = sentence[entity["start"]:entity["end"]]
                        else:
                            break
                    entities.append(entity)
        return entities


    def extract_entities(self, sentence, context=None, fact_to_extracted_entities={}):
        """
        extract entities from sentence
        """
        if self.retrieve_facts != "entity_similarity":
            return []

        if sentence == "":
            return []

        def prefix_allowed_tokens_fn(batch_id, sent): 
            return [2] if len(self.entity_trie.get(sent.tolist())) == 0 else self.entity_trie.get(sent.tolist())

        entities = []  # get from KB
        new_entity_names = []
        if sentence in fact_to_extracted_entities:
            entity_names = fact_to_extracted_entities[sentence]
            for entity in entity_names:
                if entity not in self.all_entities:
                    new_entity_names.append(entity)
                else:
                    entities.append(self.all_entities[entity])
        else:
            # query LM
            sentence_content = f"Sentence: {sentence}"
            if context is not None:
                sentence_content = f"[Begin Context] {context} [End Context]\n{sentence_content}"
            messages = [
                {"role": "system", "content": f"""List all named entities the sentence pertains to. You may be given a context with further information for disambiguating the named entity. Only output a list and nothing else.
E.g.
Input: Sam Altman was fired by the board of OpenAI on Friday. Following his ouster, hundreds of OpenAI employees, including co-founder and board member Ilya Sutskever, signed a letter demanding that remaining board members resign or they would leave. Altman has been reinstated by the board. Microsoft CEO Satya Nadella said Monday that Altman and others from OpenAI would be joining to start a "new advanced AI research team."
Sentence: On November 17, 2023, the board removed Altman as CEO, following which Sutskever signed a letter demanding remaining board members resign.

Output: ["board of OpenAI", "Sam Altman", "Ilya Sutskever"]

Input: Sentence: who is the voice of tony?
Output: ["Tony"]"""},
                {"role": "user", "content": sentence_content},
            ]
            output_text, _ = self.query_lm_fn(
                messages,
                inference_kwargs={"temperature": 0.0, "max_tokens": min(2 * count_tokens(self.tokenizer, sentence), self.context_length), "repetition_penalty": 1.0},
            )
            output_text = "[" + output_text.split("[")[-1].split("]")[0] + "]"
            try:
                entity_names = json.loads(output_text.split("\n")[0].strip())
                for entity in entity_names:
                    if entity not in self.all_entities:
                        new_entity_names.append(entity)
                    else:
                        entities.append(self.all_entities[entity])
            except:
                extracted_entities = self.ner(sentence)

                entity_scores = []
                hypothesized_entities = []
                hypothesized_new_entity = []
                new_entity_scores = []
                for ent in extracted_entities:
                    sentences=[
                        sentence[:ent['start']] + "[START_ENT] " + ent['text'] + " [END_ENT] " + sentence[ent['end']:]
                    ]
                    constrained_outputs = self.cached_generate(
                        self.entity_linker, self.entity_linker_tokenizer, sentences,
                        num_beams=5, num_return_sequences=5, return_dict_in_generate=True, output_scores=True,
                        # OPTIONAL: use constrained beam search
                        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    )
                    unconstrained_outputs = self.cached_generate(
                        self.entity_linker, self.entity_linker_tokenizer, sentences,
                        num_beams=5, num_return_sequences=5, return_dict_in_generate=True, output_scores=True,
                    )
                    entity_scores.append(math.exp(constrained_outputs['sequences_scores'][0]))
                    hypothesized_entities.append(constrained_outputs['sequences'][0])
                    hypothesized_new_entity.append(unconstrained_outputs['sequences'][0])
                    new_entity_scores.append(math.exp(unconstrained_outputs['sequences_scores'][0]))
                    # check whether the entity was correctly identified / whether we need to go down the list(?)
                    if math.exp(constrained_outputs['sequences_scores'][0]) > 0.6:
                        entities.append(self.all_entities[constrained_outputs['sequences'][0]])
                    else:
                        if math.exp(unconstrained_outputs['sequences_scores'][0]) > 0.8:
                            # make a new entity
                            # TODO what if this mention isn't the best name for the entity?
                            entity_name = unconstrained_outputs['sequences'][0]
                            new_entity_names.append(entity_name)
                        else:
                            entity_name = ent['text']
                            new_entity_names.append(entity_name)
        for entity_name in new_entity_names:
            entity_tokens = [self.entity_linker_tokenizer.eos_token_id] + self.entity_linker_tokenizer.encode(str(entity_name), add_special_tokens=False) + [self.entity_linker_tokenizer.eos_token_id]
            if entity_name not in self.all_entities:
                # May be another way of tokenizing the same entity...
                new_entity = Entity(name=entity_name)
                entities.append(new_entity)
                self.all_entities[entity_name] = new_entity
            self.entity_trie.add(entity_tokens)
            self.all_entity_tokens.append(entity_tokens)
        # uniquify
        uniq_entities = []
        for entity in entities:
            if entity not in uniq_entities:
                uniq_entities.append(entity)
        if self.logging:
            with open(f"{self.save_fn}_fact_extraction.txt", "a") as wf:
                wf.write(f"[Fact:] {sentence}\n[Entities:] {[uniq_ent.name for uniq_ent in uniq_entities]}\n")
        return uniq_entities


    def contains_entity(self, sentence_entities, all_entities):
        """
        check if intersection of sentence_entities and all_entities is non-empty
        """
        if sentence_entities is None:
            return True
        return len(set(sentence_entities).intersection(set(all_entities))) > 0


    def retrieve_facts_to_edit(self, context, ts, chunk_idx=0, initial_chunk_facts=None):
        """
        Retrieve relevant facts to edit based on context
        """
        all_relevant_facts = []
        all_relevant_facts_ts_truthscores = []
        all_relevant_facts_nl = []

        if not self.overwrite_facts:
            return "", all_relevant_facts

        # retrieve facts for overwriting
        relevance_threshold = self.relevance_threshold
        if self.retrieve_facts == "entity_similarity":
            # retrieve ones with same entities, above this threshold
            relevance_threshold -= 0.2
        relevant_facts, relevant_facts_ts_truthscores, relevance_scores, relevant_facts_nl = self.retrieve(
            query=context, ts=ts, max_sentences_to_retrieve=self.context_length,
            relevance_threshold=relevance_threshold,
            instruction="Extract facts that are supported, contradicted, or should be updated based on the input text.",
            retrieve_facts=self.overwrite_facts,
        )
        all_relevant_facts.extend(relevant_facts)
        all_relevant_facts_nl.extend(relevant_facts_nl)
        all_relevant_facts_ts_truthscores.extend(relevant_facts_ts_truthscores)
        # save retrieved facts
        if self.logging:
            with open(f"{self.save_fn}_fact_extraction.txt", "a") as wf:
                wf.write(f"[1-HOP RETRIEVED FACTS:]\n{json.dumps(all_relevant_facts_nl, indent=4)}\n")
        one_hop_facts = copy.deepcopy(all_relevant_facts_nl)
        # iterative retrieval for multi-hop facts:
        if self.hops == 2:
            for r, rel_fact in enumerate(relevant_facts):
                rel_fact_score = relevance_scores[r]
                # retrieve more facts
                relevant_facts_2, _, _, _ = self.retrieve(
                    query=rel_fact.sentence, ts=ts, max_sentences_to_retrieve=self.context_length, relevance_threshold=(relevance_threshold / rel_fact_score),
                    instruction="Extract facts that are supported, contradicted, or should be updated based on the input text.",
                    retrieve_facts=self.overwrite_facts,
                )
                all_relevant_facts_ts_truthscores.extend([self.kb[fact] for fact in relevant_facts_2 if fact not in all_relevant_facts])
                all_relevant_facts_nl.extend([fact.sentence for fact in relevant_facts_2 if fact not in all_relevant_facts])
                all_relevant_facts.extend([fact for fact in relevant_facts_2 if fact not in all_relevant_facts])
        if chunk_idx > 0:
            assert initial_chunk_facts is not None #and len(initial_chunk_facts) > 0
            all_relevant_facts = initial_chunk_facts[:5] + [fact for fact in all_relevant_facts if fact not in initial_chunk_facts]
            all_relevant_facts_ts_truthscores = [self.kb[fact] for fact in all_relevant_facts]
            all_relevant_facts_nl = [fact.sentence for fact in all_relevant_facts]
        
        two_hop_facts = list(set(all_relevant_facts_nl) - set(one_hop_facts))
        if self.logging:
            with open(f"{self.save_fn}_fact_extraction.txt", "a") as wf:
                wf.write(f"[2-HOP RETRIEVED FACTS:]\n{json.dumps(two_hop_facts, indent=4)}\n")

        retrieved_facts_tokens_percent = (1-((count_tokens(self.tokenizer, context))/self.context_length))
        prompt_facts = self.convert_facts_to_nl_story(
            all_relevant_facts, all_relevant_facts_ts_truthscores,
            use_timestamps=False, use_likelihood=False, remove_untrue=True,
            percent_of_context=retrieved_facts_tokens_percent, use_recent=False,
        )
        
        return prompt_facts, all_relevant_facts


    def edit_retrieved_facts(self, context, ts, prompt_facts, all_relevant_facts):
        """
        Edit retrieved facts based on context
        """
        extracted_facts = {}
        if self.overwrite_facts and len(all_relevant_facts) > 0:
            all_prompt_facts = []
            still_true_facts = []
            false_unknown_facts = []
            for f, fact in enumerate(prompt_facts.split("*===*")):
                fact = fact.strip()
                if fact.strip() == "" or fact in all_prompt_facts:
                    continue
                all_prompt_facts.append(fact)
            for f, fact in enumerate(all_prompt_facts):
                query_messages = [{"role": "user", "content": f"""[Input] [Timestamp: {ts}] {context} [End Input]

The fact "{fact}" was previously true. In light of the input, is "{fact}" likely still true as of {ts}? Begin by summarizing the changes we learned from the input, then reasoning briefly about them to give your final answer with "Answer: Reinforce" (if the input makes the fact more likely) or "Answer: Make False" (if the input makes the fact less likely) or "Answer: No Change" (if the input doesn't affect the fact, e.g. if the input is irrelevant to the fact). Assume that the fact is still true (keep true) if nothing in the input contradicts it."""}]
                output_text, _ = self.query_lm_fn(
                    query_messages,
                    inference_kwargs={"temperature": 0.0, "max_tokens": 100})
                output_text = output_text.strip().strip("\"").strip().split("\n")[0].lstrip("<").rstrip(">")
                not_valid = True
                num_loops = 0
                while not_valid:
                    output_text = output_text.strip().split("Answer:")[-1].strip().strip(".")
                    try:
                        assert "reinforce" in output_text.lower() or "make false" in output_text.lower() or "no change" in output_text.lower()
                        break
                    except:
                        num_loops += 1
                        query_messages.append({"role": "user", "content": f"""Give your final answer with "Answer: Reinforce" or "Answer: Make False" or "Answer: No Change". Include no other text."""})
                        output_text, _ = self.query_lm_fn(
                            query_messages,
                            inference_kwargs={"temperature": 0.0, "max_tokens": 100})
                        if num_loops > 5:
                            output_text = "no change"
                            break
                if "make false" in output_text.lower():
                    false_unknown_facts.append(fact)
                else:
                    still_true_facts.append(fact)
                    extracted_facts[fact] = "reinforce" if "reinforce" in output_text.lower() else "irrelevant"
            for fact in false_unknown_facts:
                query_messages = [{"role": "user", "content": f"""[Input] [Timestamp: {ts}] {context}
Other True Facts at {ts}: {", ".join(still_true_facts)}
[End Input]

The fact "{fact}" was previously true but no longer. Given the above input and true facts, can you rewrite it into one that is true as of {ts}? Output your answer in form "rewrite: rewritten fact" or "no rewrite possible"."""}]
                output_text, _ = self.query_lm_fn(
                    query_messages,
                    inference_kwargs={"temperature": 0.0, "max_tokens": 2 * len(fact)})
                output_text = output_text.strip().split("Corrected Version:")[-1].split("Corrected Fact:")[-1].split("Output:")[-1].strip()
                output_text = output_text.strip().split("Rewrite:")[-1].split("rewrite:")[-1].strip()
                if "no rewrite possible" in output_text.lower():
                    extracted_facts[fact] = "make false"
                else:
                    extracted_facts[fact] = f"update: {output_text}"
            
            if self.logging:
                with open(f"{self.save_fn}_fact_extraction.txt", "a") as wf:
                    wf.write(f"\n\n[FACTS:]\n{json.dumps(extracted_facts, indent=4)}\n\n")

            rewritten_retrieved_facts = []
            for f, fact in enumerate(extracted_facts):
                if extracted_facts[fact] == "set true":
                    rewritten_retrieved_facts.append(fact)
                elif extracted_facts[fact] == "set false":
                    pass
                elif extracted_facts[fact].startswith("rewrite"):
                    rewritten_retrieved_facts.append(": ".join(extracted_facts[fact].split(": ")[1:]))
            rewritten_retrieved_facts = "\n*===*\n".join(rewritten_retrieved_facts)
            prompt_facts = rewritten_retrieved_facts
        
        return extracted_facts


    def extract_new_facts(self, context, ts, source_id, chunk_idx, full_operation_list, all_changed_facts):
        """
        Extract new facts from context
        """
        fact_to_extracted_entities = {}
        messages = [
            {"role": "system", "content": f"""Extract all facts from the input text, with each fact on a new line and without bullet points or numbered lists. Facts should be simple, independent, standalone, and decontextualized. Break up long facts into smaller facts. Resolve all references (e.g. pronouns, definite articles, etc.) by copying full reference object everywhere it is referenced. Only include facts referring to the current world state (what is true *now*), as opposed to facts true in the past. If there are no facts, please output "No new facts." Do not include any other text."""},
        ]
        if FEWSHOT_EXAMPLES:
            messages.extend([
                {"role": "user", "content": f"""[Begin Example]
[Input] [Timestamp: 2023-11-22 09:30:00+00:00] OpenAI brings Sam Altman back as CEO less than a week after he was fired by board

Sam Altman was fired by the board of OpenAI on Friday. Following his ouster, hundreds of OpenAI employees, including co-founder and board member Ilya Sutskever, signed a letter demanding that remaining board members resign or they would leave. Altman has been reinstated by the board. Microsoft CEO Satya Nadella said Monday that Altman and others from OpenAI would be joining to start a "new advanced AI research team." [End Input]"""},
                {"role": "assistant", "content": """Sam Altman is CEO of OpenAI.
OpenAI brings Sam Altman back as CEO less than a week after Sam Altman was fired by board.
Following Sam Altman's ouster, hundreds of OpenAI employees, including Ilya Sutskever, signed a letter demanding that remaining board members resign or they would leave.
Ilya Sutskever is a co-founder and board member of OpenAI.
Satya Nadella said that Sam Altman and others from OpenAI would be joining to start a "new advanced AI research team."
Satya Nadella is the CEO of Microsoft."""},
            ])
        messages.extend([
            {"role": "user", "content": f"""[End Example]
[Begin Example]
[Input] {ts}: {context} [End Input]"""}
            ])
        if self.logging:
            with open(f"{self.save_fn}_fact_extraction.txt", "a") as wf:
                wf.write(f"\n\n[EXTRACTED FACTS:]\n")
        output_text, _ = self.query_lm_fn(
            messages,
            model_name=self.model_name,
            inference_kwargs={"max_tokens": min(self.context_length, count_tokens(self.tokenizer, context)),
                              "temperature": 0.0,
                              "repetition_penalty": 1.0})
        if self.logging:
            with open(f"{self.save_fn}_fact_extraction.txt", "a") as wf:
                wf.write(f"{output_text}\n\n")
        added_facts_nl = []
        if "No new facts" not in output_text.strip().strip("\""):
            added_facts_nl = output_text.split("\n")
        added_facts_uniq = []
        for fact in added_facts_nl:
            fact = fact.strip()
            if fact == "":
                continue
            if count_tokens(self.tokenizer, fact) > 500:
                # truncate fact
                truncated_fact = " ".join(self.tokenizer.tokenize(fact)[:500])
                if fact in fact_to_extracted_entities:
                    fact_to_extracted_entities[truncated_fact] = fact_to_extracted_entities[fact]
                fact = truncated_fact
            if fact not in added_facts_uniq:
                added_facts_uniq.append(fact)
        added_facts = []
        if self.retrieve_facts == "entity_similarity":
            # extract entities here
            messages = [
                {
                    "role": "system",
                    "content": """List all named entities that each fact pertains to, outputting a JSON of form:
{
    fact in text: entities fact is about
}
Output only JSON with no other text.

E.g.
Input:
Jeff Bezos is CEO of Amazon.
Alice is a lawyer.

Output:
{
    "Jeff Bezos is CEO of Amazon.": ["Jeff Bezos", "Amazon"],
    "Alice is a lawyer.": ["Alice"],
}"""
                },
                {
                    "role": "user",
                    "content": "\n".join(added_facts_uniq),
                },
            ]
            if len(added_facts_uniq) > 0:
                output_text, _ = self.query_lm_fn(
                    messages,
                    inference_kwargs={"temperature": 0.0, "max_tokens": min(self.context_length, 2 * count_tokens(self.tokenizer, "\n".join(added_facts_uniq))), "repetition_penalty": 1.0})
                try:
                    fact_to_extracted_entities = json.loads(output_text.strip())
                except:
                    fact_to_extracted_entities = {}
            else:
                fact_to_extracted_entities = {}

        for fact in added_facts_uniq:
            if fact.strip() == "":
                continue
            new_fact = Fact(sentence=fact.strip(), entities=self.extract_entities(
                fact.strip(),
                fact_to_extracted_entities=fact_to_extracted_entities,
            ))
            full_operation_list.append({
                "operation": "add_fact", "fact": new_fact,
                "new_contexts": [context],
                "new_source_ids": [source_id],
                "new_connected_facts": all_changed_facts,
            })
            added_facts.append(new_fact)
        
        return added_facts, full_operation_list, all_changed_facts, fact_to_extracted_entities


    def get_operations(self, context, ts, source_id, chunk_idx, initial_chunk_facts=None):
        """
        Extract operations on facts (reinforce, delete, update, add)
        """
        full_operation_list = []
        all_changed_facts = []
        if self.logging:
            with open(f"{self.save_fn}_fact_extraction.txt", "a") as wf:
                wf.write(f"[CONTEXT:]\n[Begin example]\n[Input] [Timestamp: {ts}] {context} [End Input]\n")
        if not self.save_as_facts:
            full_operation_list.append({
                "operation": "add_fact", "fact": Fact(sentence=context, entities=[]),
                "new_contexts": [context],
                "new_source_ids": [source_id],
                "new_connected_facts": [],
            })
            return full_operation_list, [], {}

        prompt_facts, all_relevant_facts = self.retrieve_facts_to_edit(
            context, ts, chunk_idx, initial_chunk_facts)

        extracted_facts = self.edit_retrieved_facts(context, ts, prompt_facts, all_relevant_facts)

        added_facts, full_operation_list, all_changed_facts, fact_to_extracted_entities = self.extract_new_facts(
            context, ts, source_id, chunk_idx, full_operation_list, all_changed_facts,
        )

        for fact in extracted_facts:
            relevant_fact = self.find_fact(fact, all_relevant_facts)
            current_source_ids = self.kb[relevant_fact][-1]['source_ids']
            if source_id in current_source_ids:
                # do not modify facts that were added/modified in the same context
                continue
            if extracted_facts[fact] == "make false":
                if relevant_fact is not None:
                    full_operation_list.append({
                        "operation": "update_fact", "fact": relevant_fact,
                        "update_direction": "decrease",
                        "new_contexts": [context],
                        "new_source_ids": [current_source_ids],
                        "new_connected_facts": added_facts,
                    })
                    all_changed_facts.append(relevant_fact)
            elif extracted_facts[fact] == "reinforce":
                if relevant_fact is not None:
                    full_operation_list.append({
                        "operation": "update_fact", "fact": relevant_fact,
                        "update_direction": "increase",
                        "new_contexts": [context],
                        "new_source_ids": [current_source_ids],
                        "new_connected_facts": added_facts,
                    })
                    all_changed_facts.append(relevant_fact)
                else:
                    if fact.strip() == "":
                        continue
                    new_fact = Fact(sentence=fact.strip(), entities=self.extract_entities(fact, context=context))
                    full_operation_list.append({
                        "operation": "add_fact", "fact": new_fact,
                        "new_contexts": [context],
                        "new_source_ids": [current_source_ids],
                        "new_connected_facts": added_facts,
                    })
                    all_changed_facts.append(new_fact)
            elif extracted_facts[fact].startswith("update"):
                fact_nl = ": ".join(extracted_facts[fact].split(": ")[1:]).strip()
                new_fact = Fact(sentence=fact_nl, entities=self.extract_entities(fact_nl, context=context))
                if relevant_fact != new_fact:
                    full_operation_list.append({
                        "operation": "rewrite_fact",
                        "fact": relevant_fact,
                        "new_fact": new_fact,
                        "new_contexts": [context],
                        "new_source_ids": [current_source_ids],
                        "new_connected_facts": added_facts,
                    })
                    all_changed_facts.append(relevant_fact)
        if self.logging:
            with open(f"{self.save_fn}_fact_extraction.txt", "a") as wf:
                wf.write(f"*===*\n\n")
        return full_operation_list, all_relevant_facts, fact_to_extracted_entities


    def retrieve(
        self, query: str=None, ts: int=-1,
        max_sentences_to_retrieve=None, relevance_threshold=None,
        min_sentences_to_retrieve=1, instruction=None, retrieve_facts=None
    ):
        """
        Retrieve relevant facts from kb (sorted by relevance score)

        mode = "oracle" (for rule-based) or "lm" (for lm-based)

        Returns:
            relevant_facts (List[namedtuple]): list of relevant facts
            relevance_scores (List[float]): list of relevance scores
            relevant_facts_ts_truthscores (List[Dict[str, float]]): dict of {"ts": float, "probability": float} for each relevant fact
            relevant_facts_nl (List[str]): list of relevant facts in natural language
        """
        if max_sentences_to_retrieve is None:
            max_sentences_to_retrieve = self.num_sentences_to_retrieve
        if relevance_threshold is None:
            relevance_threshold = 0  #self.relevance_threshold
        if retrieve_facts is None:
            retrieve_facts = self.retrieve_facts
        if retrieve_facts == "entity_similarity":
            relevance_threshold = min(0, relevance_threshold - 0.2)
        query_entities = None
        if retrieve_facts is None:
            return copy.deepcopy(self.all_facts), [
                self.kb[fact] for fact in self.all_facts
            ], [1.0 for _ in self.all_facts], [fact.sentence for fact in self.all_facts]
        elif self.index.ntotal == 0:
            return [], [], [], []
        elif "similarity" in retrieve_facts:
            if retrieve_facts == "similarity":
                fact_nl = query
            elif retrieve_facts == "instruct_similarity":
                if instruction is None:
                    fact_nl = f"Find relevant facts for answering the question. [SEP] {query}"
                else:
                    fact_nl = f"{instruction} [SEP] {query}"
            elif retrieve_facts == "entity_similarity":
                fact_nl = query
                query_entities = self.extract_entities(query)
            # take similarity of encodings
            input_embedding = self.embed_facts([Fact(sentence=fact_nl, entities=[])])
            relevance_scores, relevant_indexes = self.index.search(input_embedding, min(
                max_sentences_to_retrieve, self.index.ntotal))
            relevance_scores, relevant_indexes = relevance_scores.flatten(), relevant_indexes.flatten()  # de-batch
            relevant_facts = [
                self.all_facts[idx] for i, idx in enumerate(relevant_indexes) if relevance_scores[i] > relevance_threshold and self.contains_entity(
                    query_entities, self.all_facts[idx].entities)]
            relevance_scores_copy = [
                relevance_scores[i] for i, idx in enumerate(relevant_indexes) if relevance_scores[i] > relevance_threshold and self.contains_entity(
                    query_entities, self.all_facts[idx].entities)]

            if len(relevant_facts) < min_sentences_to_retrieve:
                relevant_facts = [self.all_facts[idx] for i, idx in enumerate(relevant_indexes)][:min_sentences_to_retrieve]
                relevance_scores = [relevance_scores[i] for i, idx in enumerate(relevant_indexes)][:min_sentences_to_retrieve]
            else:
                relevance_scores = relevance_scores_copy
            
            return relevant_facts, [self.kb[fact] for fact in relevant_facts], relevance_scores, [fact.sentence for fact in relevant_facts]
        else:
            raise NotImplementedError

    def convert_facts_to_nl_story(
        self, facts: List[namedtuple],
        facts_ts_truthscores: List[List[Dict[str, float]]],
        facts_truth_values: List[List[float]] = None,
        use_timestamps: bool = False,
        use_likelihood: bool = False,
        remove_untrue: bool = False,
        percent_of_context: float = 1.0,
        use_recent: bool = False,  # sliding window (most recent context)
    ):
        """
        convert facts to nl story:

        As of time t, <fact1>. <fact2>. <fact3>. ...
        As of time t+1, <fact1>. <fact2>. <fact3>. ...
        """
        nl_story = []
        token_count = 0
        for f, fact in enumerate(facts):
            fact_nl = f"{fact.sentence}"
            token_count += count_tokens(self.tokenizer, fact_nl)
            if remove_untrue and facts_ts_truthscores[f][-1]['probability'] < 0.5:
                continue
            if use_timestamps or use_likelihood:
                seen_ts = set()
                valid_ts_truthscores = []
                for t, ts_prob_tuple in enumerate(facts_ts_truthscores[f]):
                    ts = ts_prob_tuple["ts"].strftime('%Y-%m-%d')
                    if ts in seen_ts or facts_ts_truthscores[f][t].get('fact', fact.sentence) != fact.sentence:
                        # remove ones that are duplicates or have been rewritten
                        continue
                    seen_ts.add(ts)
                    valid_ts_truthscores.append(ts_prob_tuple)

                fact_nl += " ("
                for t, valid_ts_truthscore in enumerate(valid_ts_truthscores):
                    ts = ts_prob_tuple["ts"].strftime('%Y-%m-%d')
                    if use_likelihood:
                        fact_nl += f"probability {ts_prob_tuple['probability']:.3f} "
                    elif facts_truth_values is not None:
                        fact_nl += f"{facts_truth_values[f][t]} "
                    else:
                        if any(valid_ts_truthscores[t_sub]['probability'] > 0.5 for t_sub in range(t)) and ts_prob_tuple['probability'] < 0.5:
                            fact_nl = f"It is no longer the case that {fact.sentence} (True "
                        else:
                            fact_nl += f"{ts_prob_tuple['probability'] > 0.5} "
                    if use_timestamps:
                        fact_nl += f"at {ts}"
                    fact_nl += ", "
                fact_nl = fact_nl[:-2] + ")"
            if token_count > percent_of_context * float(self.context_length):
                if not use_recent:
                    break
                else:
                    while token_count > percent_of_context * float(self.context_length):
                        token_count -= count_tokens(self.tokenizer, nl_story[0])
                        token_count -= count_tokens(self.tokenizer, "\n*===*\n")
                        nl_story = nl_story[1:]
            nl_story.append(fact_nl)
            token_count += count_tokens(self.tokenizer, "\n*===*\n")
        nl_story = "\n*===*\n".join(nl_story)
        return nl_story


    def add_fact(self, fact: Fact, fact_prob_true: float, ts: int, source_ids: List[str], connected_facts: List[namedtuple], contexts: List[str], read_ts: int):
        if fact not in self.kb:
            self.kb[fact] = []
            # also extends all_facts
            self.add_fact_to_kb([fact])

        # insert new probability at the right position
        added_fact = False
        for i, fact_instance in enumerate(self.kb[fact]):
            if fact_instance["ts"] == ts:
                self.kb[fact][i] = {"ts": ts, "probability": fact_prob_true, "source_ids": source_ids, "connected_facts": connected_facts, "contexts": contexts, "read_ts": read_ts}
                added_fact = True
                break
        if not added_fact:
            # timestep not in list yet, add probability
            self.kb[fact].append({"ts": ts, "probability": fact_prob_true, "source_ids": source_ids, "connected_facts": connected_facts, "contexts": contexts, "read_ts": read_ts})
        # sort
        self.kb[fact] = sorted(self.kb[fact], key=lambda fact_instance: fact_instance["ts"])

        for entity in fact.entities:
            assert entity.name in self.all_entities
            entity.add_fact(fact)


    def stringify_kb(self):
        """
        stringify kb for saving
        """
        stringified_kb = {}
        for fact in self.kb:
            stringified_kb[fact.sentence] = []
            for fact_info in self.kb[fact]:
                stringified_kb[fact.sentence].append({
                    "ts": fact_info["ts"].strftime("%Y-%m-%d"),
                    "probability": fact_info["probability"],
                    "source_ids": fact_info["source_ids"],
                    "connected_facts": [connected_fact.sentence for connected_fact in fact_info["connected_facts"]],
                    "contexts": fact_info["contexts"],
                    "read_ts": fact_info["read_ts"].strftime("%Y-%m-%d"),
                })
        return stringified_kb

