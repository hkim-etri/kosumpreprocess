import glob
import json

import torch
from torch.utils import data

from typing import List

from nltk import sent_tokenize, word_tokenize
from tqdm import tqdm
from config import Config

from transformers import AutoTokenizer
from kobart import get_kobart_tokenizer

config = Config()


def pad_list(input_list: list, pad_value: int = -1):
    """Pad list to convert tensor
    """
    max_len = max(list(map(len, input_list)))
    for idx in range(len(input_list)):
        input_list[idx].extend([pad_value] * (max_len - len(input_list[idx])))


class LoadDataset():
    def __init__(self, load_train=True):

        # Load tokenizer.
        self.retriever_tokenizer = AutoTokenizer.from_pretrained(config.retriever_name_or_path)
        self.generator_tokenizer = get_kobart_tokenizer()

        # Load dataset.
        print('----- Loading data -----')
        if load_train:
            self.train_set = DYLE_KO_Dataset(
                'train',
                retriever_tokenizer=self.retriever_tokenizer,
                generator_tokenizer=self.generator_tokenizer
            )
        self.val_set = DYLE_KO_Dataset(
            'val',
            retriever_tokenizer=self.retriever_tokenizer,
            generator_tokenizer=self.generator_tokenizer
        )
        self.test_set = DYLE_KO_Dataset(
            'test',
            retriever_tokenizer=self.retriever_tokenizer,
            generator_tokenizer=self.generator_tokenizer
        )



class DYLE_KO_Dataset(data.Dataset):
    """The Ko dataset."""

    def __init__(self, mode, retriever_tokenizer, generator_tokenizer):
        super().__init__()
        
        self.mode = mode
        self.retriever_tokenizer = retriever_tokenizer
        self.generator_tokenizer = generator_tokenizer

        self.features = list()
        self.root = config.dataset[0]
        self.cached_dataset = f"{self.root}/{config.cached_dataset}_{self.mode}"

        file_names = glob.glob(f'{self.root}/{mode}/*.json')
        self.file_names = file_names

        for f in self.file_names:
            print(f)

        self.get_references()

        self.features = self.read_dialogue_summarization()
        print("Saving features into cached file {}".format(self.cached_dataset))
        torch.save(self.features, self.cached_dataset)


    def get_references(self):
        self.eval_references = []

        self.eval_file_query_names = {'total':[], 'topic':[], 'speaker':[]}
        self.summary_type = ['total', 'topic', 'speaker']

        for file_name in self.file_names:
            file_path = file_name

            with open(file_path) as f:
                session = json.load(f)
                dict_dialouge = dict()

                new_dialoue = []
                for s_i, sentence in enumerate(session['dialogue']):
                    if sentence['sentence'] == None:
                        print("error dialouge : {}".format(sentence))
                        continue
                    if isinstance(sentence['sentence_id'], int):
                        new_dialoue.append(sentence)
                    else:
                        print("error dialouge : {}".format(sentence))
                session['dialogue'] = new_dialoue

                for s_i, sentence in enumerate(session['dialogue']):
                    if sentence['sentence_id'] in dict_dialouge:
                        continue

                    sentence['s_i'] = s_i
                    dict_dialouge[sentence['sentence_id']] = sentence

                for query_type in self.summary_type:
                    for idx, pair in enumerate(session['{}_summary'.format(query_type)]):
                        eval_reference = "\n".join(sent_tokenize(' '.join(word_tokenize(pair['{}_asummary'.format(query_type)].lower()))))

                        self.eval_references.append(eval_reference)
                        if query_type == 'topic':
                            topic = pair['topic'].strip()
                        elif query_type == 'speaker':
                            topic = "{} : {}".format(pair['speaker'].strip(), pair['speaker_topic'].strip())
                        else:
                            topic = pair['{}_topic'.format(query_type)].strip()

                        if query_type == 'total':
                            if 'speaker_sentence_ids' in pair:
                                evidence_ids = pair['speaker_sentence_ids']
                            else:
                                evidence_ids = pair['total_sentence_ids']
                        else:
                            evidence_ids = pair['{}_sentence_ids'.format(query_type)]

                        if isinstance(evidence_ids, int):
                            evidence_ids = [evidence_ids]

                        new_evidence_ids = []
                        for e_i in evidence_ids:
                            if e_i not in dict_dialouge:
                                print(" - Error(evidence) : {}".format(e_i))
                                continue

                            new_evidence_ids.append(dict_dialouge[e_i]['s_i'])

                        self.eval_file_query_names[query_type].append({"file_name":file_name, "topic":topic, "query_type":"{}".format(query_type),
                                                           "query_i":idx,
                                                           "asummary": eval_reference,
                                                           "esummary": pair['{}_esummary'.format(query_type)],
                                                           "evidence_ids":new_evidence_ids})

    def read_dialogue_summarization(self):
        print(("Reading dialogue as turns from {}/{}".format(self.root, self.mode)))
        features = []
        self.all_info = {}

        file_i = -1
        for file_name in tqdm(self.file_names):
            file_i += 1
            assert file_name not in self.all_info

            file_path = file_name
            print(file_path)

            dict_dialogue = dict()
            dialogue: List[str] = []

            with open(file_path) as f:
                session = json.load(f)

            new_dialoue = []
            for s_i, sentence in enumerate(session['dialogue']):
                if sentence['sentence'] == None:
                    continue
                if isinstance(sentence['sentence_id'], int):
                    new_dialoue.append(sentence)
            session['dialogue'] = new_dialoue

            # Make sent_mapping idx, dialogue append
            for s_i, sentence in enumerate(session['dialogue']):
                if sentence['sentence_id'] in dict_dialogue:
                    continue
                sentence['s_i'] = s_i
                dict_dialogue[sentence['sentence_id']] = sentence
                dialogue.append(sentence['speaker'].lower().strip() + ': ' + sentence['sentence'].strip().replace('\n',' ').replace('n/',' ').replace('l/', ' '))

            query_info = {'total': [], 'topic': [], 'speaker': []}

            for query_type in self.summary_type:
                for idx, pair in enumerate(session['{}_summary'.format(query_type)]):
                    eval_reference = "\n".join(
                        sent_tokenize(' '.join(word_tokenize(pair['{}_asummary'.format(query_type)].lower()))))

                    topic = pair['topic'].strip()
                    evidence_ids = pair['sentence_ids']

                    if isinstance(evidence_ids, int):
                        evidence_ids = [evidence_ids]

                    new_evidence_ids = []
                    for e_i in evidence_ids:
                        if e_i not in dict_dialogue:
                            print(" - Error(evidence) : {}".format(e_i))
                            continue
                        new_evidence_ids.append(dict_dialogue[e_i]['s_i'])

                    query_info[query_type].append(
                        {"file_name": file_name, "topic": topic, "query_type": "{}".format(query_type),
                         "query_i": idx,
                         "asummary": eval_reference,
                         "esummary": pair['{}_esummary'.format(query_type)],
                         "evidence_ids": new_evidence_ids})

            self.all_info[file_name] = {'dialogue':dialogue, 'query_info':query_info}

            queries: List[str] = []
            summaries: List[str] = []
            oracle_cls_ids_list: list = []

            # get text from querys
            for type in self.summary_type:
                for query_idx ,query_info in enumerate(self.all_info[file_name]['query_info'][type]):
                    if len(query_info['evidence_ids']) == 0:
                        continue
                    queries.append(query_info['topic'])
                    summaries.append(query_info['asummary'])
                    oracle_cls_ids_list.append([query_info['evidence_ids']])

            assert len(queries) == len(summaries) == len(oracle_cls_ids_list)

            for query, summary, oracle_cls_ids in zip(queries, summaries, oracle_cls_ids_list):
                retriever_inputs: dict = self.tokenize_retriever(dialogue, query, oracle_cls_ids)
                generator_inputs: dict = self.tokenize_generator(dialogue, query, summary)
                features.append((retriever_inputs, generator_inputs))

        return features
        

    def tokenize_retriever(self, text, query, oracle_cls_ids):
        tokenized_query = self.retriever_tokenizer.encode(query)
        tokenized_sentence_list = [self.retriever_tokenizer.encode(turn) for turn in text]

        input_ids_list:List[List[int]] = []
        cls_ids:List[int] = []
        oracle_list: List[List[int]] = []
        idx_offset = 0
        turn_id = 0
        list_id = 0

        # make chunks with sentences
        while turn_id < len(tokenized_sentence_list) and list_id < config.max_chunks:
            # text
            input_ids = []

            # Append query
            input_ids.extend(tokenized_query)

            # Append each dialogue until chunk is (almost)full
            while turn_id < len(tokenized_sentence_list):
                tokenized_sentence = tokenized_sentence_list[turn_id]
                # exceed max length
                if len(input_ids) + len(tokenized_sentence) > config.max_retrieval_len:
                    # stop at first
                    if len(input_ids) == len(tokenized_query):
                        tokenized_sentence = tokenized_sentence[:config.max_retrieval_len - len(input_ids)]
                    else:
                        break
                input_ids.extend(tokenized_sentence)
                
                # Inform end position of each sentence
                cls_ids.append(len(input_ids) - 1 + idx_offset)
                turn_id += 1

            # Append pad token
            num_pad = config.max_retrieval_len - len(input_ids)
            input_ids.extend([self.retriever_tokenizer.pad_token_id] * num_pad)

            # Save
            input_ids_list.append(input_ids)
            idx_offset += config.max_retrieval_len
            list_id += 1

        for oracle_cls in oracle_cls_ids:
            oracle = [
                oracle_id for oracle_id in oracle_cls
                if oracle_id < turn_id and len(text[oracle_id].split(" ")) > 3
            ]
            oracle_list.append(oracle)
        
        # Fill padding
        pad_list(oracle_list)
        retriever_inputs = {
            'input_ids': input_ids_list,
            'cls_ids': cls_ids,
            'oracle': oracle_list,
        }
        
        return retriever_inputs



    def tokenize_generator(self, text, query, summary):
        context_input_ids = []
        labels = None
        context_attention_mask = []

        for turn_id in range(len(text)):
            text_turn = text[turn_id]

            input_dict = self.generator_tokenizer.prepare_seq2seq_batch(
                src_texts=text_turn + " // " + query,
                tgt_texts=summary,
                max_length=config.max_source_len,
                max_target_length=config.max_target_len,
                padding="max_length",
                truncation=True,
            )
            context_attention_mask.append(input_dict.attention_mask)
            context_input_ids.append(input_dict.input_ids)
            if labels is None:
                labels = input_dict.labels
            else:
                assert labels == input_dict.labels

        generator_inputs = {
            'context_input_ids': context_input_ids,
            'context_attention_mask': context_attention_mask,
            'labels': labels
        }
        if labels is None:
            raise ValueError(text)

        return generator_inputs
