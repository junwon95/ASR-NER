from __future__ import absolute_import, division, print_function, unicode_literals
import json
import pickle
import torch
from gluonnlp.data import SentencepieceTokenizer
from NER.model.net import KobertCRF
from NER.data_utils.utils import Config
from NER.data_utils.vocab_tokenizer import Tokenizer
from NER.data_utils.pad_sequence import keras_pad_fn
from pathlib import Path

import sys
sys.path.append('NER')


class DecoderFromNamedEntitySequence():
    def __init__(self, tokenizer, index_to_ner):
        self.tokenizer = tokenizer
        self.index_to_ner = index_to_ner

    def __call__(self, list_of_input_ids, list_of_pred_ids, input_text):

        input_token = self.tokenizer.decode_token_ids(list_of_input_ids)[0]
        pred_ner_tag = [self.index_to_ner[pred_id] for pred_id in list_of_pred_ids[0]]

        # ----------------------------- parsing list_of_ner_word ----------------------------- #
        list_of_ner_word = []
        entity_word, entity_tag, prev_entity_tag = "", "", ""
        for i, pred_ner_tag_str in enumerate(pred_ner_tag):
            if "B-" in pred_ner_tag_str:
                entity_tag = pred_ner_tag_str[-3:]

                if prev_entity_tag != entity_tag and prev_entity_tag != "":
                    list_of_ner_word.append(
                        {"word": entity_word.replace("▁", " "), "tag": prev_entity_tag, "prob": None})

                entity_word = input_token[i]
                prev_entity_tag = entity_tag
            elif "I-" + entity_tag in pred_ner_tag_str:
                entity_word += input_token[i]
            else:
                if entity_word != "" and entity_tag != "":
                    list_of_ner_word.append({"word": entity_word.replace("▁", " "), "tag": entity_tag, "prob": None})
                entity_word, entity_tag, prev_entity_tag = "", "", ""

        # ----------------------------- parsing decoding_ner_sentence ----------------------------- #
        decoding_ner_sentence = ""
        is_prev_entity = False
        prev_entity_tag = ""
        is_there_B_before_I = False
        token_str_len_sum = 0
        for i, (token_str, pred_ner_tag_str) in enumerate(zip(input_token, pred_ner_tag)):
            if i == 0 or i == len(pred_ner_tag) - 1:  # remove [CLS], [SEP]
                continue
            token_str = token_str.replace('▁', ' ')  # '▁' 토큰을 띄어쓰기로 교체

            """
            print("input_token: ",input_token[i])
            print("pred_ner_tag_str: ",pred_ner_tag_str)
            print("pred_ner_tag: ",pred_ner_tag[i])
            print("prev_entity_tag: ",prev_entity_tag)
            print("")
            """

            # UNK 태그라면 그 글자에 해당하는 것을 input_text에서 가져온다.
            if token_str == '[UNK]':
                decoding_ner_sentence += input_text[token_str_len_sum]
                token_str = ""
                token_str_len_sum = token_str_len_sum + 1
            token_str_len_sum += len(token_str.strip())

            # 이 부분 공부
            if 'B-' in pred_ner_tag_str:
                if is_prev_entity is True:
                    if pred_ner_tag_str[-3:] in ["PER", "LOC", "ORG"]:
                        decoding_ner_sentence += ']'

                # token_str is not None and token_str[0]== ' '
                if token_str == "":
                    if pred_ner_tag_str[-3:] == "PER":
                        decoding_ner_sentence += '|' + token_str
                    elif pred_ner_tag_str[-3:] == "LOC":
                        decoding_ner_sentence += '$' + token_str
                    elif pred_ner_tag_str[-3:] == "ORG":
                        decoding_ner_sentence += '{' + token_str
                    else:
                        decoding_ner_sentence += '' + token_str
                elif token_str[0] == ' ':
                    token_str = list(token_str)
                    if pred_ner_tag_str[-3:] == "PER":
                        token_str[0] = ' |'
                    elif pred_ner_tag_str[-3:] == "LOC":
                        token_str[0] = ' $'
                    elif pred_ner_tag_str[-3:] == "ORG":
                        token_str[0] = ' {'
                    else:
                        token_str[0] = ' '
                    token_str = ''.join(token_str)
                    decoding_ner_sentence += token_str
                else:
                    if pred_ner_tag_str[-3:] == "PER":
                        decoding_ner_sentence += '|' + token_str
                    elif pred_ner_tag_str[-3:] == "LOC":
                        decoding_ner_sentence += '$' + token_str
                    elif pred_ner_tag_str[-3:] == "ORG":
                        decoding_ner_sentence += '{' + token_str
                    else:
                        decoding_ner_sentence += '' + token_str
                is_prev_entity = True
                prev_entity_tag = pred_ner_tag_str[-3:]  # 첫번째 예측을 기준으로 하겠음, B-PER이면 PER만 가지게
                is_there_B_before_I = True

            elif 'I-' in pred_ner_tag_str:
                # 만약 I-태그 상태로 문장이 끝나면 닫는 대괄호를 추가해준다.
                if token_str_len_sum == len(input_text):
                    decoding_ner_sentence += token_str + ']'
                else:
                    decoding_ner_sentence += token_str

                if is_there_B_before_I is True:  # I가 나오기전에 B가 있어야하도록 체크
                    is_prev_entity = True
            else:
                # 'O'일때
                if is_prev_entity is True:
                    if prev_entity_tag in ["PER", "LOC", "ORG"]:
                        decoding_ner_sentence += ']' + token_str
                    else:
                        decoding_ner_sentence += token_str
                    is_prev_entity = False
                    is_there_B_before_I = False
                else:
                    decoding_ner_sentence += token_str

        return list_of_ner_word, decoding_ner_sentence


def NER():
    model_dir = Path('NER/experiments/base_model_with_crf')
    model_config = Config(json_path=model_dir / 'config.json')

    # load vocab & tokenizer
    tok_path = "NER/ptr_lm_model/tokenizer_78b3253a26.model"
    ptr_tokenizer = SentencepieceTokenizer(tok_path)

    with open(model_dir / "vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)

    tokenizer = Tokenizer(vocab=vocab, split_fn=ptr_tokenizer, pad_fn=keras_pad_fn, maxlen=model_config.maxlen)

    # load ner_to_index.json
    with open(model_dir / "ner_to_index.json", 'rb') as f:
        ner_to_index = json.load(f)
        index_to_ner = {v: k for k, v in ner_to_index.items()}

    # model
    model = KobertCRF(config=model_config, num_classes=len(ner_to_index), vocab=vocab)

    # load
    model_dict = model.state_dict()
    checkpoint = torch.load("NER/experiments/base_model_with_crf/best-epoch-14-step-1250-acc-0.960.bin",
                            map_location=torch.device('cpu'))
    # checkpoint = torch.load("./experiments/base_model_with_crf_val/best-epoch-12-step-1000-acc-0.960.bin", map_location=torch.device('cpu'))
    convert_keys = {}
    for k, v in checkpoint['model_state_dict'].items():
        new_key_name = k.replace("module.", '')
        if new_key_name not in model_dict:
            print("{} is not int model_dict".format(new_key_name))
            continue
        convert_keys[new_key_name] = v

    model.load_state_dict(convert_keys, strict=False)
    model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    # [UNK]는 tokenizer을 고쳐야한다.
    decoder_from_res = DecoderFromNamedEntitySequence(tokenizer=tokenizer, index_to_ner=index_to_ner)

    # 수정 시작
    fin = open("OUTPUTS/ASR-OUT/transcripts.txt", 'r', encoding='cp949')
    fout = open("OUTPUTS/NER-OUT/final_output.txt", 'w', encoding='UTF8')
    temp = 0
    while (True):
        input_text = fin.readline()
        if not input_text: break
        input_text = ''.join([i for i in input_text if not i.isdigit()])
        input_text = input_text.strip()

        list_of_input_ids = tokenizer.list_of_string_to_list_of_cls_sep_token_ids([input_text])
        x_input = torch.tensor(list_of_input_ids).long()
        list_of_pred_ids, confidence = model(x_input)
        list_of_ner_word, decoding_ner_sentence = decoder_from_res(list_of_input_ids=list_of_input_ids,
                                                                   list_of_pred_ids=list_of_pred_ids,
                                                                   input_text=input_text.replace(" ", ""))

        fout.write(decoding_ner_sentence + "\n")

        temp = temp + 1
        if temp % 500 == 0:
            print(temp)
        # print("list_of_ner_word:",list_of_ner_word)
    fin.close()
    fout.close()