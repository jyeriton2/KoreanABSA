import json
import os

import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from transformers import XLMRobertaModel, AutoTokenizer, AlbertModel
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW
from datasets import load_metric
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import copy

PADDING_TOKEN = 0   #1
S_OPEN_TOKEN = 2    #0
S_CLOSE_TOKEN = 3   #2  안 쓰지만 그냥 [SEP] token 으로 일단 설정하겠음.   albert kor : pad 0, unk 1, cls 2, sep 3, mask 4

entity_property_pair = [
    '제품 전체#일반', '제품 전체#가격', '제품 전체#디자인', '제품 전체#품질', '제품 전체#편의성', '제품 전체#인지도', '제품 전체#다양성',
    '본품#일반', '본품#디자인', '본품#품질', '본품#편의성', '본품#다양성', '본품#인지도', '본품#가격',
    '패키지/구성품#일반', '패키지/구성품#디자인', '패키지/구성품#품질', '패키지/구성품#편의성', '패키지/구성품#다양성', '패키지/구성품#가격',
    '브랜드#일반', '브랜드#가격', '브랜드#디자인', '브랜드#품질', '브랜드#인지도',
                    ]
label1_name_to_id = {'제품 전체': 0, '본품': 1, '패키지/구성품': 2, '브랜드': 3}
label2_name_to_id = {'일반': 0, '가격': 1, '디자인': 2, '품질': 3, '편의성': 4, '다양성': 5, '인지도': 6}

polarity_id_to_name = ['positive', 'negative', 'neutral']
polarity_name_to_id = {polarity_id_to_name[i]: i for i in range(len(polarity_id_to_name))}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

special_tokens_dict = { # 필요 없는 것 아닌가...
    'additional_special_tokens': ['&name&', '&affiliation&', '&social-security-num&', '&tel-num&', '&card-num&', '&bank-account&', '&num&', '&online-account&']
}


def jsonload(fname, encoding="utf-8"):
    with open(fname, encoding=encoding) as f:
        j = json.load(f)
    return j


# json 개체를 파일이름으로 깔끔하게 저장
def jsondump(j, fname):
    with open(fname, "w", encoding="UTF8") as f:
        json.dump(j, f, ensure_ascii=False)


# jsonl 파일 읽어서 list에 저장
def jsonlload(fname, encoding="utf-8"):
    json_list = []
    with open(fname, encoding=encoding) as f:
        for line in f.readlines():
            json_list.append(json.loads(line))
    return json_list


# json list를 jsonl 형태로 저장
def jsonldump(j_list, fname):
    f = open(fname, "w", encoding='utf-8')
    for json_data in j_list:
        f.write(json.dumps(json_data, ensure_ascii=False)+'\n')


def parse_args():
    parser = argparse.ArgumentParser(description="sentiment analysis")
    parser.add_argument(
        "--train_data", type=str, default="../data/NIKL_ABSA_2022_COMPETITION_v1.0/nikluge-sa-2022-train.jsonl",
        help="train file"
    )
    parser.add_argument(
        "--test_data", type=str, default="../data/NIKL_ABSA_2022_COMPETITION_v1.0/nikluge-sa-2022-test.jsonl",
        help="test file"
    )
    parser.add_argument(
        "--dev_data", type=str, default="../data/NIKL_ABSA_2022_COMPETITION_v1.0/nikluge-sa-2022-dev.jsonl",
        help="dev file"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001     #3e-6
    )
    parser.add_argument(
        "--eps", type=float, default=1e-8
    )
    parser.add_argument(
        "--do_train", action="store_true"
    )
    parser.add_argument(
        "--do_eval", action="store_true"
    )
    parser.add_argument(
        "--do_test", action="store_true"
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=20
    )
    parser.add_argument(
        "--base_model", type=str, default="../saved_model/models--smartmind--albert-kor-base-tweak\snapshots\9e831dab7dff34cf158f8c8dccbdda10841f659a/"    #"xlm-roberta-base"
    )
    parser.add_argument(
        "--entity_property_model_path", type=str, default="../saved_model/category_extraction/"
    )
    parser.add_argument(
        "--polarity_model_path", type=str, default="../saved_model/polarity_classification/"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output/default_path/"
    )
    parser.add_argument(
        "--do_demo", action="store_true"
    )
    parser.add_argument(
        "--max_len", type=int, default=90   # 256
    )
    parser.add_argument(
        "--classifier_hidden_size", type=int, default=1024  # 768
    )
    parser.add_argument(
        "--classifier_dropout_prob", type=int, default=0.1, help="dropout in classifier"
    )
    args = parser.parse_args()
    return args


class ConvBlock(nn.Module):
    def __init__(self, embedding_size, filters, kernel_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_size)   # layernorm : (batch, seq_length, embedding_size)
        # 다른 layer : (batch, channel, seq_length)
        self.mish = nn.Mish()
        self.conv1 = nn.Conv1d(embedding_size, filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.norm2 = nn.LayerNorm(filters)
        self.conv2 = nn.Conv1d(filters, filters, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, features):
        # features shape : batch, seq_length, embedding_size
        x = self.norm1(features)
        # transpose
        x = torch.transpose(x, 1, 2)  # batch, embedding_size, seq_length
        x = self.mish(x)
        x = self.conv1(x)
        # transpose
        x = torch.transpose(x, 1, 2)  # batch, seq_length, embedding_size
        x = self.norm2(x)
        # transpose
        x = torch.transpose(x, 1, 2)  # batch, embedding_size, seq_length
        x = self.mish(x)
        x = self.conv2(x)
        # transpose
        x = torch.transpose(x, 1, 2)  # batch, seq_length, embedding_size
        return x


class IdentityBlock(nn.Module):
    def __init__(self, embedding_size, kernel_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_size)   # layernorm : (batch, seq_length, embedding_size)
        # 다른 layer : (batch, channel, seq_length)
        self.pool = nn.AvgPool1d(kernel_size=3, stride=1, padding=1, count_include_pad=False)
        self.mish = nn.Mish()
        self.conv1 = nn.Conv1d(embedding_size, embedding_size, kernel_size=kernel_size, padding=kernel_size//2)
        self.norm2 = nn.BatchNorm1d(embedding_size)
        self.conv2 = nn.Conv1d(embedding_size, embedding_size, kernel_size=kernel_size, padding=kernel_size//2)

    def forward(self, features):
        # features shape : batch, seq_length, embedding_size
        x = self.norm1(features)
        # transpose
        x = torch.transpose(x, 1, 2)    # batch, embedding_size, seq_length
        x = self.pool(x)
        # transpose
        x = torch.transpose(x, 1, 2)    # batch, seq_length, embedding_size
        x = torch.add(features, x)
        # transpose
        x = torch.transpose(x, 1, 2)    # batch, embedding_size, seq_length
        x = self.mish(x)
        y = self.conv1(x)
        y = self.norm2(y)
        y = self.mish(y)
        y = self.conv2(y)
        y = self.norm2(y)
        x = torch.add(x, y)
        # transpose
        x = torch.transpose(x, 1, 2)    # batch, seq_length, embedding_size
        return x


class SimpleClassifier(nn.Module):
    def __init__(self, args, num_label):
        super().__init__()
        # self.dense = nn.Linear(args.classifier_hidden_size, args.classifier_hidden_size)
        self.conv_block1 = ConvBlock(768, 512, 9)
        self.identity_block1 = IdentityBlock(512, 5)
        self.conv_block2 = ConvBlock(512, 256, 13)
        self.identity_block2 = IdentityBlock(256, 5)
        self.conv_block3 = ConvBlock(256, 128, 17)
        self.identity_block3 = IdentityBlock(128, 5)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dense1 = nn.Linear(128, args.classifier_hidden_size)
        self.dropout = nn.Dropout(args.classifier_dropout_prob)
        self.dense2 = nn.Linear(args.classifier_hidden_size, args.classifier_hidden_size)
        self.mish = nn.Mish()
        self.output = nn.Linear(args.classifier_hidden_size, num_label)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        x = self.conv_block1(features)
        x = self.identity_block1(x)
        x = self.conv_block2(x)
        x = self.identity_block2(x)
        x = self.conv_block3(x)
        x = self.identity_block3(x)
        # transpose
        x = torch.transpose(x, 1, 2)  # batch, embedding_size, seq_length
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.mish(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.mish(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x


class AlBertBaseClassifier(nn.Module):
    def __init__(self, args, num_label, len_tokenizer):
        super(AlBertBaseClassifier, self).__init__()

        self.num_label = num_label
        self.albert = AlbertModel.from_pretrained(args.base_model)
        self.albert.resize_token_embeddings(len_tokenizer)

        self.labels_classifier = SimpleClassifier(args, self.num_label)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None
        )

        sequence_output = outputs[0]
        # pretrain shape output index 0 : batch, seq_length, embedding_size
        # pretrain shape output index 1 : batch, embedding_size
        logits = self.labels_classifier(sequence_output)

        loss = None

        if labels is not None:
            loss_fct = nn.BCELoss()
            loss = loss_fct(logits, labels)

        return loss, logits


class AlBertMultiGroupClassifier(nn.Module):
    def __init__(self, args, num_label1, num_label2, len_tokenizer):
        super(AlBertMultiGroupClassifier, self).__init__()

        self.num_label1 = num_label1
        self.num_label2 = num_label2
        self.albert = AlbertModel.from_pretrained(args.base_model)
        self.albert.resize_token_embeddings(len_tokenizer)

        self.label1s_classifier = SimpleClassifier(args, self.num_label1)
        self.label2s_classifier = SimpleClassifier(args, self.num_label2)

    def forward(self, input_ids, attention_mask, label1s=None, label2s=None):
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None
        )

        sequence_output = outputs[0]
        # pretrain shape output index 0 : batch, seq_length, embedding_size
        # pretrain shape output index 1 : batch, embedding_size
        label1_logits = self.label1s_classifier(sequence_output)
        label2_logits = self.label2s_classifier(sequence_output)

        loss = None

        if label1s is not None and label2s is not None:
            loss_fct = nn.BCELoss()
            loss1 = loss_fct(label1_logits, label1s)
            loss2 = loss_fct(label2_logits, label2s)
            loss = (loss1 + loss2) / 2

        return loss, label1_logits, label2_logits


def multi_label_one_hot(x, num_classes):
    return F.one_hot(torch.Tensor(x).to(torch.int64), num_classes=num_classes).sum(dim=0).to(torch.float).tolist()


polarity_count = 0
entity_property_count = 0


def tokenize_and_align_labels(tokenizer, form, annotations, max_len):

    global polarity_count
    global entity_property_count

    entity_property_data_dict = {
        'input_ids': [],
        'attention_mask': [],
        'label1': [],
        'label2': []
    }
    polarity_data_dict = {
        'input_ids': [],
        'attention_mask': [],
        'label': []
    }

    tokenized_data = tokenizer(form, padding='max_length', max_length=max_len, truncation=True)
    entity_property_label1 = []
    entity_property_label2 = []
    for annotation in annotations:
        entity_property = annotation[0]
        polarity = annotation[2]

        # # 데이터가 =로 시작하여 수식으로 인정된경우
        # if pd.isna(entity) or pd.isna(property):
        #     continue

        if polarity == '------------':
            continue

        _label1 = entity_property.split('#')[0]
        _label2 = entity_property.split('#')[1]
        entity_property_label1.append(label1_name_to_id[_label1])
        entity_property_label2.append(label2_name_to_id[_label2])

        # token 수 적은 것은 label 다르게 줄 것.
        polarity_count += 1
        polarity_data_dict['input_ids'].append(tokenized_data['input_ids'])
        polarity_data_dict['attention_mask'].append(tokenized_data['attention_mask'])

        if np.array(tokenized_data['attention_mask']).sum() < 6:
            polarity_label = [0.4] * len(polarity_id_to_name)
            polarity_label[polarity_name_to_id[polarity]] = 0.8
            polarity_data_dict['label'].append(polarity_label)
        else:
            polarity_data_dict['label'].append(
                multi_label_one_hot([polarity_name_to_id[polarity]], num_classes=len(polarity_id_to_name)))

    entity_property_label1 = list(set(entity_property_label1))
    entity_property_label2 = list(set(entity_property_label2))

    entity_property_count += 1
    entity_property_data_dict['input_ids'].append(tokenized_data['input_ids'])
    entity_property_data_dict['attention_mask'].append(tokenized_data['attention_mask'])
    entity_property_data_dict['label1'].append(
        multi_label_one_hot(entity_property_label1, num_classes=len(label1_name_to_id)))
    entity_property_data_dict['label2'].append(
         multi_label_one_hot(entity_property_label2, num_classes=len(label2_name_to_id)))

    return entity_property_data_dict, polarity_data_dict


def get_dataset(raw_data, tokenizer, max_len):
    input_ids_list = []
    attention_mask_list = []
    token_label1s_list = []
    token_label2s_list = []

    polarity_input_ids_list = []
    polarity_attention_mask_list = []
    polarity_token_labels_list = []

    for utterance in raw_data:
        entity_property_data_dict, polarity_data_dict = tokenize_and_align_labels(tokenizer, utterance['sentence_form'], utterance['annotation'], max_len)
        input_ids_list.extend(entity_property_data_dict['input_ids'])
        attention_mask_list.extend(entity_property_data_dict['attention_mask'])
        token_label1s_list.extend(entity_property_data_dict['label1'])
        token_label2s_list.extend(entity_property_data_dict['label2'])

        polarity_input_ids_list.extend(polarity_data_dict['input_ids'])
        polarity_attention_mask_list.extend(polarity_data_dict['attention_mask'])
        polarity_token_labels_list.extend(polarity_data_dict['label'])

    print('polarity_data_count: ', polarity_count)
    print('entity_property_data_count: ', entity_property_count)

    return TensorDataset(torch.tensor(input_ids_list), torch.tensor(attention_mask_list),
                         torch.tensor(token_label1s_list), torch.tensor(token_label2s_list)), \
           TensorDataset(torch.tensor(polarity_input_ids_list), torch.tensor(polarity_attention_mask_list),
                         torch.tensor(polarity_token_labels_list))


def evaluation(y_true, y_pred, label_len):
    count_list = [0]*label_len
    hit_list = [0]*label_len
    for i in range(len(y_true)):
        count_list[y_true[i]] += 1
        if y_true[i] == y_pred[i]:
            hit_list[y_true[i]] += 1
    acc_list = []

    for i in range(label_len):
        acc_list.append(hit_list[i]/count_list[i] if count_list[i] != 0 else 0)

    print(count_list)
    print(hit_list)
    print(acc_list)
    print('accuracy: ', (sum(hit_list) / sum(count_list)))
    print('macro_accuracy: ', sum(acc_list) / 3)
    # print(y_true)

    y_true = list(map(int, y_true))
    y_pred = list(map(int, y_pred))

    print('f1_score: ', f1_score(y_true, y_pred, average=None))
    print('f1_score_micro: ', f1_score(y_true, y_pred, average='micro'))
    print('f1_score_macro: ', f1_score(y_true, y_pred, average='macro'))


def train_sentiment_analysis(args=None, select='entity'):
    if not os.path.exists(args.entity_property_model_path):
        os.makedirs(args.entity_property_model_path)
    if not os.path.exists(args.polarity_model_path):
        os.makedirs(args.polarity_model_path)

    if select == 'entity':
        model_saved_dir = args.entity_property_model_path
    elif select == 'polarity':
        model_saved_dir = args.polarity_model_path
    else:
        raise Exception('Do not select sentiment model.')

    print('train_sentiment_analysis')
    print('entity property model would be saved at ', args.entity_property_model_path)
    print('polarity model would be saved at ', args.polarity_model_path)

    print('loading train data')
    train_data = jsonlload(args.train_data)
    dev_data = jsonlload(args.dev_data)

    print('tokenizing train data')
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print('We have added', num_added_toks, 'tokens')

    entity_property_train_data, polarity_train_data = get_dataset(train_data, tokenizer, args.max_len)
    entity_property_dev_data, polarity_dev_data = get_dataset(dev_data, tokenizer, args.max_len)

    if select == 'entity':
        entity_property_aug_data, _ = get_dataset(
            jsonlload(str(args.train_data).replace('train.jsonl', 'aug-entity_property.jsonl')),
            tokenizer, args.max_len)
        entity_property_train_data = ConcatDataset([entity_property_train_data, entity_property_aug_data, entity_property_dev_data])
        train_dataloader = DataLoader(entity_property_train_data, shuffle=True,
                                                      batch_size=args.batch_size)
        dev_dataloader = DataLoader(entity_property_dev_data, shuffle=True,
                                                    batch_size=args.batch_size)

        print('loading model')
        model = AlBertMultiGroupClassifier(args, len(label1_name_to_id), len(label2_name_to_id), len(tokenizer))
        model.to(device)
        print('end loading')
    elif select == 'polarity':
        _, polarity_aug_data = get_dataset(
            jsonlload(str(args.train_data).replace('train.jsonl', 'aug-polarity.jsonl')),
            tokenizer, args.max_len)
        polarity_train_data = ConcatDataset([polarity_train_data, polarity_aug_data, polarity_dev_data])
        train_dataloader = DataLoader(polarity_train_data, shuffle=True,
                                               batch_size=args.batch_size)
        dev_dataloader = DataLoader(polarity_dev_data, shuffle=True,
                                             batch_size=args.batch_size)

        print('loading model')
        model = AlBertBaseClassifier(args, len(polarity_id_to_name), len(tokenizer))
        model.to(device)
        print('end loading')

    # entity_property_model_optimizer_setting
    epochs = args.num_train_epochs
    total_steps = epochs * len(train_dataloader)
    max_grad_norm = 1.0
    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.eps
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    epoch_step = 0

    for _ in trange(epochs, desc="Epoch"):
        model.train()
        epoch_step += 1

        # train
        total_loss = 0

        if select == 'entity':
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_label1s, b_label2s = batch

                model.zero_grad()

                loss, _logits1, _logits2 = model(b_input_ids, b_input_mask, b_label1s, b_label2s)

                loss.backward()

                total_loss += loss.item()
                # print('batch_loss: ', loss.item())

                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
                scheduler.step()
        elif select == 'polarity':
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                model.zero_grad()

                loss, _ = model(b_input_ids, b_input_mask, b_labels)

                loss.backward()

                total_loss += loss.item()
                # print('batch_loss: ', loss.item())

                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
                scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        print("Entity_Property_Epoch: ", epoch_step)
        print("Average train loss: {}".format(avg_train_loss))

        model_saved_path = model_saved_dir + 'saved_model_epoch_' + str(epoch_step) + '.pt'
        torch.save(model.state_dict(), model_saved_path)

        if args.do_eval:
            model.eval()
            if select == 'entity':
                pred1_list = []
                pred2_list = []
                label1_list = []
                label2_list = []

                for batch in train_dataloader:        # dev_dataloader
                    batch = tuple(t.to(device) for t in batch)
                    b_input_ids, b_input_mask, b_label1s, b_label2s = batch

                    with torch.no_grad():
                        # loss, logits = model(b_input_ids, b_input_mask, b_labels)
                        loss, label1_logits, label2_logits = model(b_input_ids, b_input_mask)

                    pred1_list.extend(torch.argmax(label1_logits, dim=-1))
                    pred2_list.extend(torch.argmax(label2_logits, dim=-1))
                    label1_list.extend(torch.argmax(b_label1s, dim=-1))
                    label2_list.extend(torch.argmax(b_label2s, dim=-1))

                evaluation(label1_list, pred1_list, len(label1_name_to_id))
                evaluation(label2_list, pred2_list, len(label2_name_to_id))

            elif select == 'polarity':
                pred_list = []
                label_list = []

                for batch in train_dataloader:        # dev_dataloader
                    batch = tuple(t.to(device) for t in batch)
                    b_input_ids, b_input_mask, b_labels = batch

                    with torch.no_grad():
                        # loss, logits = model(b_input_ids, b_input_mask, b_labels)
                        loss, logits = model(b_input_ids, b_input_mask)

                    predictions = torch.argmax(logits, dim=-1)
                    pred_list.extend(predictions)
                    b_labels = torch.argmax(b_labels, dim=-1)
                    label_list.extend(b_labels)
                evaluation(label_list, pred_list, len(polarity_id_to_name))

    print("training is done")


def predict_from_korean_form(tokenizer, ce_model, pc_model, data):

    ce_model.to(device)
    ce_model.eval()
    count = 0
    for sentence in data:
        form = sentence['sentence_form']
        sentence['annotation'] = []
        count += 1
        if type(form) != str:
            print("form type is arong: ", form)
            continue

        tokenized_data = tokenizer(form, padding='max_length', max_length=90, truncation=True)

        input_ids = torch.tensor([tokenized_data['input_ids']]).to(device)
        attention_mask = torch.tensor([tokenized_data['attention_mask']]).to(device)
        with torch.no_grad():
            _, label1_logits, label2_logits = ce_model(input_ids, attention_mask)

        with torch.no_grad():
            _, pc_logits = pc_model(input_ids, attention_mask)

        # for pair in entity_property_pair:
        #     # ce_predictions = torch.argmax(ce_logits, dim = -1)
        #     ce1_predictions = label1_logits[0][label1_name_to_id[pair.split('#')[0]]]
        #     ce2_predictions = label2_logits[0][label2_name_to_id[pair.split('#')[1]]]
        #
        #     ce_result = 'True' if ce1_predictions > 0.8 and ce2_predictions > 0.8 else 'False'
        #
        #     if ce_result == 'True':
        #         # pc_predictions = torch.argmax(pc_logits, dim=-1)
        #         # pc_result = polarity_id_to_name[pc_predictions[0]]
        #
        #         # sentence['annotation'].append([pair, pc_result])
        #         c = 0
        #         for pc_prediction in pc_logits[0]:
        #             if pc_prediction > 0.8:
        #                 pc_result = polarity_id_to_name[c]
        #                 sentence['annotation'].append([pair, pc_result])
        #             c += 1
        #
        # if len(sentence['annotation']) == 0:
        #     pair = list(label1_name_to_id.keys())[int(torch.argmax(label1_logits, dim=-1))] + '#' + list(label2_name_to_id.keys())[int(torch.argmax(label2_logits, dim=-1))]
        #     pc_result = polarity_id_to_name[int(torch.argmax(pc_logits, dim=-1))]
        #     sentence['annotation'].append([pair, pc_result])
        #     print(sentence)
        pair = list(label1_name_to_id.keys())[int(torch.argmax(label1_logits, dim=-1))] + '#' + \
               list(label2_name_to_id.keys())[int(torch.argmax(label2_logits, dim=-1))]
        pc_result = polarity_id_to_name[int(torch.argmax(pc_logits, dim=-1))]
        sentence['annotation'].append([pair, pc_result])

    return data


def evaluation_f1(true_data, pred_data):
    true_data_list = true_data
    pred_data_list = pred_data

    ce_eval = {
        'TP': 0,
        'FP': 0,
        'FN': 0,
        'TN': 0
    }

    pipeline_eval = {
        'TP': 0,
        'FP': 0,
        'FN': 0,
        'TN': 0
    }

    for i in range(len(true_data_list)):

        # TP, FN checking
        is_ce_found = False
        is_pipeline_found = False
        for y_ano in true_data_list[i]['annotation']:
            y_category = y_ano[0]
            y_polarity = y_ano[2]

            for p_ano in pred_data_list[i]['annotation']:
                p_category = p_ano[0]
                p_polarity = p_ano[1]

                if y_category == p_category:
                    is_ce_found = True
                    if y_polarity == p_polarity:
                        is_pipeline_found = True

                    break

            if is_ce_found is True:
                ce_eval['TP'] += 1
            else:
                ce_eval['FN'] += 1

            if is_pipeline_found is True:
                pipeline_eval['TP'] += 1
            else:
                pipeline_eval['FN'] += 1

            is_ce_found = False
            is_pipeline_found = False

        # FP checking
        for p_ano in pred_data_list[i]['annotation']:
            p_category = p_ano[0]
            p_polarity = p_ano[1]

            for y_ano in true_data_list[i]['annotation']:
                y_category = y_ano[0]
                y_polarity = y_ano[2]

                if y_category == p_category:
                    is_ce_found = True
                    if y_polarity == p_polarity:
                        is_pipeline_found = True

                    break

            if is_ce_found is False:
                ce_eval['FP'] += 1

            if is_pipeline_found is False:
                pipeline_eval['FP'] += 1

    ce_precision = ce_eval['TP'] / (ce_eval['TP'] + ce_eval['FP'])
    ce_recall = ce_eval['TP'] / (ce_eval['TP'] + ce_eval['FN'])

    ce_result = {
        'Precision': ce_precision,
        'Recall': ce_recall,
        'F1': 2 * ce_recall * ce_precision / (ce_recall + ce_precision)
    }

    pipeline_precision = pipeline_eval['TP'] / (pipeline_eval['TP'] + pipeline_eval['FP'])
    pipeline_recall = pipeline_eval['TP'] / (pipeline_eval['TP'] + pipeline_eval['FN'])

    pipeline_result = {
        'Precision': pipeline_precision,
        'Recall': pipeline_recall,
        'F1': 2 * pipeline_recall * pipeline_precision / (pipeline_recall + pipeline_precision)
    }

    return {
        'category extraction result': ce_result,
        'entire pipeline result': pipeline_result
    }


def test_sentiment_analysis(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    test_data = jsonlload(args.test_data)

    entity_property_test_data, polarity_test_data = get_dataset(test_data, tokenizer, args.max_len)
    entity_property_test_dataloader = DataLoader(entity_property_test_data, shuffle=True,
                                                 batch_size=args.batch_size)

    # polarity_test_dataloader = DataLoader(polarity_test_data, shuffle=True,
    #                                       batch_size=args.batch_size)

    model = AlBertMultiGroupClassifier(args, len(label1_name_to_id), len(label2_name_to_id), len(tokenizer))
    model.load_state_dict(torch.load(args.entity_property_model_path, map_location=device))
    model.to(device)
    model.eval()

    polarity_model = AlBertBaseClassifier(args, len(polarity_id_to_name), len(tokenizer))
    polarity_model.load_state_dict(torch.load(args.polarity_model_path, map_location=device))
    polarity_model.to(device)
    polarity_model.eval()

    pred_data = predict_from_korean_form(tokenizer, model, polarity_model, copy.deepcopy(test_data))

    # jsondump(pred_data, './pred_data.json')
    jsonldump(pred_data, './pred_data.json')
    # pred_data = jsonload('./pred_data.json')

    print('F1 result: ', evaluation_f1(test_data, pred_data))

    pred_list = []
    label_list = []
    print('polarity classification result')
    for batch in entity_property_test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            # loss, logits = polarity_model(b_input_ids, b_input_mask, b_labels)
            loss, logits = polarity_model(b_input_ids, b_input_mask)

        predictions = torch.argmax(logits, dim=-1)
        pred_list.extend(predictions)
        b_labels = torch.argmax(b_labels, dim=-1)
        label_list.extend(b_labels)

    evaluation(label_list, pred_list, len(polarity_id_to_name))


if __name__ == '__main__':
    args = parse_args()
    args.do_eval = True
    train_sentiment_analysis(args, select='entity')   # entity polarity
    # args.entity_property_model_path = "../saved_model/category_extraction/saved_model_epoch_15.pt"
    # args.polarity_model_path = "../saved_model/polarity_classification/saved_model_epoch_20_old.pt"
    # test_sentiment_analysis(args)
