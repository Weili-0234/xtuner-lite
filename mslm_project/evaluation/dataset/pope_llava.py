import json
import os

import pandas as pd
import torch
from mmengine.dist import master_only
from PIL import Image
from torch.utils.data import Dataset

from xtuner.dataset.utils import expand2square
from xtuner.registry import BUILDER
from xtuner.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX


def process_punctuation(inText):
    import re
    outText = inText
    punct = [
        ';', r'/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-',
        '>', '<', '@', '`', ',', '?', '!'
    ]
    commaStrip = re.compile(r'(\d)(,)(\d)')  # noqa: W605
    periodStrip = re.compile(r'(?!<=\d)(\.)(?!\d)')  # noqa: W605
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(
                commaStrip, inText) is not None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub('', outText, re.UNICODE)
    return outText


def YOrN_Extraction(output):
    s = output.lower()
    words = process_punctuation(s).split()
    if 'yes' in words and 'no' not in words:
        return 'Yes'
    if 'yes' not in words and 'no' in words:
        return 'No'
    return 'Unknown'


def eval_func(pred_list, label_list):
    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print(f'{TP}\t{FP}\t{TN}\t{FN}')

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print(f'Accuracy: {acc}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 score: {f1}')
    print(f'Yes ratio: {yes_ratio}')
    return f1


def load_jsonl(json_file):
    with open(json_file) as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(json.loads(line))
    return data


class POPELLaVADataset(Dataset):

    def __init__(self,
                 data_file,
                 coco_val_path,
                 prompt_template,
                 image_processor,
                 tokenizer,
                 pad_image_to_square=True):
        if isinstance(data_file, str):
            data_file = [data_file]
        self.raw_data = [load_jsonl(f) for f in data_file]

        self.name = [
            os.path.splitext(os.path.basename(f))[0] for f in data_file
        ]

        self.coco_val_path = coco_val_path

        template = prompt_template
        self.template = template

        self.tokenizer = BUILDER.build(tokenizer)
        self.image_processor = BUILDER.build(image_processor)
        self.pad_image_to_square = pad_image_to_square

        self.results_xlsx_path = 'pope-results.xlsx'
        self.data = self.load_data_list()

    def get_image(self, image):
        image = Image.open(os.path.join(self.coco_val_path, image))
        return image

    def __len__(self):
        return len(self.data)

    def load_data_list(self):
        data_list = []
        idx = 0
        for data_idx in range(len(self.raw_data)):
            for sample_idx in range(len(self.raw_data[data_idx])):
                sample = self.raw_data[data_idx][sample_idx]
                index = sample['question_id']
                image_path = sample['image']
                question = sample['text']
                answer = sample['label']
                category = self.name[data_idx]
                assert answer in ['yes', 'no']
                data = {
                    'id': idx,
                    'index': index,
                    'img': image_path,
                    'question': question,
                    'answer': answer,
                    'category': category
                }
                data_list.append(data)
                idx += 1
        return data_list

    def __getitem__(self, idx):
        data = self.data[idx]
        data_dict = {'id': data['id']}

        text = data['question']
        text = DEFAULT_IMAGE_TOKEN + '\n' + text

        inputs = ''
        inputs += self.template['INSTRUCTION'].format(input=text, round=1)

        chunk_encode = []
        for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
            if idx == 0:
                cur_encode = self.tokenizer.encode(chunk)
            else:
                cur_encode = self.tokenizer.encode(
                    chunk, add_special_tokens=False)
            chunk_encode.append(cur_encode)
        assert len(chunk_encode) == 2
        ids = []
        for idx, cur_chunk_encode in enumerate(chunk_encode):
            ids.extend(cur_chunk_encode)
            if idx != len(chunk_encode) - 1:
                ids.append(IMAGE_TOKEN_INDEX)
        ids = torch.tensor(ids)
        data_dict['input_ids'] = ids

        image = self.get_image(data['img']).convert('RGB')
        if self.pad_image_to_square:
            image = expand2square(
                image,
                tuple(int(x * 255) for x in self.image_processor.image_mean))
        image = self.image_processor.preprocess(
            image, return_tensors='pt')['pixel_values'][0]
        data_dict['pixel_values'] = image

        return data_dict

    @master_only
    def postprocess_results(self, result, work_dir, timestamp, show=True):
        orig_index = [x['id'] for x in self.data]
        results = []
        for pred_dict in result:
            index = pred_dict['id']
            new_index = orig_index.index(index)
            filtered_rows = self.data[new_index]
            cur_result = {}
            cur_result['question'] = filtered_rows.get('question')
            cur_result['prediction'] = pred_dict['prediction']
            cur_result['category'] = filtered_rows['category']
            cur_result['index'] = filtered_rows.get('index')
            cur_result['answer'] = filtered_rows.get('answer')
            results.append(cur_result)

        results_df = pd.DataFrame(results)
        with pd.ExcelWriter(
                os.path.join(work_dir, self.results_xlsx_path),
                engine='openpyxl') as writer:
            results_df.to_excel(writer, index=False)

        score = 0
        for sub_name in self.name:
            sub_results = [x for x in results if x['category'] == sub_name]
            pred_list = [
                int(YOrN_Extraction(x['prediction']) == 'Yes')
                for x in sub_results
            ]
            label_list = [
                int(YOrN_Extraction(x['answer']) == 'Yes') for x in sub_results
            ]
            print('============================================')
            print('Category: {}, # samples: {}'.format(sub_name,
                                                       len(sub_results)))
            cur_f1 = eval_func(pred_list, label_list)
            score += cur_f1

        score /= len(self.name)
        print('============================================')
        print('============================================')
        print(f'Average F1-score: {score}')
        print('============================================')
        print('POPE successfully finished evaluating')
        return score
