import json
import os.path as osp
import re
import os
import csv
import math
import openai

from collections import Counter
from sklearn.metrics import f1_score, fbeta_score
from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import (ICL_EVALUATORS, LOAD_DATASET,
                                  TEXT_POSTPROCESSORS)

from .base import BaseDataset

__all__ = [
    'MyDataset',
    'mydataset_mcq_postprocess',
    'mydataset_zysb_postprocess',
    'mydataset_xxpp_postprocess',
    'mydataset_zbcq_postprocess',
    'mydataset_score_content_accuracy_postprocess',
    'MyDatasetlEvaluator_mcq',
    'MyDatasetlEvaluator_F1',
    'MyDatasetlEvaluator_F0_5',
    'MyDatasetlEvaluator_F1Soft_zbcq',
    'MyDatasetlEvaluator_F1Beta_xxpp',
    'MyDatasetlEvaluator_gpt4o',
    'MyDatasetlEvaluator_KendallTau',
]


class CJRCEvaluator:
    # def __init__(self, gold_file):
    #     self.gold_data = CJRCEvaluator.gold_answers_to_dict(gold_file)

    @staticmethod
    def normalize_answer(s):
        """Lower text and remove punctuation, storys and extra whitespace."""

        def remove_punc(text):
            return "".join(ch for ch in text if ch.isdigit() or ch.isalpha())

        def lower(text):
            return text.lower()

        return remove_punc(lower(s))

    @staticmethod
    def get_tokens(s):
        if not s: return []
        return list(CJRCEvaluator.normalize_answer(s))

    @staticmethod
    def compute_f1(a_gold, a_pred):
        gold_toks = CJRCEvaluator.get_tokens(a_gold)
        pred_toks = CJRCEvaluator.get_tokens(a_pred)
        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1


# 分点评分的答案处理
def mydataset_bgsc_preprocess(refer):
    if not isinstance(refer, dict):
        raise TypeError("输入必须是字典类型")
    else:
        out_list = refer.values()
        out = [f"{i + 1}.{item}" for i, item in enumerate(out_list)]

        return ";\n".join(out)

def get_QA_chatGPT4(user_msg):

    openai.api_key = os.environ.get('OPENAI_API_KEY')
    if not openai.api_key:
        raise ValueError('OPENAI_API_KEY环境变量未设置，请设置后再运行')
    system_msg = (
        '你现在是一个专业的结构领域专家，现在希望基于以下标准答案对模型给出的答案进行打分,标准答案为一个字典，标准答案中content为得分点，content内容后面的[score:分值]表示该得分点的分值，总分为1。'
        '现在希望你基于标准答案的得分点分值，*严谨准确*地给模型回答进行打分。以字典格式进行输出，其中key为模型的分数（如model_rate）和模型的打分依据（如model_reason），仅输出json字典格式.')
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_msg},
                  {"role": "user", "content": user_msg}],
        temperature=0.9,
        max_tokens=2000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    content = response.choices[0].message.content
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    return content, prompt_tokens, completion_tokens


def _coerce_number(value):
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _extract_score_from_text(text: str, score_key: str = '总得分'):
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return _coerce_number(text)
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            payload = json.loads(text[start:end + 1])
            if isinstance(payload, dict):
                if score_key in payload:
                    return _coerce_number(payload[score_key])
                if 'model_rate' in payload:
                    return _coerce_number(payload['model_rate'])
        except Exception:
            pass
    match = re.search(rf'{re.escape(score_key)}[^0-9]*([0-9]+(?:\.[0-9]+)?)', text)
    if match:
        return _coerce_number(match.group(1))
    match = re.search(r'model_rate[^0-9]*([0-9]+(?:\.[0-9]+)?)', text)
    if match:
        return _coerce_number(match.group(1))
    return None


def _extract_content_accuracy(text: str):
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return _coerce_number(text)
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            payload = json.loads(text[start:end + 1])
            if isinstance(payload, dict):
                detail = payload.get('评分明细')
                if isinstance(detail, dict) and '内容准确性' in detail:
                    return _coerce_number(detail['内容准确性'])
                if '内容准确性' in payload:
                    return _coerce_number(payload['内容准确性'])
        except Exception:
            pass
    match = re.search(r'内容准确性[^0-9]*([0-9]+(?:\.[0-9]+)?)', text)
    if match:
        return _coerce_number(match.group(1))
    return None


@LOAD_DATASET.register_module()
class MyDataset(BaseDataset):
    @staticmethod
    def load(path: str, name: str):
        with open(osp.join(path, f'{name}.json'), 'r', encoding='utf-8') as f:
            # print("TTT:",osp.join(path, f'{name}.json'))
            data = json.load(f)
        dataset = Dataset.from_list(data)
        return dataset


# 选择提取
@TEXT_POSTPROCESSORS.register_module('mydataset-mcq')
def mydataset_mcq_postprocess(text: str) -> str:
    pattern = r"[A-D]"
    matches = re.findall(pattern, text)
    out = "".join(matches)
    return out


@TEXT_POSTPROCESSORS.register_module('mydataset-score')
def mydataset_score_postprocess(text: str):
    score = _extract_score_from_text(text)
    if score is None:
        return float('nan')
    return score


@TEXT_POSTPROCESSORS.register_module('mydataset-score-content-accuracy')
def mydataset_score_content_accuracy_postprocess(text: str):
    score = _extract_content_accuracy(text)
    if score is None:
        return float('nan')
    return score


# 专业识别预测提取
@TEXT_POSTPROCESSORS.register_module('mydataset-zysb')
def mydataset_zysb_postprocess(text: str) -> str:
    pattern = r'(建筑|结构|机电)[\(|（](.*?)[）|\)]'
    match = re.search(pattern, text)
    if match:
        category = match.group(1)
        content = match.group(2)
        return [category, content]
    else:
        found_category = False
        for category in ["建筑", "结构", "机电"]:
            if category in text:
                found_category = True
                return [category, ""]

        if not found_category:
            return ['', '']


# 信息抽取任务结果提取
@TEXT_POSTPROCESSORS.register_module('mydataset-zbcq')
def mydataset_zbcq_postprocess(text: str) -> str:
    text_list = text.split(r'<SEG>')
    sort_order = {
        "项目名称": 0,
        "项目类型": 1,
        "项目地区": 2,
        "项目预算": 3,
        "企业资格要求": 4,
        "建设用地规模": 5,
        "建设工期": 6,
        "计价说明": 7,
        "投标截止时间": 8
    }
    # 根据关键字提取信息并排序
    sorted_list = [""] * len(sort_order)  # 初始化排序后的列表，用空字符串填充
    # 将原始列表中的元素按照指定顺序放到sorted_list中
    for item in text_list:
        for key, v in sort_order.items():
            if key in item:
                sorted_list[sort_order[key]] = item  # 减1是因为索引从0开始
    return sorted_list


# 信息匹配
@TEXT_POSTPROCESSORS.register_module('mydataset-xxpp')
def mydataset_xxpp_postprocess(text: str) -> str:
    match = re.search(r'\[投标偏离\](.*?)<eoa>', text)
    content = ""
    if match:
        content = match.group(1)
        return content
    else:
        return content

# 选择题
@ICL_EVALUATORS.register_module()
class MyDatasetlEvaluator_mcq(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                         'length'
            }
        details = []
        cnt = 0
        for pred, ref in zip(predictions, references):
            detail = {'pred': pred, 'answer': ref, 'correct': False}
            ref = mydataset_mcq_postprocess(ref)
            if pred == ref:
                cnt += 1
                detail['correct'] = True
            details.append(detail)

        score = cnt / len(predictions) * 100

        return {'score': score, 'details': details}


# 专业识别F1
@ICL_EVALUATORS.register_module()
class MyDatasetlEvaluator_F1(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                         'length'
            }
        score = 0
        for reference, prediction in zip(references, predictions):
            reference = mydataset_zysb_postprocess(reference)
            score += f1_score(reference, prediction, average='weighted')

        score = score / len(references) * 100

        return {'score': score}


# 文本校对
@ICL_EVALUATORS.register_module()
class MyDatasetlEvaluator_F0_5(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                         'length'
            }
        score = 0
        for reference, prediction in zip(references, predictions):
            unique_chars = set(reference) | set(prediction)
            y_true = [1 if char in reference else 0 for char in unique_chars]
            y_pred = [1 if char in prediction else 0 for char in unique_chars]

            score += fbeta_score(y_true, y_pred, beta=0.5, average='binary')

        score = score / len(references) * 100

        return {'score': score}


# 招标信息抽取
@ICL_EVALUATORS.register_module()
class MyDatasetlEvaluator_F1Soft_zbcq(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                         'length'
            }
        score = 0
        for reference, prediction in zip(references, predictions):
            reference = mydataset_zbcq_postprocess(reference)
            intersected = [CJRCEvaluator.compute_f1(r, h) for r, h in zip(reference, prediction)]

            prec = sum(intersected) / len(prediction) if len(prediction) > 0 else 0
            rec = sum(intersected) / len(reference) if len(reference) > 0 else 0

            score += 2 * prec * rec / (prec + rec + 1e-10)

        score = score / len(references) * 100

        return {'score': score}


# 信息匹配
@ICL_EVALUATORS.register_module()
class MyDatasetlEvaluator_F1Beta_xxpp(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                         'length'
            }
        score_sum = 0
        for reference, prediction in zip(references, predictions):
            reference = mydataset_xxpp_postprocess(reference)
            unique_chars = set(reference) | set(prediction)
            y_true = [1 if char in reference else 0 for char in unique_chars]
            y_pred = [1 if char in prediction else 0 for char in unique_chars]

            score = fbeta_score(y_true, y_pred, beta=1, average='binary')
            score_sum += score
        score_out = score_sum / len(references) * 100

        return {'score': score_out}


# 设计方案推荐、报告生成、信息核对
@ICL_EVALUATORS.register_module()
class MyDatasetlEvaluator_gpt4o(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                         'length'
            }
        score = 0
        for reference, prediction in zip(references, predictions):
            reference = mydataset_bgsc_preprocess(reference)
            user_msg = f"""
            你作为已经考取结构注册从业资格证书的结构项目负责人，并且具有丰富的建筑以及结构注册相关知识.
            现在希望基于以下标准答案对模型回答进行打分,标准答案为一个字典，标准答案中的content为答案的得分点，content内容后面的[score:分值]表示该得分点的分值，总分为1。
            打分逻辑如下所示：
            1）得分点1：判断标准答案和模型答案是否意思一致，若意思一致，则该得分点分值为标准答案content尾部[score:分值]中的分值，若不一致，该得分点分值为0；
            2）标准答案有几个得分点，则进行上述判别步骤几次，得到每个得分点得分值列表；
            3）对该分值列表进行求和，得到数值和为该模型回答得分值（model_rate）。
            ****************************************

            标准答案：{reference}   

            模型回答：{prediction}

            *****************************************
            以字典格式进行输出，其中key为模型的分数（如model_rate）和模型的打分依据（如model_reason），仅输出json字典格式."""
            content, prompt_tokens, completion_tokens = get_QA_chatGPT4(user_msg)

            match = re.search(r'"model_rate"\s*[:|：]\s*(\d+(\.\d+)?)[,|，]', content)
            result = 0
            if match:
                result = float(match.group(1))
            # print("result:",result)

            if result:
                score += result
            else:
                score += 0

        score = score / len(references) * 100

        return {'score': score}


@ICL_EVALUATORS.register_module()
class MyDatasetlEvaluator_KendallTau(BaseEvaluator):

    def __init__(self,
                 score_file: str = None,
                 score_format: str = 'auto',
                 sheet_name=0,
                 score_column: int = 1,
                 row_start: int = 0,
                 row_end: int = None,
                 score_key: str = '总得分',
                 has_header: bool = False):
        self.score_file = score_file
        self.score_format = score_format
        self.sheet_name = sheet_name
        self.score_column = score_column
        self.row_start = row_start
        self.row_end = row_end
        self.score_key = score_key
        self.has_header = has_header

    def _load_scores(self):
        score_file = self.score_file or os.environ.get('AECBENCH_EVAL_SCORE_FILE')
        if not score_file:
            raise ValueError('score_file not set; provide score_file in config or set AECBENCH_EVAL_SCORE_FILE.')
        ext = osp.splitext(score_file)[1].lower()
        fmt = self.score_format
        if fmt == 'auto':
            if ext in ('.csv',):
                fmt = 'csv'
            elif ext in ('.xls', '.xlsx'):
                fmt = 'excel'
        scores = []
        if fmt == 'csv':
            with open(score_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                if self.has_header:
                    next(reader, None)
                for row in reader:
                    if self.score_column >= len(row):
                        scores.append(float('nan'))
                        continue
                    scores.append(_coerce_number(row[self.score_column]))
        elif fmt == 'excel':
            try:
                import pandas as pd
            except Exception as e:
                raise ImportError('pandas is required to read Excel score files.') from e
            df = pd.read_excel(score_file, sheet_name=self.sheet_name)
            if isinstance(self.score_column, int):
                col = df.iloc[:, self.score_column]
            else:
                col = df[self.score_column]
            scores = [ _coerce_number(v) for v in col.tolist() ]
        else:
            raise ValueError(f'Unsupported score_format: {self.score_format}')
        if self.row_end is None:
            scores = scores[self.row_start:]
        else:
            scores = scores[self.row_start:self.row_end]
        return scores

    @staticmethod
    def _kendall_tau_b(x, y):
        n = len(x)
        c = d = tx = ty = 0
        for i in range(n - 1):
            xi = x[i]
            yi = y[i]
            for j in range(i + 1, n):
                xj = x[j]
                yj = y[j]
                if xi == xj and yi == yj:
                    continue
                if xi == xj:
                    tx += 1
                    continue
                if yi == yj:
                    ty += 1
                    continue
                s = (xi - xj) * (yi - yj)
                if s > 0:
                    c += 1
                elif s < 0:
                    d += 1
        denom = math.sqrt((c + d + tx) * (c + d + ty))
        if denom == 0:
            return 0.0
        return (c - d) / denom

    def score(self, predictions, references):
        if not predictions:
            return {'error': 'predictions is empty'}
        pred_scores = [_extract_score_from_text(p, self.score_key) for p in predictions]
        if self.score_file or os.environ.get('AECBENCH_EVAL_SCORE_FILE'):
            expert_scores = self._load_scores()
        else:
            expert_scores = [_coerce_number(r) for r in references]
        if len(pred_scores) != len(expert_scores):
            return {
                'error': 'predictions and expert scores have different length',
                'predictions_len': len(pred_scores),
                'expert_scores_len': len(expert_scores)
            }
        filtered = [(p, e) for p, e in zip(pred_scores, expert_scores)
                    if p is not None and e is not None]
        if len(filtered) < 2:
            return {'error': 'not enough valid score pairs', 'valid_count': len(filtered)}
        xs, ys = zip(*filtered)
        tau = self._kendall_tau_b(xs, ys)
        return {
            'score': tau,
            'kendall_tau': tau,
            'valid_count': len(filtered)
        }
