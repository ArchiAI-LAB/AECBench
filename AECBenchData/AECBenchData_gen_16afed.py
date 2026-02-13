from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import JiebaRougeEvaluator
from opencompass.datasets import (MyDataset,mydataset_mcq_postprocess,
                                  MyDatasetlEvaluator_mcq,mydataset_zysb_postprocess,mydataset_xxpp_postprocess,
                                  MyDatasetlEvaluator_F0_5,MyDatasetlEvaluator_F1,mydataset_zbcq_postprocess,MyDatasetlEvaluator_F1Soft_zbcq,MyDatasetlEvaluator_F1Beta_xxpp,
                                  MyDatasetlEvaluator_gpt4o,MyDatasetlEvaluator_KendallTau,mydataset_score_content_accuracy_postprocess)

#单选选择
choice_sets = ['1-1规范记忆', '1-2专业术语', '1-3简称识别',
               '2-1规范条文解读',
               '2-2设计通识问答',
               '2-3规范条文解读（表格数据）',
               '3-1设计决策制定',
               '3-2建筑类型判断',
               '3-3设计决策制定（表格数据）',
               '4-1建筑设计中的计算',
               '4-2MEP设计中的计算',
               '4-3施工中的计算',
               '4-4工程经济学中的计算', '5-1-3合规审查']#,
terminology_sets = [] #1-3专业术语
professional_identification_sets =['5-1-1专业识别'] #专业识别
error_correction_sets = ['5-1-2文本纠错'] #文本纠错
information_extraction_sets = ['5-1-5信息抽取'] #招标抽取
information_matching_sets = ['5-1-4品牌合规验证'] #信息匹配
proposal_recommendation_sets = ['5-3-1设计方案推荐']  #3-3设计方案推荐
report_generation_sets = ['5-3-2报告生成']  #报告生成
information_check_sets = ['5-3-3招投标信息验证']  #信息核对
evaluation_sets = ['5-2-1建筑设计方案的评估', '5-2-2结构设计方案的评估'] #评估层级
AECBenchData_datasets = []


# 单选题推理配置
for _name in choice_sets:
    mydataset_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role="HUMAN",
                        prompt = f'{{instruction}}{{question}}'
                    )
                ])),
        # 上下文样本配置，此处指定 `ZeroRetriever`，即不使用上下文样本
        retriever=dict(type=ZeroRetriever),
        # 推理方式配置
        #   - PPLInferencer 使用 PPL（困惑度）获取答案
        #   - GenInferencer 使用模型的生成结果获取答案
        inferencer=dict(type=GenInferencer))

    # 评估配置，使用 mydataset_postprocess 作为评估指标
    mydataset_eval_cfg =  dict(
            evaluator=dict(type=MyDatasetlEvaluator_mcq),
            pred_role="BOT",
            pred_postprocessor=dict(type=mydataset_mcq_postprocess))

    # 数据集配置，以上各个变量均为此配置的参数
    # 为一个列表，用于指定一个数据集各个评测子集的配置。
    AECBenchData_datasets.append(
        dict(
             abbr= _name,
            type=MyDataset,
            path='./AECBench/one-shot',
            name=_name,
            reader_cfg=dict(input_columns=['instruction','question'],
                    output_column='answer'),
            infer_cfg=mydataset_infer_cfg.copy(),
            eval_cfg=mydataset_eval_cfg.copy())
    )

# 专业术语问答题 推理配置
for _name in terminology_sets:
    mydataset_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role="HUMAN",
                        prompt= f'{{instruction}}{{question}}'
                    )
                ])),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer))

    #内置的ROUGH_L模型分两种，分别适用与中文和英文，"JiebaRougeEvaluator"更适合于中文，"RougeEvaluator"更适用于英文。
    mydataset_eval_cfg =  dict(evaluator=dict(type=JiebaRougeEvaluator),pred_role="BOT")
    # mydataset_eval_cfg =  dict(evaluator=dict(type=MyDatasetlEvaluator),pred_role="BOT")

    AECBenchData_datasets.append(
        dict(
            abbr=_name,
            type=MyDataset,
            path='./AECBench/one-shot',
            name=_name,
            reader_cfg=dict(input_columns=['instruction','question'],
                    output_column='answer'),
            infer_cfg=mydataset_infer_cfg.copy(),
            eval_cfg=mydataset_eval_cfg.copy())
    )

# 专业识别推理配置
for _name in professional_identification_sets:
    mydataset_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role="HUMAN",
                        prompt= f'{{instruction}}{{question}}'
                    )
                ])),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer))

    mydataset_eval_cfg =  dict(
            evaluator=dict(type=MyDatasetlEvaluator_F1),
            pred_role="BOT",
            pred_postprocessor=dict(type=mydataset_zysb_postprocess))

    AECBenchData_datasets.append(
        dict(
            abbr=_name,
            type=MyDataset,
            path='./AECBench/one-shot',
            name=_name,
            reader_cfg=dict(input_columns=['instruction','question'],
                    output_column='answer'),
            infer_cfg=mydataset_infer_cfg.copy(),
            eval_cfg=mydataset_eval_cfg.copy())
    )
#文本纠错推理配置
for _name in error_correction_sets:
    mydataset_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role="HUMAN",
                        prompt= f'{{instruction}}{{question}}'
                    )
                ])),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer))

    mydataset_eval_cfg =  dict(
            evaluator=dict(type=MyDatasetlEvaluator_F0_5),
            pred_role="BOT")

    AECBenchData_datasets.append(
        dict(
            abbr=_name,
            type=MyDataset,
            path='./AECBench/one-shot',
            name=_name,
            reader_cfg=dict(input_columns=['instruction','question'],
                    output_column='answer'),
            infer_cfg=mydataset_infer_cfg.copy(),
            eval_cfg=mydataset_eval_cfg.copy())
    )

# 信息抽取
for _name in information_extraction_sets:
    mydataset_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role="HUMAN",
                        prompt= f'{{instruction}}{{question}}'
                    )
                ])),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer))

    mydataset_eval_cfg =  dict(
            evaluator=dict(type=MyDatasetlEvaluator_F1Soft_zbcq),
            pred_role="BOT",
            pred_postprocessor=dict(type=mydataset_zbcq_postprocess)

            )

    AECBenchData_datasets.append(
        dict(
            abbr=_name,
            type=MyDataset,
            path='./AECBench/one-shot',
            name=_name,
            reader_cfg=dict(input_columns=['instruction','question'],
                    output_column='answer'),
            infer_cfg=mydataset_infer_cfg.copy(),
            eval_cfg=mydataset_eval_cfg.copy())
    )

#信息匹配
for _name in information_matching_sets:
    mydataset_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role="HUMAN",
                        prompt= f'{{instruction}}{{question}}'
                    )
                ])),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer))

    mydataset_eval_cfg =  dict(
        evaluator=dict(type=MyDatasetlEvaluator_F1Beta_xxpp),
        pred_role="BOT",
        pred_postprocessor=dict(type=mydataset_xxpp_postprocess)
            )

    AECBenchData_datasets.append(
        dict(
            abbr=_name,
            type=MyDataset,
            path='./AECBench/one-shot',
            name=_name,
            reader_cfg=dict(input_columns=['instruction','question'],
                    output_column='answer'),
            infer_cfg=mydataset_infer_cfg.copy(),
            eval_cfg=mydataset_eval_cfg.copy())
    )

# 设计方案推荐
for _name in proposal_recommendation_sets:
    mydataset_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role="HUMAN",
                        prompt= f'{{instruction}}{{question}}'
                    )
                ])),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer))

    mydataset_eval_cfg =  dict(
            evaluator=dict(type=MyDatasetlEvaluator_gpt4o),
            pred_role="BOT")

    AECBenchData_datasets.append(
        dict(
            abbr=_name,
            type=MyDataset,
            path='./AECBench/one-shot',
            name=_name,
            reader_cfg=dict(input_columns=['instruction','question'],
                    output_column='answer'),
            infer_cfg=mydataset_infer_cfg.copy(),
            eval_cfg=mydataset_eval_cfg.copy())
    )

# 报告生成
for _name in report_generation_sets:
    mydataset_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role="HUMAN",
                        prompt= f'{{instruction}}{{question}}'
                    )
                ])),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer))

    mydataset_eval_cfg =  dict(
            evaluator=dict(type=MyDatasetlEvaluator_gpt4o),
            pred_role="BOT")

    AECBenchData_datasets.append(
        dict(
            abbr=_name,
            type=MyDataset,
            path='./AECBench/one-shot',
            name=_name,
            reader_cfg=dict(input_columns=['instruction','question'],
                    output_column='answer'),
            infer_cfg=mydataset_infer_cfg.copy(),
            eval_cfg=mydataset_eval_cfg.copy())
    )


# 信息核对
for _name in information_check_sets:
    mydataset_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role="HUMAN",
                        prompt= f'{{instruction}}{{question}}'
                    )
                ])),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer))

    mydataset_eval_cfg =  dict(
            evaluator=dict(type=MyDatasetlEvaluator_gpt4o),
            pred_role="BOT")

    AECBenchData_datasets.append(
        dict(
            abbr=_name,
            type=MyDataset,
            path='./AECBench/one-shot',
            name=_name,
            reader_cfg=dict(input_columns=['instruction','question'],
                    output_column='answer'),
            infer_cfg=mydataset_infer_cfg.copy(),
            eval_cfg=mydataset_eval_cfg.copy())
    )

# 评估层级（LLM-as-a-judge Kendall-Tau）
for _name in evaluation_sets:
    mydataset_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role="HUMAN",
                        prompt= f'{{instruction}}{{new_prompts}}'
                    )
                ])),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer))

    mydataset_eval_cfg =  dict(
            evaluator=dict(type=MyDatasetlEvaluator_KendallTau),
            pred_role="BOT",
            pred_postprocessor=dict(type=mydataset_score_content_accuracy_postprocess))

    AECBenchData_datasets.append(
        dict(
            abbr=_name,
            type=MyDataset,
            path='./AECBench/one-shot',
            name=_name,
            reader_cfg=dict(input_columns=['instruction','new_prompts'],
                    output_column='answer'),
            infer_cfg=mydataset_infer_cfg.copy(),
            eval_cfg=mydataset_eval_cfg.copy())
    )
