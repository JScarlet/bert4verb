import logging
from pathlib import Path

import torch
from fast_bert.data_cls import BertDataBunch
from fast_bert.metrics import accuracy
from fast_bert.learner_cls import BertLearner

from definitions import ROOT_DIR

if __name__ == '__main__':
    data_dir = Path(ROOT_DIR) / 'data'
    label_dir = Path(ROOT_DIR) / 'data'
    databunch = BertDataBunch(data_dir, label_dir,
                              tokenizer='bert-base-uncased',
                              train_file='train.csv',
                              val_file='val.csv',
                              label_file='labels.csv',
                              text_col='text',
                              label_col='label',
                              batch_size_per_gpu=8,
                              max_seq_length=512,
                              multi_gpu=True,
                              multi_label=False,
                              model_type='bert')
    logger = logging.getLogger()
    # device_cuda = torch.device("cuda")
    device = torch.device('cuda') if torch.cuda.device_count() else torch.device('cpu')
    metrics = [{'name': 'accuracy', 'function': accuracy}]

    output_dir = Path(ROOT_DIR) / "model_output" / ""
    output_dir.mkdir(parents=True, exist_ok=True)
    # BertLearner可以实现训练，验证，测试
    learner = BertLearner.from_pretrained_model(
        databunch,
        pretrained_path='bert-base-uncased',  # 预训练模型的位置
        metrics=metrics,  # 希望模型在验证集上计算的度量函数的列表，如精度、beta等
        device=device,  # cpu or gpu
        logger=logger,  # 日志
        output_dir=output_dir,  # 输出目录，存放trained artefacts, tokenizer vocabulary, tensorboard files
        finetuned_wgts_path=None,  # 微调语言模型的位置
        warmup_steps=500,
        multi_gpu=True,
        is_fp16=True,  # cpu上训练应该设为False
        multi_label=False,
        logging_steps=50)
    learner.fit(epochs=1,
                lr=6e-5,
                validate=True,  # Evaluate the model after each epoch
                schedule_type="warmup_cosine",
                optimizer_type="lamb")  # lamb,adamw
    learner.save_model()
