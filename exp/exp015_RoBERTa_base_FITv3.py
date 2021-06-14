import gc
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import (DataLoader, Dataset, RandomSampler, SequentialSampler)
from transformers import AdamW, AutoConfig, AutoModel, AutoTokenizer, \
    get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup


# =================================================
# Utilities #
# =================================================
def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_logger(log_file='train.log'):  # コンソールとログファイルの両方に出力
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)  # ログレベルを設定
    handler1 = StreamHandler()  # ログをコンソール出力するための設定
    handler1.setFormatter(Formatter("%(message)s"))  # ログメッセージ
    handler2 = FileHandler(filename=log_file)  # ログのファイル出力先を設定（4）
    handler2.setFormatter(Formatter("%(message)s"))  # ログメッセージ
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


# =================================================
# Config #
# =================================================
class Config:
    seed = 42
    epochs = 1
    max_len = 300
    batch_size = 8
    model_name = '../out/exp014_RoBERTa_base_ITPTv2/roberta-base-2-epochs/'
    train_file = '../input/commonlitreadabilityprize/train_folds.csv'
    warmup_proportion = 1
    optimizer = 'AdamW'
    decay_name = 'linear'
    scaler = None
    learning_rate = 2e-5
    tokenizer = 'roberta-base'
    filename = __file__.replace(".py", "")
    out_dir = '../out/' + filename + '/roberta-base-2-epochs' + str(epochs)
    check_dir = out_dir + '/checkpoint'
    log_interval = 10
    evaluate_interval = 40


def convert_examples_to_features(data, tokenizer, max_len, is_test=False):
    data = data.replace('\n', '')
    tok = tokenizer.encode_plus(
        data,
        max_length=max_len,
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True
    )
    curr_sent = {}
    padding_length = max_len - len(tok['input_ids'])
    curr_sent['input_ids'] = tok['input_ids'] + ([0] * padding_length)
    curr_sent['token_type_ids'] = tok['token_type_ids'] + \
                                  ([0] * padding_length)
    curr_sent['attention_mask'] = tok['attention_mask'] + \
                                  ([0] * padding_length)
    return curr_sent


class DatasetRetriever(Dataset):
    def __init__(self, data, tokenizer, max_len, is_test=False):
        self.data = data
        if 'excerpt' in self.data.columns:
            self.excerpts = self.data.excerpt.values.tolist()
        else:
            self.excerpts = self.data.text.values.tolist()
        self.targets = self.data.target.values.tolist()
        self.tokenizer = tokenizer
        self.is_test = is_test
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        excerpt, label = self.excerpts[item], self.targets[item]
        features = convert_examples_to_features(
            excerpt, self.tokenizer,
            self.max_len, self.is_test
        )
        return {
            'input_ids': torch.tensor(features['input_ids'], dtype=torch.long),
            'token_type_ids': torch.tensor(features['token_type_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(features['attention_mask'], dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.double),
        }


class CommonLitModel(nn.Module):
    def __init__(self, model_name, config, multisample_dropout=False, output_hidden_states=False):
        super(CommonLitModel, self).__init__()
        self.config = config
        self.roberta = AutoModel.from_pretrained(model_name,
                                                 output_hidden_states=output_hidden_states
                                                 )
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        if multisample_dropout:
            self.dropouts = nn.ModuleList([
                nn.Dropout(0.5) for _ in range(5)
            ])
        else:
            self.dropouts = nn.ModuleList([nn.Dropout(0.3)])
        # self.regressor = nn.Linear(config.hidden_size*2, 1)
        self.regressor = nn.Linear(config.hidden_size, 1)
        self._init_weights(self.layer_norm)
        self._init_weights(self.regressor)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[1]
        sequence_output = self.layer_norm(sequence_output)

        # max-avg head
        # average_pool = torch.mean(sequence_output, 1)
        # max_pool, _ = torch.max(sequence_output, 1)
        # concat_sequence_output = torch.cat((average_pool, max_pool), 1)

        # multi-sample dropout
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                logits = self.regressor(dropout(sequence_output))
            else:
                logits += self.regressor(dropout(sequence_output))

        logits /= len(self.dropouts)

        # calculate loss
        loss = None
        if labels is not None:
            # regression task
            loss_fn = torch.nn.MSELoss()
            logits = logits.view(-1).to(labels.dtype)
            loss = torch.sqrt(loss_fn(logits, labels.view(-1)))

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


def get_optimizer_params(model):
    # differential learning rate and weight decay
    param_optimizer = list(model.named_parameters())
    learning_rate = Config.learning_rate
    no_decay = ['bias', 'gamma', 'beta']
    group1 = ['layer.0.', 'layer.1.', 'layer.2.', 'layer.3.']
    group2 = ['layer.4.', 'layer.5.', 'layer.6.', 'layer.7.']
    group3 = ['layer.8.', 'layer.9.', 'layer.10.', 'layer.11.']
    group_all = ['layer.0.', 'layer.1.', 'layer.2.', 'layer.3.', 'layer.4.', 'layer.5.', 'layer.6.',
                 'layer.7.', 'layer.8.', 'layer.9.', 'layer.10.', 'layer.11.']
    optimizer_parameters = [
        {'params': [p for n, p in model.roberta.named_parameters() if
                    not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.roberta.named_parameters() if
                    not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],
         'weight_decay': 0.01, 'lr': learning_rate / 2.6},
        {'params': [p for n, p in model.roberta.named_parameters() if
                    not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],
         'weight_decay': 0.01, 'lr': learning_rate},
        {'params': [p for n, p in model.roberta.named_parameters() if
                    not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],
         'weight_decay': 0.01, 'lr': learning_rate * 2.6},
        {'params': [p for n, p in model.roberta.named_parameters() if
                    any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],
         'weight_decay': 0.0},
        {'params': [p for n, p in model.roberta.named_parameters() if
                    any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],
         'weight_decay': 0.0, 'lr': learning_rate / 2.6},
        {'params': [p for n, p in model.roberta.named_parameters() if
                    any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],
         'weight_decay': 0.0, 'lr': learning_rate},
        {'params': [p for n, p in model.roberta.named_parameters() if
                    any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],
         'weight_decay': 0.0, 'lr': learning_rate * 2.6},
        {'params': [p for n, p in model.named_parameters() if "roberta" not in n], 'lr': 1e-3,
         "momentum": 0.99},
    ]
    return optimizer_parameters


def make_model(model_name, num_labels=1):
    tokenizer = AutoTokenizer.from_pretrained(Config.tokenizer)
    config = AutoConfig.from_pretrained(model_name)
    config.update({'num_labels': num_labels})
    model = CommonLitModel(model_name, config=config)
    return model, tokenizer


def make_optimizer(model, optimizer_name="AdamW"):
    optimizer_grouped_parameters = get_optimizer_params(model)
    kwargs = {
        'lr': 2e-5,  # 1e-3
        'weight_decay': 0.01,
        'betas': (0.9, 0.98),  # defaults to (0.9, 0.999))
        'eps': 1e-06  # defaults 1e-6
    }
    if optimizer_name == "LAMB":
        optimizer = Lamb(optimizer_grouped_parameters, **kwargs)
        return optimizer
    elif optimizer_name == "Adam":
        optimizer = Adam(optimizer_grouped_parameters, **kwargs)
        return optimizer
    elif optimizer_name == "AdamW":
        optimizer = AdamW(optimizer_grouped_parameters, **kwargs)
        return optimizer
    else:
        raise Exception('Unknown optimizer: {}'.format(optimizer_name))


def make_scheduler(optimizer, decay_name='linear', t_max=None, warmup_steps=None):
    if decay_name == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[30, 60, 90],
            gamma=0.1
        )
    elif decay_name == 'cosine':
        scheduler = lrs.CosineAnnealingLR(
            optimizer,
            T_max=t_max
        )
    elif decay_name == "cosine_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=t_max
        )
    elif decay_name == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=t_max
        )
    else:
        raise Exception('Unknown lr scheduler: {}'.format(decay_name))
    return scheduler


def make_loader(data, tokenizer, max_len, batch_size, fold=0):
    train_set, valid_set = data[data['kfold'] != fold], data[data['kfold'] == fold]
    train_dataset = DatasetRetriever(train_set, tokenizer, max_len)
    valid_dataset = DatasetRetriever(valid_set, tokenizer, max_len)

    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        pin_memory=True, drop_last=False, num_workers=4
    )

    valid_sampler = SequentialSampler(valid_dataset)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        pin_memory=True, drop_last=False, num_workers=4
    )

    return train_loader, valid_loader


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0
        self.min = 1e5

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if val > self.max:
            self.max = val
        if val < self.min:
            self.min = val


class Trainer:
    def __init__(self, model, optimizer, scheduler, scalar=None, log_interval=Config.log_interval,
                 evaluate_interval=Config.evaluate_interval):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scalar = scalar
        self.log_interval = log_interval
        self.evaluate_interval = evaluate_interval
        self.evaluator = Evaluator(self.model, self.scalar)

    def train(self, train_loader, valid_loader, epoch,
              result_dict, tokenizer, fold, logger):
        count = 0
        losses = AverageMeter()
        self.model.train()

        for batch_idx, batch_data in enumerate(train_loader):
            input_ids, attention_mask, token_type_ids, labels = batch_data['input_ids'], \
                                                                batch_data['attention_mask'], \
                                                                batch_data['token_type_ids'], \
                                                                batch_data['label']
            input_ids, attention_mask, token_type_ids, labels = \
                input_ids.cuda(), attention_mask.cuda(), token_type_ids.cuda(), labels.cuda()

            if self.scalar is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels
                    )
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )

            loss, logits = outputs[:2]
            count += labels.size(0)
            losses.update(loss.item(), input_ids.size(0))

            if self.scalar is not None:
                self.scalar.scale(loss).backward()
                self.scalar.step(self.optimizer)
                self.scalar.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()
            self.optimizer.zero_grad()

            if batch_idx % self.log_interval == 0:
                _s = str(len(str(len(train_loader.sampler))))
                ret = [
                    ('epoch: {:0>3} [{: >' + _s + '}/{} ({: >3.0f}%)]')
                        .format(epoch, count, len(train_loader.sampler),
                                100 * count / len(train_loader.sampler)),
                    'train_loss: {: >4.5f}'.format(losses.avg),
                ]
                print(', '.join(ret))

            checkdir = Path(Config.check_dir)
            checkdir.mkdir(exist_ok=True, parents=True)

            if batch_idx % self.evaluate_interval == 0:
                result_dict = self.evaluator.evaluate(
                    valid_loader, epoch, result_dict, tokenizer
                )
                if result_dict['val_loss'][-1] < result_dict['best_val_loss']:
                    logger.info("{} epoch, best epoch was updated! valid_loss: {: >4.5f}"
                                .format(epoch, result_dict['val_loss'][-1]))
                    result_dict["best_val_loss"] = result_dict['val_loss'][-1]
                    torch.save(self.model.state_dict(), f"{Config.check_dir}/model{fold}.bin")

        result_dict['train_loss'].append(losses.avg)
        return result_dict


class Evaluator:
    def __init__(self, model, scalar=None):
        self.model = model
        self.scalar = scalar

    def worst_result(self):
        ret = {
            'loss': float('inf'),
            'accuracy': 0.0
        }
        return ret

    def result_to_str(self, result):
        ret = [
            'epoch: {epoch:0>3}',
            'loss: {loss: >4.2e}'
        ]
        for metric in self.evaluation_metrics:
            ret.append('{}: {}'.format(metric.name, metric.fmtstr))
        return ', '.join(ret).format(**result)

    def save(self, result):
        with open('result_dict.json', 'w') as f:
            f.write(json.dumps(result, sort_keys=True, indent=4, ensure_ascii=False))

    def load(self):
        result = self.worst_result
        if os.path.exists('result_dict.json'):
            with open('result_dict.json', 'r') as f:
                try:
                    result = json.loads(f.read())
                except:
                    pass
        return result

    def evaluate(self, data_loader, epoch, result_dict, tokenizer):
        losses = AverageMeter()

        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                input_ids, attention_mask, token_type_ids, labels = batch_data['input_ids'], \
                                                                    batch_data['attention_mask'], \
                                                                    batch_data['token_type_ids'], \
                                                                    batch_data['label']
                input_ids, attention_mask, token_type_ids, labels = input_ids.cuda(), \
                                                                    attention_mask.cuda(), token_type_ids.cuda(), labels.cuda()

                if self.scalar is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=labels
                        )
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels
                    )

                loss, logits = outputs[:2]
                losses.update(loss.item(), input_ids.size(0))

        print('----Validation Results Summary----')
        print('Epoch: [{}] valid_loss: {: >4.5f}'.format(epoch, losses.avg))

        result_dict['val_loss'].append(losses.avg)
        return result_dict


def config(fold=0):
    epochs = Config.epochs
    max_len = Config.max_len
    batch_size = Config.batch_size

    model, tokenizer = make_model(
        model_name=Config.model_name, num_labels=1)
    train = pd.read_csv(Config.train_file)

    train_loader, valid_loader = make_loader(
        train, tokenizer, max_len=max_len, batch_size=batch_size, fold=fold
    )
    num_update_steps_per_epoch = len(train_loader)
    max_train_steps = epochs * num_update_steps_per_epoch
    warmup_proportion = Config.warmup_proportion

    if warmup_proportion != 0:
        warmup_steps = math.ceil((max_train_steps * 2) / 100)
    else:
        warmup_steps = 0

    optimizer = make_optimizer(model, Config.optimizer)

    scheduler = make_scheduler(
        optimizer, decay_name=Config.decay_name,
        t_max=max_train_steps,
        warmup_steps=warmup_steps
    )

    if torch.cuda.device_count() >= 1:
        print('Model pushed to {} GPU(s), type {}.'.format(
            torch.cuda.device_count(),
            torch.cuda.get_device_name(0))
        )
        model = model.cuda()
    else:
        raise ValueError('CPU training is not supported')

    # scaler = torch.cuda.amp.GradScaler()
    scaler = Config.scaler

    result_dict = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'best_val_loss': np.inf
    }
    return (
        model, tokenizer, optimizer, scheduler,
        scaler, train_loader, valid_loader, result_dict, epochs
    )


def run(fold, logger):
    model, tokenizer, optimizer, scheduler, scaler, \
    train_loader, valid_loader, result_dict, epochs = config(fold)

    trainer = Trainer(model, optimizer, scheduler, scaler)
    train_time_list = []

    for epoch in range(epochs):
        result_dict['epoch'] = epoch

        torch.cuda.synchronize()
        tic1 = time.time()

        result_dict = trainer.train(train_loader, valid_loader, epoch,
                                    result_dict, tokenizer, fold, logger)

        torch.cuda.synchronize()
        tic2 = time.time()
        train_time_list.append(tic2 - tic1)

    torch.cuda.empty_cache()
    del model, tokenizer, optimizer, scheduler, scaler, train_loader, valid_loader,
    gc.collect()
    return result_dict


if __name__ == '__main__':
    set_seed(Config.seed)
    result_list = []
    result_list2 = []
    out_dir = Path(Config.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    logger = init_logger(log_file=Config.out_dir + "/train.log")
    for fold in range(5):
        logger.info('----')
        logger.info(f'FOLD: {fold}')
        result_dict = run(fold, logger)
        result_list.append(result_dict)
        logger.info('----')
        result_list2.append(result_dict["best_val_loss"])
    for i in range(5):
        logger.info('fold ' + str(i) + ':' + str(result_list2[i]))
    logger.info('CV ' + ':' + str(np.mean(result_list2)))