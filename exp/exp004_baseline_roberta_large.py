import os
import platform
import random
import warnings
from pathlib import Path
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from sklearn.metrics import mean_squared_error
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


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


def get_device() -> torch.device:
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
    epochs = 10
    lr = 1e-5
    max_len = 314
    train_bs = 16
    valid_bs = 32
    model = '../input/huggingface-roberta/roberta-large'
    model_name = 'roberta-large'
    train_file = '../input/commonlitreadabilityprize/train_folds.csv'
    out_dir = '../out/exp004'
    tokenizer = transformers.RobertaTokenizer.from_pretrained("roberta-large",
                                                              do_lower_case=True)  # 全て小文字に
    scaler = GradScaler()
    train = True


# =================================================
# Dataset #
# =================================================
class BERTDataset(Dataset):
    def __init__(self, review, target=None, is_test=False):
        self.review = review
        self.target = target
        self.is_test = is_test
        self.tokenizer = Config.tokenizer
        self.max_len = Config.max_len

    def __len__(self):
        return len(self.review)

    def __getitem__(self, idx):
        review = str(self.review[idx])
        review = ' '.join(review.split())
        global inputs

        inputs = self.tokenizer.encode_plus(
            review,
            None,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True
        )
        ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
        mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
        # token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.long)

        if self.is_test:
            return {
                'ids': ids,
                'mask': mask,
                # 'token_type_ids': token_type_ids,
            }
        else:
            targets = torch.tensor(self.target[idx], dtype=torch.float)
            return {
                'ids': ids,
                'mask': mask,
                # 'token_type_ids': token_type_ids,
                'targets': targets
            }


# =================================================
# dataloading
# =================================================
data = pd.read_csv(Config.train_file)
# data = data.sample(frac=1).reset_index(drop=True)
data = data[['excerpt', 'target', 'kfold']]

train_data = data[data.kfold != 0].reset_index(drop=True)
valid_data = data[data.kfold == 0].reset_index(drop=True)

train_set = BERTDataset(
    review=train_data['excerpt'].values,
    target=train_data['target'].values
)

valid_set = BERTDataset(
    review=valid_data['excerpt'].values,
    target=valid_data['target'].values
)

train = DataLoader(
    train_set,
    batch_size=Config.train_bs,
    shuffle=True,
    num_workers=8
)

valid = DataLoader(
    valid_set,
    batch_size=Config.valid_bs,
    shuffle=False,
    num_workers=8
)


# =================================================
# Model #
# =================================================
class ROBERTA_LARGE(nn.Module):
    def __init__(self):
        super(ROBERTA_LARGE, self).__init__()
        self.roberta = transformers.RobertaModel.from_pretrained(Config.model)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(1024, 1)

    def forward(self, ids, mask):
        _, output = self.roberta(ids, attention_mask=mask,
                                 return_dict=False)
        output = self.drop(output)
        output = self.fc(output)
        return output


# =================================================
# Trainer
# =================================================
class Trainer:
    def __init__(
            self,
            model,
            optimizer,
            scheduler,
            train_dataloader,
            valid_dataloader,
            device
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_data = train_dataloader
        self.valid_data = valid_dataloader
        self.loss_fn = self.yield_loss
        self.device = device

    def yield_loss(self, outputs, targets):
        """
        This is the loss function for this task
        """
        return torch.sqrt(nn.MSELoss()(outputs, targets))

    def train_one_epoch(self):
        """
        This function trains the model for 1 epoch through all batches
        """
        prog_bar = tqdm(enumerate(self.train_data), total=len(self.train_data))
        self.model.train()
        with autocast():
            for idx, inputs in prog_bar:
                ids = inputs['ids'].to(self.device, dtype=torch.long)
                mask = inputs['mask'].to(self.device, dtype=torch.long)
                # ttis = inputs['token_type_ids'].to(self.device, dtype=torch.long)
                targets = inputs['targets'].to(self.device, dtype=torch.float)

                outputs = self.model(ids=ids, mask=mask).view(-1)

                loss = self.loss_fn(outputs, targets)
                prog_bar.set_description('loss: {:.2f}'.format(loss.item()))

                Config.scaler.scale(loss).backward()
                Config.scaler.step(self.optimizer)
                Config.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()

    def valid_one_epoch(self):
        """
        This function validates the model for one epoch through all batches of the valid dataset
        It also returns the validation Root mean squared error for assesing model performance.
        """
        prog_bar = tqdm(enumerate(self.valid_data), total=len(self.valid_data))
        self.model.eval()
        all_targets = []
        all_predictions = []
        with torch.no_grad():
            for idx, inputs in prog_bar:
                ids = inputs['ids'].to(self.device, dtype=torch.long)
                mask = inputs['mask'].to(self.device, dtype=torch.long)
                # ttis = inputs['token_type_ids'].to(self.device, dtype=torch.long)
                targets = inputs['targets'].to(self.device, dtype=torch.float)

                outputs = self.model(ids=ids, mask=mask).view(-1)
                all_targets.extend(targets.cpu().detach().numpy().tolist())
                all_predictions.extend(outputs.cpu().detach().numpy().tolist())

        val_rmse_loss = np.sqrt(mean_squared_error(all_targets, all_predictions))
        print('Validation RMSE: {:.2f}'.format(val_rmse_loss))

        return val_rmse_loss

    def get_model(self):
        return self.model


# =================================================
# Optimizer and Scheduler #
# =================================================
def yield_optimizer(model):
    """
    Returns optimizer for specific parameters
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.003,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return transformers.AdamW(optimizer_parameters, lr=Config.lr)


# =================================================
# explain
# =================================================

if __name__ == '__main__':
    warnings.simplefilter('ignore')
    set_seed(Config.seed)
    # logging
    filename = __file__.split("/")[-1].replace(".py", "")
    logdir = Path(f"../out/{filename}")
    logdir.mkdir(exist_ok=True, parents=True)
    checkdir = Path(f"../out/{filename}/checkpoint")
    checkdir.mkdir(exist_ok=True, parents=True)
    fold = 0

    if Config.train:
        logger = init_logger(log_file=logdir / "train.log")
        if torch.cuda.is_available():
            logger.info("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
        else:
            logger.info("\n[INFO] GPU not found. Using CPU: {}\n".format(platform.processor()))
        DEVICE = get_device()
        logger.info("-" * 20)
        logger.info(f"Fold {fold} Training")
        logger.info("-" * 20)
        model = ROBERTA_LARGE().to(DEVICE)
        nb_train_steps = int(len(train_data) / Config.train_bs * Config.epochs)
        optimizer = yield_optimizer(model)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=nb_train_steps
        )

        trainer = Trainer(model, optimizer, scheduler, train, valid, DEVICE)

        best_loss = 100
        for epoch in range(1, Config.epochs + 1):
            logger.info(f"\n{'--' * 5} EPOCH: {epoch} {'--' * 5}\n")
            # Train for 1 epoch
            trainer.train_one_epoch()
            # Validate for 1 epoch
            current_loss = trainer.valid_one_epoch()
            logger.info(f"current_loss in this fold: {current_loss:.4f}")
            if current_loss < best_loss:
                logger.info(f"Saving best model in this fold: {current_loss:.4f}")
                torch.save(trainer.get_model().state_dict(),
                           f"{checkdir}/{Config.model_name}_fold_{fold}.pt")
                best_loss = current_loss

        logger.info(f"Best RMSE in fold: {fold} was: {best_loss:.4f}")
        logger.info(f"Final RMSE in fold: {fold} was: {current_loss:.4f}")

        del train_set, valid_set, train, valid, model, optimizer, scheduler, trainer, current_loss
        gc.collect()
        torch.cuda.empty_cache()
