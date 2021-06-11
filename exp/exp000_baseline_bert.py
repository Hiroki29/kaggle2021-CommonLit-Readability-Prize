import gc
import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm

warnings.simplefilter('ignore')


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


# =================================================
# Config #
# =================================================
class Config:
    epochs = 10
    lr = 1e-5
    max_len = 185
    n_splits = 5
    train_bs = 16
    valid_bs = 32
    bert_model = '../input/huggingface-bert/bert-large-uncased'
    model_name = 'bert-large-uncased'
    train_file = '../input/commonlitreadabilityprize/train.csv'
    test_file = '../input/commonlitreadabilityprize/test.csv'
    state_dir = '../output/exp001/'
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)
    scaler = GradScaler()
    train = True
    inference = False


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
        token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.long)

        if self.is_test:
            return {
                'ids': ids,
                'mask': mask,
                'token_type_ids': token_type_ids,
            }
        else:
            targets = torch.tensor(self.target[idx], dtype=torch.float)
            return {
                'ids': ids,
                'mask': mask,
                'token_type_ids': token_type_ids,
                'targets': targets
            }


# =================================================
# dataloading
# =================================================
data = pd.read_csv(Config.train_file)
data = data.sample(frac=1).reset_index(drop=True)
data = data[['excerpt', 'target']]

# Do Kfolds training and cross validation
kf = StratifiedKFold(n_splits=Config.n_splits)
nb_bins = int(np.floor(1 + np.log2(len(data))))
data.loc[:, 'bins'] = pd.cut(data['target'], bins=nb_bins, labels=False)

for fold, (train_idx, valid_idx) in enumerate(kf.split(X=data, y=data['bins'].values)):
    if fold != 0:
        continue
    train_data = data.loc[train_idx]
    valid_data = data.loc[valid_idx]

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

test_file = pd.read_csv(Config.test_file)
test_data = BERTDataset(test_file['excerpt'].values, is_test=True)
test_data = DataLoader(
    test_data,
    batch_size=Config.train_bs,
    shuffle=False
)


# =================================================
# Model #
# =================================================

class BERT_BASE_UNCASED(nn.Module):
    def __init__(self):
        super(BERT_BASE_UNCASED, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(Config.bert_model)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        _, output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output = self.drop(output)
        output = self.fc(output)
        return output


class BERT_LARGE_UNCASED(nn.Module):
    def __init__(self):
        super(BERT_LARGE_UNCASED, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(Config.bert_model)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(1024, 1)

    def forward(self, ids, mask, token_type_ids):
        _, output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output = self.drop(output)
        output = self.fc(output)
        return output


class DBERT_BASE_UNCASED(nn.Module):
    def __init__(self):
        super(DBERT_BASE_UNCASED, self).__init__()
        self.dbert = transformers.DistilBertModel.from_pretrained(Config.bert_model)
        self.drop = nn.Dropout(0.2)
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask):
        output = self.dbert(ids, attention_mask=mask, return_dict=False)
        output = self.drop(output)
        output = self.out(output)
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
                ttis = inputs['token_type_ids'].to(self.device, dtype=torch.long)
                targets = inputs['targets'].to(self.device, dtype=torch.float)

                outputs = self.model(ids=ids, mask=mask, token_type_ids=ttis).view(-1)

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
                ttis = inputs['token_type_ids'].to(self.device, dtype=torch.long)
                targets = inputs['targets'].to(self.device, dtype=torch.float)

                outputs = self.model(ids=ids, mask=mask, token_type_ids=ttis).view(-1)
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
# Inference
# =================================================
@torch.no_grad()
def inference(model, states_list, test_dataloader, device=torch.device('cuda:0')):
    """
    Do inference for different model folds
    """
    model.eval()
    all_preds = []
    for state in states_list:
        print(f"State: {state}")
        state_dict = torch.load(state)
        model.load_state_dict(state_dict)
        model = model.to(device)

        # Clean
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()

        preds = []
        prog = tqdm(test_dataloader, total=len(test_dataloader))
        for data in prog:
            ids = data['ids'].to(DEVICE, dtype=torch.long)
            mask = data['mask'].to(DEVICE, dtype=torch.long)
            ttis = data['token_type_ids'].to(DEVICE, dtype=torch.long)

            outputs = model(ids=ids, mask=mask, token_type_ids=ttis)
            preds.append(outputs.squeeze(-1).cpu().detach().numpy())

        all_preds.append(np.concatenate(preds))

        # Clean
        gc.collect()
        torch.cuda.empty_cache()

    return all_preds


# =================================================
# explain
# =================================================

if __name__ == '__main__':
    DEVICE = get_device()

    if Config.train:
        model = BERT_LARGE_UNCASED().to(DEVICE)
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
            print(f"\n{'--' * 5} EPOCH: {epoch} {'--' * 5}\n")

            # Train for 1 epoch
            trainer.train_one_epoch()

            # Validate for 1 epoch
            current_loss = trainer.valid_one_epoch()

            if current_loss < best_loss:
                print(f"Saving best model in this fold: {current_loss:.4f}")
                torch.save(trainer.get_model().state_dict(), f"{Config.model_name}_fold_{fold}.pt")
                best_loss = current_loss

        print(f"Best RMSE in fold: {fold} was: {best_loss:.4f}")
        print(f"Final RMSE in fold: {fold} was: {current_loss:.4f}")

    if Config.inference:
        state_list = [os.path.join(Config.state_dir, x) for x in os.listdir(Config.state_dir) if x.endswith(".pt")]
        model = BERT_BASE_UNCASED()

        print("Doing Predictions for all folds")
        predictions = inference(model, state_list, test_data, device=DEVICE)

        final_predictions = pd.DataFrame(predictions).T.mean(axis=1).tolist()
