import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn import model_selection


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_logger(log_file='train.log'):  # コンソールとログファイルの両方に出力
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)  # ログレベルを設定
    handler1 = StreamHandler()  # ログをコンソール出力するための設定
    handler1.setFormatter(Formatter("%(message)s"))  # ログメッセージ
    handler2 = FileHandler(filename=log_file)  ## ログのファイル出力先を設定（4）
    handler2.setFormatter(Formatter("%(message)s"))  # ログメッセージ
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def create_folds(data, num_splits):
    data["kfold"] = -1
    data = data.sample(frac=1).reset_index(drop=True)
    # calculate number of bins by Sturge's rule
    # I take the floor of the value, you can also
    # just round it
    num_bins = int(np.floor(1 + np.log2(len(data))))
    # bin targets
    data.loc[:, "bins"] = pd.cut(
        data["target"], bins=num_bins, labels=False
    )
    kf = model_selection.StratifiedKFold(n_splits=num_splits)
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'kfold'] = f
    # drop the bins column
    data = data.drop("bins", axis=1)
    # return dataframe with folds
    return data


if __name__ == '__main__':
    set_seed(42)
    filename = __file__.split("/")[-1].replace(".py", "")
    logdir = Path(f"../out/{filename}")
    logdir.mkdir(exist_ok=True, parents=True)
    logger = init_logger(log_file=logdir / "train.log")
    logger.info('start\n')
    # read training data
    df = pd.read_csv("../input/commonlitreadabilityprize/train.csv")
    # create folds
    df = create_folds(df, num_splits=5)
    logger.info(df.head())
    logger.info('\n')
    df.to_csv("../input/commonlitreadabilityprize/train_folds.csv", index=False)
    logger.info('end')
