# The following code has been implemented using pytorch-forecasting 
# Refer to their doc: https://pytorch-forecasting.readthedocs.io/en/latest/ to follow along

import os
import warnings
from tqdm import tqdm
from tqdm import tqdm_notebook
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import gc

warnings.filterwarnings("ignore")


# Pytorch lightning to keep track of best model state
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch

from pytorch_forecasting import Baseline, NBeats, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder, TorchNormalizer
from pytorch_forecasting.data.examples import generate_ar_data
from pytorch_forecasting.metrics import SMAPE, QuantileLoss, RMSE, DistributionLoss
import math

thread_store = pickle.load(open('./time_series.pkl', 'rb'))

df = pd.DataFrame(columns=['series', 'time_idx', 'value'])
final = []
c = 0
x = 0
y = 250
pl.seed_everything(42)

# Keep track of errors or fails in execution
fails = []

# Code to convert data from time_series data to required format of pytorch-forecasting
for i in tqdm(range(x, y)):
    try:
        df = pd.DataFrame(columns=['series', 'time_idx', 'value'])
        sum_thread = thread_store[i][0]
        start_thread = thread_store[i][1]
        end_thread = thread_store[i][2]
        max_thread = thread_store[i][3]
        min_thread = thread_store[i][4]
        file_name = thread_store[i][5]
        tempdf = pd.DataFrame()

        # Using all features
        tempdf['value'] = sum_thread
        tempdf['start'] = start_thread
        tempdf['end'] = end_thread
        tempdf['max'] = max_thread
        tempdf['min'] = min_thread
        tempdf['time_idx'] = range(0, len(tempdf))
        tempdf['series'] = c
        df = tempdf
        df = df.reset_index()
        df = df.drop('index', 1)
        df['series'] = df['series'].astype(np.int16)
        df['time_idx'] = df['time_idx'].astype(np.int16)
        df['value'] = df['value'].astype(np.float64)
        mean_val = df['value'].mean()
        std = df['value'].std()
        # Normalizing the code
        df['norm'] = (df['value'] - mean_val)/std
        max_encoder_length = 20
        max_prediction_length = 1
        training_cutoff = math.floor(len(df) * 0.7)
        
        context_length = max_encoder_length
        prediction_length = max_prediction_length
        l = int(training_cutoff - max_encoder_length)
        
        training = TimeSeriesDataSet(
            df[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="norm",
            categorical_encoders={"series": NaNLabelEncoder().fit(df.series)},
            group_ids=["series"],
            time_varying_unknown_reals=["norm"],
            max_encoder_length=context_length,
            max_prediction_length=1,
        )
        testing = TimeSeriesDataSet(
            df[lambda x: x.time_idx > training_cutoff - context_length],
            time_idx="time_idx",
            target="norm",
            categorical_encoders={"series": NaNLabelEncoder().fit(df.series)},
            group_ids=["series"],
            time_varying_unknown_reals=["norm"],
            max_encoder_length=context_length,
            max_prediction_length=1,
        )

        validation = TimeSeriesDataSet(
            df[lambda x: (x.time_idx > 0.95 * training_cutoff - context_length)][lambda x: (x.time_idx <= training_cutoff)],
            time_idx="time_idx",
            target="norm",
            categorical_encoders={"series": NaNLabelEncoder().fit(df.series)},
            group_ids=["series"],
            time_varying_unknown_reals=["norm"],
            max_encoder_length=context_length,
            max_prediction_length=1,
        )
        
        batch_size = 16
        train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
        test_dataloader = testing.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

        # Early stopping tuner
        pl.seed_everything(42)
        trainer = pl.Trainer(gpus=1, gradient_clip_val=0.1)
        net = NBeats.from_dataset(training, learning_rate=3e-2, weight_decay=1e-2, widths=[32, 512], backcast_loss_ratio=0.1)
        res = trainer.tuner.lr_find(net, train_dataloader=train_dataloader, val_dataloaders=val_dataloader, min_lr=1e-5)
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
        trainer = pl.Trainer(
            max_epochs=100,
            gpus=1,
            weights_summary="top",
            gradient_clip_val=0.1,
            callbacks=[early_stop_callback],
            limit_train_batches=30,
        )

        net = NBeats.from_dataset(
            training,
            learning_rate=4e-3,
            log_interval=10,
            log_val_interval=1,
            weight_decay=1e-2,
        )

        # Change learning rate
        net.hparams.learning_rate = res.suggestion()

        # Fit trainer
        trainer.fit(
            net,
            train_dataloader=train_dataloader,
            val_dataloaders=val_dataloader,
        )


        best_model_path = trainer.checkpoint_callback.best_model_path
        best_model = NBeats.load_from_checkpoint(best_model_path)

        actuals = torch.cat([y[0] for x, y in iter(test_dataloader)])
        predictions = best_model.predict(test_dataloader)
        # Save results
        nbeats_df = pd.DataFrame(columns = ['original', 'predicted'])
        nbeats_df['original'] = np.array(actuals.flatten()*std+mean_val).tolist()
        nbeats_df['predicted'] = np.array(predictions.flatten()*std+mean_val).tolist()
        nbeats_df.to_csv('{2}_{3}.csv'.format(training_cutoff, len(actuals), file_name.split('.')[0], 'nbeats'))
        # Pthon garbage collector
        gc.collect()
    except:
        fails.append(thread_store[i][5])