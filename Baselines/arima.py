# pip install pmdarima

from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
from pmdarima import auto_arima

time_series = pickle.load(open('time_series.pkl', 'rb'))

for i in tqdm(time_series):
    train = i[0][:int(len(i[0])*0.7) + 1]
    actuals = i[0][int(len(i[0])*0.7) + 1:]
    stepwise_model = auto_arima(train, start_p=0, start_q=0,
                           max_p=3, max_q=3, m=0,
                           start_P=0, seasonal=False,
                           d=0, D=1, trace=False,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
    stepwise_model.fit(train)
    predictions = stepwise_model.predict(n_periods=len(actuals))
    df = pd.DataFrame(columns = ['original', 'predicted'])
    df['original'] = train.tolist() + actuals.tolist()
    df['predicted'] = train.tolist() + predictions.tolist()
    df.to_csv('{0}_{1}.csv'.format(i[5].split('.')[0], 'arima'))