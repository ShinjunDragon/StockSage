# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 14:46:31 2024

@author: KITCOOP
"""

import numpy as np
import pandas as pd
import FinanceDataReader as fdr
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import RMSprop

def MinMaxScaler(data):
    """최솟값과 최댓값을 이용하여 0 ~ 1 값으로 변환"""
    numerator = data - np.min(data, axis=0)
    denominator = np.max(data, axis=0) - np.min(data, axis=0)
    return numerator / (denominator + 1e-7)

# 종목 리스트 (예: '005930', '000660', '035420')
#index 283부터 시작
kospi_stocks = fdr.StockListing('KOSPI')
tickers = kospi_stocks['Code'].tolist()

#062040 오류(156)
#487570 오류(504)


for t in tickers[804:]:
    print(f"Processing ticker: {t}")

    # 데이터 로드
    today = pd.to_datetime('today').strftime('%Y-%m-%d')
    df = fdr.DataReader('008250', '2015-01-01', today)
    dfx = df[['Open', 'High', 'Low', 'Volume', 'Close']]
    dfx = MinMaxScaler(dfx)
    dfy = dfx[['Close']]
    dfx = dfx[['Open', 'High', 'Low', 'Volume']]
    X = dfx.values
    y = dfy.values

    window_size = 10

    # 시계열 데이터 생성
    data_X = []
    data_y = []
    for i in range(len(y) - window_size):
        _X = X[i:i + window_size]
        _y = y[i + window_size]
        data_X.append(_X)
        data_y.append(_y)

    data_X = np.array(data_X)
    data_y = np.array(data_y)

    # 데이터 분할
    train_size = int(len(data_y) * 0.7)
    train_X = data_X[:train_size]
    train_y = data_y[:train_size]
    test_X = data_X[train_size:]
    test_y = data_y[train_size:]

    # 기초 모델 정의 및 훈련
    model = Sequential()
    model.add(LSTM(units=100, activation='tanh', return_sequences=True, input_shape=(window_size, 4)))
    model.add(Dropout(0.3))
    model.add(LSTM(units=100, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(units=1))

    # 옵티마이저와 학습률 조정
    optimizer = RMSprop(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.fit(train_X, train_y, epochs=100, batch_size=32, validation_split=0.1)

    # 모델 저장
    model.save(f'{t}_basic_model.keras')

    # 초기 예측
    initial_pred_y = model.predict(test_X)
    initial_pred_y = initial_pred_y.flatten()
    test_y = test_y.flatten()

    # 오차 계산
    errors = test_y - initial_pred_y

    # 오차 보정 모델 정의 및 훈련
    error_model = Sequential()
    error_model.add(LSTM(units=20, activation='tanh', return_sequences=True, input_shape=(window_size, 4)))
    error_model.add(Dropout(0.1))
    error_model.add(LSTM(units=20, activation='tanh'))
    error_model.add(Dropout(0.1))
    error_model.add(Dense(units=1))

    error_model.compile(optimizer='adam', loss='mean_squared_error')

    # 오차 보정 데이터 준비
    train_size_error = len(test_y)  # 전체 데이터 사용
    train_error_X = test_X
    train_error_y = errors

    # 오차 보정 모델 훈련
    error_model.fit(train_error_X, train_error_y, epochs=70, batch_size=30, validation_split=0.1)

    # 오차 보정 모델 저장
    error_model.save(f'{t}_error_correction_model.keras')

    # 오차 예측
    error_pred_y = error_model.predict(test_X)
    error_pred_y = error_pred_y.flatten()

    # 최종 예측 (기초 모델 + 오차 보정)
    final_pred_y = initial_pred_y + error_pred_y

    # 예측 가격 계산
    predictPrice = df.Close[-1] * final_pred_y[-1] / dfy.Close[-1]
    context = {
        "ticker": t,
        "predict": predictPrice,
    }
    print(context)
