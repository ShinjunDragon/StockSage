# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 14:07:07 2024

@author: KITCOOP
"""
#pip install -U finance-datareader
#pip install pykrx
#pip install forex-python
#pip install schedule


import pandas as pd
import datetime
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from pykrx import stock
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.dates as mdates
import schedule
import time
import tensorflow as tf

df_krx = fdr.StockListing('KRX','2018-01-01','2018-12-31')
df_krx.head()
df_krx.info()
len(df_krx)
today = datetime.today().strftime("%Y-%m-%d")
df = fdr.DataReader('005930', '2017-01-01',today)
lr_Close=df[["Close"]]

import tensorflow as tf
print(fdr.__version__)



df_spx=fdr.StockListing('s&P500')
df_spx.head()
len(df_spx)

df = fdr.DataReader('062040', '2024-01-01','2024-08-01')
df.head(10)

df = fdr.DataReader('068270', '2017')
df['Close'].plot()

#코스피 2015~
df = fdr.DataReader('KS11', '2015')
df['Close'].plot()

#코스닥 2015~
df = fdr.DataReader('KQ11', '2017')
df['Close'].plot()

#원달러 환율 2015~
df = fdr.DataReader('USD/KRW', '2015')
df['Close'].plot()

#비트코인 원화 가격
df = fdr.DataReader('BTC/KRW', '2016')
df['Close'].plot()

try:
    code_data = pd.read_csv('data/krx_tickers.csv', header=0, encoding='utf-8')
except UnicodeDecodeError:
    try:
        code_data = pd.read_csv('data/krx_tickers.csv', header=0, encoding='cp949')
    except UnicodeDecodeError:
        code_data = pd.read_csv('data/krx_tickers.csv', header=0, encoding='euc-kr')
code_data.head()

fdr.DataReader('001040', '2015','2017')

def get_stock_data (stock_name,start_date='2017-01-01'):
    stock_code=code_data[code_data['종목명']==stock_name]['종목코드'].values[0]
    stock_data=fdr.DataReader(stock_code,start_date)
    
    return stock_data

get_stock_data('삼성전자')


# 삼성전자의 한국 증권 거래소 티커 심볼
ticker = '068270'  # 삼성전자의 KRX 코드
# 현재 날짜와 시간 가져오기
now = (datetime.now()- timedelta(days=1)).strftime('%Y%m%d153000')

url = f'https://finance.naver.com/item/sise_time.naver?code={ticker}&thistime={now}'


# 웹 페이지 요청
response = requests.get(url,headers={'User-agent':'Mozilla/5.0'})
soup = BeautifulSoup(response.text, 'lxml')
print(soup)
#맨 뒤 페이지 클래스 pgRR
pgrr = soup.find('td',class_='pgRR')
print(pgrr)
s=pgrr.a['href'].split('=')
print(s)
#맨 뒤 페이지
last_page=s[-1]

headers ={'User-agent':'Mozilla/5.0'}
# Naver Finance URL

# 데이터 프레임 생성
data = None
for page in range(1,int(last_page)+1) :
    response = requests.get(f'{url}&page={page}',headers=headers)
    data = pd.concat([data,pd.read_html(response.text,encoding='euc-kr')[0]],ignore_index=True)

data.dropna(inplace=True)
data.reset_index(drop=True,inplace=True)

# 시간과 가격 변환
data['체결시각'] = pd.to_datetime(data['체결시각'], format='%H:%M')
data['체결가'] = pd.to_numeric(data['체결가'], errors='coerce')
data['체결시각'] = pd.to_datetime(data['체결시각'], format='%H:%M:%S')

# 변환된 datetime 객체를 matplotlib의 숫자로 변환
data['체결시각_num'] = mdates.date2num(data['체결시각'])

plt.figure(figsize=(21, 7))

# 봉 차트 그리기
for i in range(1, len(data)):
    current_time_num = data['체결시각_num'][i]
    previous_time_num = data['체결시각_num'][i - 1]
    
    # 시간 간격이 1분 이상인 경우 무시
    if (current_time_num - previous_time_num) * 1440 > 1:  # 1440은 하루의 분 수
        continue
    
    # 봉 색상 설정
    if data['체결가'][i] > data['체결가'][i - 1]:
        plt.bar(current_time_num, data['체결가'][i] - data['체결가'][i - 1],
                bottom=data['체결가'][i - 1], color='red', width=0.001)
    elif data['체결가'][i] < data['체결가'][i - 1]:
        plt.bar(current_time_num, data['체결가'][i] - data['체결가'][i - 1],
                bottom=data['체결가'][i - 1], color='blue', width=0.001)

# 그래프 제목 및 레이블 설정
plt.title(f'{ticker} 분 단위 주식 가격')
plt.xlabel('시간')
plt.ylabel('종가')

# x축 시간 포맷 설정
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.gcf().autofmt_xdate()

# 범례 추가
handles, labels = plt.gca().get_legend_handles_labels()
# 중복된 레이블 제거
unique_labels = dict(zip(labels, handles))
plt.legend(unique_labels.values(), unique_labels.keys(), loc='upper left')


# 그래프 보여주기
plt.show()







# tickers = stock.get_market_ticker_list(조회일자(YYYYmmdd) [,market=조회할 시장(KOSPI, KOSDAQ, ALL])
# name = get_market_ticker_name(심볼 [,market=조회할 시장(KOSPI, KOSDAQ, ALL])
# ohlcv = stock.stock.get_market_ohlcv(조회일자(YYYYmmdd) [,앞에 시작일을 넣었다면 종료일] [,심볼] )
# stock.stock.get_market_ohlcv(조회일자,)
# 예시
tickers = stock.get_market_ticker_list("20240801", market="KOSPI") # 코스피 종목 리스트
name = stock.get_market_ticker_name("005930") # 심볼 -> 기업 이름
ohlcv = stock.get_market_ohlcv('20180829', market="KOSDAQ") # 특정일 코스피 종목들의 OHLCV
ohlcv2 = stock.get_market_ohlcv("20170101", '20190101', "005930") # 특정기간 특정 종목의 OHLCV

# df = stock.get_market_fundamental(시작일(YYYYmmdd), 종료일(YYYYmmdd), 심볼)

# 예시
df = stock.get_market_fundamental("20220221", "20220222", "035720")

ohlcv2['종가'].plot()

#오늘 날짜
today = datetime.today().strftime("%Y%m%d")
#어제 날짜
yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
''
#등락률 top 10
top_10=ohlcv.sort_values(by='등락률',ascending=False).head(10)
top_10_tickers = top_10.index.tolist()
top_10_tickers

df = fdr.DataReader('040300', '2024-07-25', '2024-07-26')
df.shape

def MinMaxScaler(data):
    """최솟값과 최댓값을 이용하여 0 ~ 1 값으로 변환"""
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # 0으로 나누기 에러가 발생하지 않도록 매우 작은 값(1e-7)을 더해서 나눔
    return numerator / (denominator + 1e-7)

dfx = df[['Open','High','Low','Volume', 'Close']]
dfx = MinMaxScaler(dfx)
dfy = dfx[['Close']]
dfx = dfx[['Open','High','Low','Volume']]
dfy
dfx
X = dfx.values.tolist()
y = dfy.values.tolist()

window_size = 10

data_X = []
data_y = []
for i in range(len(y) - window_size):
    _X = X[i : i + window_size] # 다음 날 종가(i+windows_size)는 포함되지 않음
    _y = y[i + window_size]     # 다음 날 종가
    data_X.append(_X)
    data_y.append(_y)
print(_X, "->", _y)

train_size = int(len(data_y) * 0.7)
train_X = np.array(data_X[0 : train_size])
train_y = np.array(data_y[0 : train_size])

test_size = len(data_y) - train_size
test_X = np.array(data_X[train_size : len(data_X)])
test_y = np.array(data_y[train_size : len(data_y)])

print('훈련 데이터의 크기 :', train_X.shape, train_y.shape)
print('테스트 데이터의 크기 :', test_X.shape, test_y.shape)

model = Sequential()
model.add(LSTM(units=20, activation='relu', return_sequences=True, input_shape=(10, 4)))
model.add(Dropout(0.1))
model.add(LSTM(units=20, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(train_X, train_y, epochs=70, batch_size=30)
pred_y = model.predict(test_X)

plt.figure()
plt.plot(test_y, color='red', label='real SEC stock price')
plt.plot(pred_y, color='blue', label='predicted SEC stock price')
plt.title('SEC stock price prediction')
plt.xlabel('time')
plt.ylabel('stock price')
plt.legend()
plt.show()
print("내일 SEC 주가 :", df.Close[-1] * pred_y[-1] / dfy.Close[-1], 'KRW')
   



#1)어제 등락률 상위 10개
yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y%m%d")

ohlcv = stock.get_market_ohlcv(yesterday, market="KOSPI")
top_10=ohlcv.sort_values(by='등락률',ascending=False).head(10)
top_10_tickers = top_10.index.tolist()
top_10_tickers

#2)코스피, 코스닥
today = datetime.today().strftime("%Y-%m-%d")
months_ago= (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")

kospi = fdr.DataReader('KS11',months_ago)
kospi['Close'].plot()


kosdaq = fdr.DataReader('KQ11',months_ago)
kosdaq['Close'].plot()


#4)나라별 환전

usd_krw = fdr.DataReader('USD/KRW', today)
eur_krw = fdr.DataReader('EUR/KRW', today)
jpy_krw = fdr.DataReader('JPY/KRW', today)
cny_krw = fdr.DataReader('CNY/KRW', today)

#5)매주 ai추천

days_ago = (datetime.today() - timedelta(days=1)).strftime("%Y%m%d")
days_ago2 = (datetime.today() - timedelta(days=2)).strftime("%Y%m%d")
days_ago3 = (datetime.today() - timedelta(days=3)).strftime("%Y%m%d")
    
df_3 = stock.get_market_ohlcv(days_ago3, market="ALL")
df_2 = stock.get_market_ohlcv(days_ago2, market="ALL")
df_1 = stock.get_market_ohlcv(days_ago, market="ALL")
    
df_3 = df_3[['등락률']]
df_2 = df_2[['등락률']]
df_1 = df_1[['등락률']]
    
 # 빈 데이터프레임 생성
df_all = pd.DataFrame()
    
    # 데이터프레임 리스트
dfs = [df_1, df_2, df_3]
    
    # 빈 데이터프레임에 새로운 컬럼 추가
for i, df in enumerate(dfs, start=1):
    col_name = f'등락률_{i}일전'
        
    df_all[col_name] = df['등락률']
    
print(df_all)
df_all = df_all[(df_all['등락률_1일전'] >= 0) & (df_all['등락률_2일전'] >= 0) & (df_all['등락률_3일전'] >= 0)]
    
df_all['등락률 합'] = df_all['등락률_1일전'] + df_all['등락률_2일전'] + df_all['등락률_3일전']
    
top_5_all=df_all.sort_values(by='등락률 합',ascending=False).head(5)
top_5_all_tickers = top_5_all.index.tolist()
top_5_all_tickers
    



import statsmodels.formula.api as smf

today = datetime.today().strftime("%Y-%m-%d")
lr = fdr.DataReader('005930', '2017-01-01',today)
lr=lr[["Close"]]
lr['Date']=lr.index
lr['Day']=(lr.index - lr.index[0]).days
model = smf.ols('Close ~ Day', data=lr).fit()
train_result=model.predict(lr['Day'])
plt.plot(lr["Date"], train_result, label = "Predict")
plt.plot(lr["Date"], lr["Close"], label = "Real")
plt.xlabel("날짜")
plt.ylabel("종가")
plt.legend()
plt.title("선형회귀분석")
plt.show()


df= fdr.DataReader('068270','2017-01-01',today)


from keras.optimizers import RMSprop
# 데이터 로드
today = pd.to_datetime('today').strftime('%Y-%m-%d')
df = fdr.DataReader('068270', '2015-01-01', today)

# 데이터 전처리
def MinMaxScaler(data):
    """최솟값과 최댓값을 이용하여 0 ~ 1 값으로 변환"""
    numerator = data - np.min(data, axis=0)
    denominator = np.max(data, axis=0) - np.min(data, axis=0)
    return numerator / (denominator + 1e-7)

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

print('훈련 데이터의 크기 :', train_X.shape, train_y.shape)
print('테스트 데이터의 크기 :', test_X.shape, test_y.shape)

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

print('오차 보정 훈련 데이터의 크기 :', train_error_X.shape, train_error_y.shape)

# 오차 보정 모델 훈련
error_model.fit(train_error_X, train_error_y, epochs=70, batch_size=30, validation_split=0.1)

# 오차 예측
error_pred_y = error_model.predict(test_X)
error_pred_y = error_pred_y.flatten()

# 최종 예측 (기초 모델 + 오차 보정)
final_pred_y = initial_pred_y + error_pred_y
'''
# 시각화
plt.figure(figsize=(12, 6))

# 날짜 인덱스
test_dates = df.index[train_size + window_size:train_size + window_size + len(test_y)]
print("test_dates 크기:", len(test_dates))
print("test_y 크기:", len(test_y))
print("final_pred_y 크기:", len(final_pred_y))

plt.plot(test_dates, test_y, color='red', label='Real Stock Price')
plt.plot(test_dates, final_pred_y, color='blue', label='Final Predicted Stock Price')

plt.title('Stock Price Prediction with Error Correction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("내일 삼성 주가 :", df.Close[-1] * final_pred_y[-1] / dfy.Close[-1], 'KRW')

mape = np.mean(np.abs((test_y - final_pred_y) / test_y)) * 100
print(mape)
'''
model.save("aaa_m.h5")
error_model.save("aaa.h5")


































