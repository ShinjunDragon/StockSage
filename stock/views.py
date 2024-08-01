# myapp/views.py
from django.http import JsonResponse
from django.shortcuts import render

import datetime
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
import urllib, base64
import yfinance as yf

def index(request):
    # 상위 10개

    # 날짜 설정
    today = datetime.today().strftime('%Y-%m-%d')
    yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')

    # KOSPI 주식 목록 가져오기
    all_stocks = fdr.StockListing('KOSPI')

    # 상위 10개 종목 선택 (등락률 기준)
    top_10_ChagesRatio = all_stocks.sort_values(by='ChagesRatio', ascending=False).head(10)

    # 하위 10개 종목 선택 (등락률 기준)
    row_10_ChagesRatio = all_stocks.sort_values(by='ChagesRatio', ascending=True).head(10)

    # 상위 10개 종목 선택 (시가총액 기준)
    top_10_Marcap = all_stocks.sort_values(by='Marcap', ascending=False).head(10)

    # 국가별 원 대비 환율 가져오기
    # 달러
    USD_exchange_rate = fdr.DataReader('USD/KRW', yesterday)
    # 유로
    EUR_exchange_rate = fdr.DataReader('EUR/KRW', yesterday)
    # 위안
    CHN_exchange_rate = fdr.DataReader('CHN/KRW', yesterday)
    # 엔화
    JPY_exchange_rate = fdr.DataReader('JPY/KRW', yesterday)

    # 등락 상위 10개 종목 주식 데이터 리스트로 준비
    top_10_stocks = []
    for i in range(len(top_10_ChagesRatio)):
        stock_data = {
            'code': top_10_ChagesRatio.iloc[i].Code,
            'name': top_10_ChagesRatio.iloc[i].Name,
            'close': top_10_ChagesRatio.iloc[i].Close,
            'change': top_10_ChagesRatio.iloc[i].Changes,
            'chagesRatio':top_10_ChagesRatio.iloc[i].ChagesRatio,
        }
        top_10_stocks.append(stock_data)

    # 등락 하위 10개 종목 주식 데이터 리스트로 준비
    row_10_stocks = []
    for i in range(len(row_10_ChagesRatio)):
        stock_data = {
            'code': row_10_ChagesRatio.iloc[i].Code,
            'name': row_10_ChagesRatio.iloc[i].Name,
            'close': row_10_ChagesRatio.iloc[i].Close,
            'change': row_10_ChagesRatio.iloc[i].Changes,
            'chagesRatio':row_10_ChagesRatio.iloc[i].ChagesRatio,
        }
        row_10_stocks.append(stock_data)

    # 상위 10개 종목 주식 데이터 리스트로 준비
    top_10_tot = []
    for i in range(len(top_10_Marcap)):
        stock_data = {
            'code': top_10_Marcap.iloc[i].Code,
            'name': top_10_Marcap.iloc[i].Name,
            'close': top_10_Marcap.iloc[i].Close,
            'change': top_10_Marcap.iloc[i].Changes,
            'chagesRatio':top_10_Marcap.iloc[i].ChagesRatio,
            'marcap':top_10_Marcap.iloc[i].Marcap,
        }
        top_10_tot.append(stock_data)

    context = {
        'top_10_stocks': top_10_stocks,
        'row_10_stocks': row_10_stocks,
        'top_10_tot': top_10_tot,
    }

    return render(request, 'index.html', context)

def list(request):
    # 날짜 설정
    today = datetime.today().strftime('%Y-%m-%d')
    yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')

    # KOSPI 주식 목록 가져오기
    all_stocks = fdr.StockListing('KOSPI')

    '''
    # 모든 종목의 데이터를 가져오기
    stock_symbols = all_stocks['Code'].tolist()

    # 어제와 오늘의 데이터 불러오기
    data_today = fdr.DataReader(stock_symbols, start=today, end=today)
    data_yesterday = fdr.DataReader(stock_symbols, start=yesterday, end=yesterday)    

    # 어제와 오늘의 데이터 병합 (같은 종목끼리 병합)
    merged_data = pd.merge(data_today, data_yesterday, on='Code', suffixes=('_today', '_yesterday'))    

    # 등락률 계산
    merged_data['Change'] = merged_data['Close_today'] - merged_data['Close_yesterday']
    merged_data['ChangePercent'] = (merged_data['Change'] / merged_data['Close_yesterday']) * 100
    
    # NaN 값 제거
    merged_data = merged_data.dropna()
    '''
    
    # 상위 10개 종목 선택 (등락률 기준)
    top_10 = all_stocks.sort_values(by='ChagesRatio', ascending=False).head(10)

    # 주식 데이터 리스트로 준비
    stocks = []
    for i in range(len(top_10)):
        stock_data = {
            'code': top_10.iloc[i].Code,
            'name': top_10.iloc[i].Name,
            'close': top_10.iloc[i].Close,
            'change': top_10.iloc[i].Changes,
            'chagesRatio':top_10.iloc[i].ChagesRatio,
        }
        stocks.append(stock_data)
    # 템플릿에 데이터 전달
    return render(request, 'stock/list.html', {'stocks': stocks})

def predict(request) :
    if request.method != 'POST':
        return render(request, 'stock/predict.html')

def info(request):
    ticker = request.GET.get('ticker')
    if ticker:
        try:
            # KOSPI 주식을 위해 ".KS" 추가
            kospi_ticker = f"{ticker}.KS"
            stock = yf.Ticker(kospi_ticker)
            hist = stock.history(period="1mo")  # 1달 데이터

            if hist.empty:
                return render(request, 'stock/info.html', {'error': '데이터를 가져올 수 없습니다.'})

            # 그래프 생성
            plt.figure(figsize=(10,5))
            plt.plot(hist.index, hist['Close'])
            plt.title(f"{ticker} 주가")
            plt.xlabel('날짜')
            plt.ylabel('종가')
            plt.xticks(rotation=45)
            plt.tight_layout()

            # 그래프를 이미지로 변환
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            graphic = base64.b64encode(image_png)
            graphic = graphic.decode('utf-8')

            plt.close()  # 메모리 누수 방지를 위해 plt 객체 닫기

            # 주식 정보 가져오기
            info = stock.info
            company_name = info.get('longName', 'N/A')
            current_price = info.get('currentPrice', 'N/A')
            previous_close = info.get('previousClose', 'N/A')

            return render(request, 'stock/info.html', {
                'graphic': graphic,
                'ticker': ticker,
                'company_name': company_name,
                'current_price': current_price,
                'previous_close': previous_close
            })

        except Exception as e:
            return render(request, 'stock/info.html', {'error': f'오류 발생: {str(e)}'})
    else:
        return render(request, 'stock/info.html')


def search_stocks(request):
    query = request.GET.get('query', '').strip()
    if query:
        try:
            # KOSPI 주식 목록을 가져옵니다
            all_stocks = fdr.StockListing('KOSPI')

            # 주식 코드가 query로 시작하는 항목을 필터링합니다
            results = all_stocks[all_stocks['Code'].str.startswith(query)]

            # 결과를 리스트로 변환합니다 (최대 5개)
            stocks = [{'code': row['Code'], 'name': row['Name']}
                      for _, row in results.head(5).iterrows()]

            return JsonResponse({'stocks': stocks})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'stocks': []})
