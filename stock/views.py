import pandas as pd
import numpy as np
import requests
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
import FinanceDataReader as fdr
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
import urllib, base64
import yfinance as yf
from io import BytesIO



# 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

def test(request) :

    return render(request, "test.html")

def index(request):
    # 상위 10개
    today = datetime.today().strftime('%Y-%m-%d')

    # KOSPI 주식 목록 가져오기
    kospi_stocks = fdr.StockListing('KOSPI')
    
    '''
    특정 날짜의 종목을 보는 방법
    code = '005930'
    kospi_stocks = fdr.DataReader(code, start=today)
    '''
    
    '''
    # Sector(종목별)와 industry(테마별)를 yfinance 모듈에서 가져와서 df로 만들어 csv로 저장
    # sector와 industry 정보를 추가하기 위한 빈 리스트를 준비합니다
    sectors = []
    industries = []
    
    # 각 주식의 sector와 industry 정보를 가져옵니다
    for code in kospi_stocks['Code']:
        try:
            ticker = f"{code}.KS"  # KOSPI 티커 형식
            stock = yf.Ticker(ticker)
            info = stock.info
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
        except Exception as e:
            sector = 'N/A'
            industry = 'N/A'
        
        sectors.append(sector)
        industries.append(industry)

    # 원본 DataFrame에 sector와 industry 컬럼을 추가합니다
    kospi_stocks['Sector'] = sectors
    kospi_stocks['Industry'] = industries
    
    # sectors와 industries 리스트를 따로 저장합니다
    sector_industry_df = pd.DataFrame({
        'Code': kospi_stocks['Code'],
        'Sector': sectors,
        'Industry': industries
    })
    
    # 저장할 파일 경로를 설정합니다
    sector_industry_df.to_csv('kospi_sectors_industries.csv', index=False)
    '''
    
    '''
    # Sector와 Industry 변환 사전
    sector_translation = {
        'Technology': '기술', a
        'Healthcare': '헬스케어', a
        'Financial Services': '금융 서비스', a
        'Consumer Discretionary': '소비재',
        'Consumer Staples': '필수 소비재',
        'Utilities': '공공 서비스', a
        'Energy': '에너지', a
        'Materials': '자재',
        'Real Estate': '부동산', a
        'Communication Services': '통신 서비스', a
        'Industrials': '산업' a
    }
    
    industry_translation = {
        'Communication Equipment': '통신 장비',
        'Medical Devices': '의료 기기',
        'Beverages - Brewers': '음료 - 양조',
        'Real Estate - Development': '부동산 - 개발',
        'Grocery Stores': '식료품점',
        'REIT - Residential': 'REIT - 주거',
        'Telecom Services': '통신 서비스',
        'Lodging': '숙박',
        'Railroads': '철도',
        'Beverages - Non-Alcoholic': '음료 - 비알콜',
        'Business Equipment & Supplies': '업무 장비 및 소모품',
        'Capital Markets': '자본 시장',
        'Tobacco': '담배',
        'REIT - Office': 'REIT - 사무실',
        'Tools & Accessories': '도구 및 액세서리',
        'REIT - Diversified': 'REIT - 다각화',
        'Security & Protection Services': '보안 및 보호 서비스',
        'Paper & Paper Products': '종이 및 종이 제품',
        'Department Stores': '백화점',
        'Aluminum': '알루미늄',
        'Beverages - Wineries & Distilleries': '음료 - 와이너리 및 증류소',
        'Insurance - Reinsurance': '보험 - 재보험',
        'Advertising Agencies': '광고 대행사',
        'Biotechnology': '생명공학',
        'Medical Instruments & Supplies': '의료 기기 및 소모품',
        'Auto & Truck Dealerships': '자동차 및 트럭 대리점',
        'Drug Manufacturers - General': '의약품 제조업체 - 일반',
        'Farm Products': '농산물',
        'Furnishings, Fixtures & Appliances': '가구, 비품 및 가전',
        'Entertainment': '오락',
        'Drug Manufacturers - Specialty & Generic': '의약품 제조업체 - 특수 및 제네릭',
        'Education & Training Services': '교육 및 훈련 서비스',
        'Food Distribution': '식품 유통',
        'Confectioners': '사탕 제조업체',
        'Apparel Manufacturing': '의류 제조',
        'Utilities - Regulated Gas': '공공 서비스 - 규제된 가스',
        'Internet Retail': '인터넷 소매',
        'Asset Management': '자산 관리',
        'Textile Manufacturing': '직물 제조',
        'Semiconductors': '반도체',
        'Restaurants': '레스토랑',
        'Oil & Gas Refining & Marketing': '석유 및 가스 정제 및 마케팅',
        'Discount Stores': '할인 매장',
        'REIT - Retail': 'REIT - 소매',
        'Building Materials': '건축 자재',
        'Packaged Foods': '포장 식품',
        'Packaging & Containers': '포장 및 용기',
        'Credit Services': '신용 서비스',
        'Agricultural Inputs': '농업 자재',
        'Software - Application': '소프트웨어 - 응용',
        'Airlines': '항공사',
        'Electronic Gaming & Multimedia': '전자 게임 및 멀티미디어',
        'Other Industrial Metals & Mining': '기타 산업 금속 및 채굴',
        'Financial Data & Stock Exchanges': '금융 데이터 및 증권 거래소',
        'Electronic Components': '전자 부품',
        'Engineering & Construction': '엔지니어링 및 건설',
        'Lumber & Wood Production': '목재 및 목재 생산',
        'Publishing': '출판',
        'Steel': '철강',
        'Rental & Leasing Services': '임대 및 리스 서비스',
        'Building Products & Equipment': '건축 제품 및 장비',
        'Leisure': '레저',
        'Software - Infrastructure': '소프트웨어 - 인프라',
        'Conglomerates': '대기업',
        'Consumer Electronics': '소비자 전자기기',
        'Medical Distribution': '의료 유통',
        'Marine Shipping': '해양 운송',
        'Travel Services': '여행 서비스',
        'Auto Parts': '자동차 부품',
        'Apparel Retail': '의류 소매',
        'REIT - Industrial': 'REIT - 산업',
        'Insurance - Property & Casualty': '보험 - 자산 및 상해',
        'Specialty Retail': '전문 소매',
        'Specialty Business Services': '전문 비즈니스 서비스',
        'Information Technology Services': '정보 기술 서비스',
        'Chemicals': '화학 물질',
        'Copper': '구리',
        'Utilities - Regulated Electric': '공공 서비스 - 규제된 전기',
        'Recreational Vehicles': '레크리에이션 차량',
        'Auto Manufacturers': '자동차 제조업체',
        'Industrial Distribution': '산업 유통',
        'Specialty Chemicals': '전문 화학 물질',
        'Specialty Industrial Machinery': '전문 산업 기계',
        'Computer Hardware': '컴퓨터 하드웨어',
        'Internet Content & Information': '인터넷 콘텐츠 및 정보',
        'Integrated Freight & Logistics': '통합 화물 및 물류',
        'Electrical Equipment & Parts': '전기 장비 및 부품',
        'Resorts & Casinos': '리조트 및 카지노',
        'Real Estate Services': '부동산 서비스',
        'Scientific & Technical Instruments': '과학 및 기술 기기',
        'Footwear & Accessories': '신발 및 액세서리',
        'Insurance - Diversified': '보험 - 다각화',
        'Aerospace & Defense': '항공우주 및 방산',
        'Household & Personal Products': '가정 및 개인 제품',
        'Electronics & Computer Distribution': '전자 및 컴퓨터 유통',
        'Broadcasting': '방송',
        'Metal Fabrication': '금속 가공',
        'Semiconductor Equipment & Materials': '반도체 장비 및 자재',
        'Utilities - Renewable': '공공 서비스 - 재생 가능',
        'Airports & Air Services': '공항 및 항공 서비스',
        'Banks - Regional': '은행 - 지역',
        'Insurance - Life': '보험 - 생명',
        'REIT - Hotel & Motel': 'REIT - 호텔 및 모텔',
        'Farm & Heavy Construction Machinery': '농업 및 중장비 기계',
        'Real Estate - Diversified': '부동산 - 다각화',
        'Solar': '태양광',
        'Pollution & Treatment Controls': '오염 및 처리 제어'
    }
    
    # 빈 리스트 준비
    sectors = []
    industries = []
    
    # 각 주식의 sector와 industry 정보를 가져옵니다
    for code in kospi_stocks['Code']:
        try:
            ticker = f"{code}.KS"  # KOSPI 티커 형식
            stock = yf.Ticker(ticker)
            info = stock.info
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
            
            # Sector와 Industry를 한글로 변환
            sector_korean = sector_translation.get(sector, sector)
            industry_korean = industry_translation.get(industry, industry)
            
        except Exception as e:
            sector_korean = 'N/A'
            industry_korean = 'N/A'
        
        sectors.append(sector_korean)
        industries.append(industry_korean)
    
    # 원본 DataFrame에 sector와 industry 컬럼을 추가합니다
    kospi_stocks['Sector'] = sectors
    kospi_stocks['Industry'] = industries
    
    # sectors와 industries 리스트를 따로 저장합니다
    sector_industry_df = pd.DataFrame({
        'Code': kospi_stocks['Code'],
        'Sector': sectors,
        'Industry': industries
    })
    
    # 저장할 파일 경로를 설정합니다
    sector_industry_df.to_csv('kospi_sectors_industries.csv', index=False)
    '''
    
    # 저장된 CSV 파일을 불러옵니다
    sector_industry_df = pd.read_csv('kospi_sectors_industries.csv')
    
    # kospi_stocks와 sector_industry_df를 'Code'를 기준으로 병합합니다
    kospi_stocks = pd.merge(kospi_stocks, sector_industry_df, on='Code', how='left')
    
    # 상위 15개 종목 선택 (등락률 기준)
    top_10_ChagesRatio = kospi_stocks.sort_values(by='ChagesRatio', ascending=False).head(15)

    # 하위 15개 종목 선택 (등락률 기준)
    row_10_ChagesRatio = kospi_stocks.sort_values(by='ChagesRatio', ascending=True).head(15)

    # 상위 15개 종목 선택 (시가총액 기준)
    top_10_Marcap = kospi_stocks.sort_values(by='Marcap', ascending=False).head(15)

    # 섹터별 평균 등락률 계산
    sector_avg_change = kospi_stocks.groupby('Sector')['ChagesRatio'].mean().reset_index()
    # 평균 등락률 기준으로 정렬
    sector_avg_change = sector_avg_change.sort_values(by='ChagesRatio', ascending=False)
    sector_avg_change = sector_avg_change.head(15)
    # DataFrame을 딕셔너리로 변환
    sectors_data = sector_avg_change.to_dict(orient='records')

    # industry별 평균 등락률 계산
    industry_avg_change = kospi_stocks.groupby('Industry')['ChagesRatio'].mean().reset_index()
    # 평균 등락률 기준으로 정렬
    industry_avg_change = industry_avg_change.sort_values(by='ChagesRatio', ascending=False)
    industry_avg_change = industry_avg_change.head(15)
    # DataFrame을 딕셔너리로 변환
    industries_data = industry_avg_change.to_dict(orient='records')

    # 환율정보 가져오기 (open API)
    url = 'https://www.koreaexim.go.kr/site/program/financial/exchangeJSON?authkey=DPModF9KEpqknLkTe9GxHGp8k1me31CC&searchdate=20240701&data=AP01'
    req = requests.get(url, verify=False)
    json_data = req.json()
    df = pd.DataFrame(json_data)
    eur = df.iloc[8]
    cnh = df.iloc[6]
    jpy = df.iloc[12]
    usd = df.iloc[22]
    gbp = df.iloc[9]
    dkk = df.iloc[7]
    request_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

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

    # 코스피 차트 생성
    kospi_data = yf.download('^KS11', start='2023-01-01', end=today)  # KOSPI 지수 데이터
    plt.figure(figsize=(10, 6))
    plt.plot(kospi_data.index, kospi_data['Close'], label='KOSPI Close Price')
    plt.title('KOSPI Index')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()

    # 차트를 메모리에 저장
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    image_data = buffer.getvalue()
    buffer.close()

    # 차트 이미지를 base64로 인코딩
    image_base64 = base64.b64encode(image_data).decode('utf-8')

    # 컨텍스트에 추가
    context = {
        'top_10_stocks': top_10_stocks,
        'row_10_stocks': row_10_stocks,
        'top_10_tot': top_10_tot,
        'eur': eur,
        'cnh': cnh,
        'jpy': jpy,
        'usd': usd,
        'gbp': gbp,
        'dkk': dkk,
        'request_time': request_time,
        'kospi_chart': image_base64,
        'sectors_data': sectors_data,
        'industries_data' : industries_data
    }

    return render(request, 'index.html', context)

def sector_detail(request, sector) :
    # 데이터를 불러오는 로직을 여기에 작성합니다.
    kospi_stocks = pd.read_csv('kospi_sectors_industries.csv')

    # 지정된 섹터에 대한 주식 종목 필터링
    sector_stocks = kospi_stocks[kospi_stocks['Sector'] == sector]

    # 등락률 기준으로 내림차순 정렬 후 상위 15개 항목 선택
    top_stocks = sector_stocks.sort_values(by='ChagesRatio', ascending=False).head(15)

    # 컨텍스트를 설정하여 템플릿에 전달
    context = {
        'sector_name': sector,
        'top_stocks': top_stocks.to_dict(orient='records'),
    }

    return render(request, "stock/sector_detail.html", context)

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

def MinMaxScaler(data):
    """최솟값과 최댓값을 이용하여 0 ~ 1 값으로 변환"""
    numerator = data - np.min(data, axis=0)
    denominator = np.max(data, axis=0) - np.min(data, axis=0)
    return numerator / (denominator + 1e-7)

def predict(request, window_size=10) :
    ticker = request.GET.get('ticker')
    if ticker :
        # 데이터 로드
        today = pd.to_datetime('today').strftime('%Y-%m-%d')
        df = fdr.DataReader(ticker, '2015-01-01', today)
        dfx = df[['Open', 'High', 'Low', 'Volume', 'Close']]
        dfx = MinMaxScaler(dfx)
        dfy = dfx[['Close']]
        dfx = dfx[['Open', 'High', 'Low', 'Volume']]
        X = dfx.values
        y = dfy.values

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
        test_X = data_X[train_size:]
        test_y = data_y[train_size:]

        # 기초 모델 로드
        basic_model = load_model(f'{ticker}_basic_model.keras')

        # 초기 예측
        initial_pred_y = basic_model.predict(test_X)
        initial_pred_y = initial_pred_y.flatten()
        test_y = test_y.flatten()

        # 오차 보정 모델 로드
        error_model = load_model(f'{ticker}_error_correction_model.keras')

        # 오차 예측
        error_pred_y = error_model.predict(test_X)
        error_pred_y = error_pred_y.flatten()

        # 최종 예측 (기초 모델 + 오차 보정)
        final_pred_y = initial_pred_y + error_pred_y

        # 예측 가격 계산
        predictPrice = df.Close[-1] * final_pred_y[-1] / dfy.Close[-1]

        context = {
            'predict' : predictPrice,
        }

        return render(request, "stock/predict.html", context)
    else :
        return render(request, "stock/predict.html")

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

            # 주식 코드가 query로 시작하거나 주식 이름에 query가 포함된 항목을 필터링합니다
            results = all_stocks[
                (all_stocks['Code'].str.startswith(query)) |
                (all_stocks['Name'].str.contains(query, case=False, na=False))
                ]

            # 결과를 리스트로 변환합니다 (최대 5개)
            stocks = [{'code': row['Code'], 'name': row['Name']}
                      for _, row in results.head(5).iterrows()]
            print(stocks)

            return JsonResponse({'stocks': stocks})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)



def get_flag_image(request, country_code):
    # 국가 코드에 따라 플래그 이미지 URL 생성
    url = f"https://flagcdn.com/w320/{country_code.lower()}.png"
    
    try:
        # 이미지 다운로드
        response = requests.get(url)
        response.raise_for_status()
        
        # 이미지 파일을 HttpResponse로 반환
        return HttpResponse(response.content, content_type="image/png")
    except requests.RequestException as e:
        # 이미지 요청에 실패하면 에러 메시지를 반환
        return HttpResponse(f"Error fetching image: {e}", status=500)
    
    
