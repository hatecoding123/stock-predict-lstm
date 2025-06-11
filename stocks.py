import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import FinanceDataReader as fdr
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import datetime

def get_price_data(stock_code, start_date):
    df = fdr.DataReader(stock_code, start_date)
    df = df[['Close']]
    df.rename(columns={'Close': '종가'}, inplace=True)
    df.index.name = '일자'
    return df

def preprocess_data(df, window_size=20, forecast_size=5):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(scaled) - window_size - forecast_size + 1):
        X.append(scaled[i:i+window_size])
        y.append(scaled[i+window_size:i+window_size+forecast_size].flatten())

    X, y = np.array(X), np.array(y)
    return X, y, scaler

def build_and_train_model(X, y, epochs=3000):
    model = Sequential()
    model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(y.shape[1]))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, batch_size=16, verbose=1)
    return model

def predict_next_week(model, recent_data, scaler, window_size=20):
    input_data = recent_data[-window_size:]
    input_scaled = scaler.transform(input_data)
    input_scaled = input_scaled.reshape(1, window_size, 1)
    pred_scaled = model.predict(input_scaled)
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1))
    return pred.flatten()

def plot_prediction(df, prediction):
    plt.figure(figsize=(12,6))
    
    # 실제 주가 (파란색)
    plt.plot(df.index, df['종가'], label='real price', color='blue')

    # 예측 주가 (주황색)
    last_date = df.index[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=len(prediction), freq='B')
    plt.plot(future_dates, prediction, label='expect price', color='orange', marker='o')

    # 레이블, 제목, 범례
    plt.title("expectation (LSTM)", fontsize=14)
    plt.xlabel("date", fontsize=12)
    plt.ylabel("price (won)", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def run(stock_code):
    start_date = (datetime.datetime.today() - datetime.timedelta(days=180)).strftime('%Y-%m-%d')
    print(f"{start_date} 이후 데이터 가져오는 중...")
    df = get_price_data(stock_code, start_date)
    if len(df) < 30:
        raise ValueError("데이터가 충분하지 않습니다. 최소 30일 이상의 데이터가 필요합니다.")

    X, y, scaler = preprocess_data(df)

    print("모델 학습 중...")
    model = build_and_train_model(X, y)

    print("주가 예측 중...")
    pred = predict_next_week(model, df.values, scaler)

    print("예측된 향후 5일 종가:")
    for i, price in enumerate(pred, 1):
        print(f"Day {i}: {price:,.2f} 원")

    plot_prediction(df, pred)

if __name__ == "__main__":
    user_input_code = input("종목 코드를 입력하세요 (예: 005930): ").strip()
    run(user_input_code)
