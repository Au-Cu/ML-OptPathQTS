"""
量化交易策略封装类
ML-OptPathQTS:Machine Learning-based Optimal Path Backtracing Quantitative Trading Signal
Author: Au_Cu
"""

import tushare as ts
import pandas as pd
import talib as ta
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class Stock:
    def __init__(self, ts_code='601988.SH', initial_cash=1_000_000):
        """
        初始化
        Initialize
        """
        self.ts_code = ts_code
        self.start_train = '20210622'
        self.end_train = '20240621'
        self.start_test = '20220622'
        self.end_test = '20250621'
        self.initial_cash = initial_cash
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.features = ['ma5', 'ma10', 'ma20', 'ma60', 'ma120', 'ma250',
                         'boll_upper', 'boll_middle', 'boll_lower', 
                         'macd', 'macd_signal', 'macd_hist', 
                         'rsi6', 'rsi12', 'rsi24', 
                         'kdj_k', 'kdj_d', 'kdj_j']
        self.pro = self.init_tushare()

    def init_tushare(self):
        """
        初始化Tushare
        Initialize Tushare
        """
        ts.set_token('87b03794207c136ab0c30071b2058ccf125cb720cf4c15ecb2cdb503')
        return ts.pro_api()

    def fetch_data(self, start_date, end_date):
        """
        获取股票数据
        Fetch stock data
        """
        df = self.pro.daily(ts_code=self.ts_code, start_date=start_date, end_date=end_date)
        return df.sort_values('trade_date').reset_index(drop=True)

    def calc_indicators(self, df):
        """
        计算技术指标
        Calculate technical indicators
        """
        df['ma5'] = ta.SMA(df['close'], 5)
        df['ma10'] = ta.SMA(df['close'], 10)
        df['ma20'] = ta.SMA(df['close'], 20)
        df['ma60'] = ta.SMA(df['close'], 60)
        df['ma120'] = ta.SMA(df['close'], 120)
        df['ma250'] = ta.SMA(df['close'], 250)
        df['boll_upper'], df['boll_middle'], df['boll_lower'] = ta.BBANDS(df['close'], 20)
        df['macd'], df['macd_signal'], df['macd_hist'] = ta.MACD(df['close'], 12, 26, 9)
        df['rsi6'] = ta.RSI(df['close'], 6)
        df['rsi12'] = ta.RSI(df['close'], 12)
        df['rsi24'] = ta.RSI(df['close'], 24)
        low_list = df['low'].rolling(9, min_periods=9).min()
        high_list = df['high'].rolling(9, min_periods=9).max()
        rsv = (df['close'] - low_list) / (high_list - low_list) * 100
        df['kdj_k'] = rsv.ewm(com=2).mean()
        df['kdj_d'] = df['kdj_k'].ewm(com=2).mean()
        df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
        return df.dropna().reset_index(drop=True)

    def generate_labels(self, df):
        """
        贪心算法生成标签
        Generate labels using greedy algorithm
        """
        labels = [0] * len(df)
        holding = False
        for i in range(1, len(df)):
            if not holding and df.loc[i, 'close'] > df.loc[i - 1, 'close']:
                labels[i - 1] = 1  # Buy yesterday
                holding = True
            elif holding and df.loc[i, 'close'] < df.loc[i - 1, 'close']:
                labels[i - 1] = -1  # Sell yesterday
                holding = False
        df['label'] = labels
        return df

    def train_model(self, df):
        """
        训练随机森林模型
        Train Random Forest model
        """
        X = df[self.features]
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        self.model.fit(X_train, y_train)
        print("Classification Report:\n", classification_report(y_test, self.model.predict(X_test)))

    def backtest(self, df):
        """
        策略回测
        Backtest strategy
        """
        df['ml_signal'] = self.model.predict(df[self.features])
        cash = self.initial_cash
        position = 0
        cash_list, position_list, total_value_list = [], [], []

        for i in range(len(df)):
            price = df.loc[i, 'close']
            signal = df.loc[i, 'ml_signal']
            if signal == 1 and cash >= price * 100:
                volume = int(cash // 3 // price // 100) * 100
                cost = volume * price
                if volume > 0:
                    position += volume
                    cash -= cost
            elif signal == -1 and position > 0:
                volume = int(position // 3 // 100) * 100
                cash += volume * price
                position -= volume
            total_value = cash + position * price
            cash_list.append(cash)
            position_list.append(position)
            total_value_list.append(total_value)

        df['cash'] = cash_list
        df['position'] = position_list
        df['total_value'] = total_value_list
        return df

    def plot_results(self, df):
        """
        绘制回测结果
        Plot backtest results
        """
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        fig, ax1 = plt.subplots(figsize=(14,6))
        ax1.set_xlabel('Trade Date')
        ax1.set_ylabel('Close Price', color='gray')
        ax1.plot(df['trade_date'], df['close'], color='gray', label='Close Price')
        buy = df[df['ml_signal'] == 1]
        sell = df[df['ml_signal'] == -1]
        ax1.scatter(buy['trade_date'], buy['close'], marker='^', color='red', label='Buy', zorder=5)
        ax1.scatter(sell['trade_date'], sell['close'], marker='v', color='green', label='Sell', zorder=5)
        ax2 = ax1.twinx()
        ax2.set_ylabel('Strategy Return', color='blue')
        ax2.plot(df['trade_date'], df['total_value']/self.initial_cash -1, color='blue', label='Strategy Yield')
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate()
        fig.legend(loc='upper left')
        plt.title('Machine Learning Strategy vs Close Price & Equity Curve')
        plt.tight_layout()
        plt.show()

    def run(self):
        """
        全流程执行
        Run the whole process
        """
        df_train = self.calc_indicators(self.fetch_data(self.start_train, self.end_train))
        df_train = self.generate_labels(df_train)
        self.train_model(df_train)
        df_test = self.calc_indicators(self.fetch_data(self.start_test, self.end_test))
        df_test = self.backtest(df_test)
        self.plot_results(df_test)