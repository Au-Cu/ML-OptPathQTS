"""
量化交易策略封装类 v1.1
ML-OptPathQTS: Machine Learning-based Optimal Path Backtracing Quantitative Trading Signal
Author: Au_Cu
"""

import tushare as ts
import pandas as pd
import talib as ta
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
class Stock:
    def __init__(self, ts_code='601988.SH', initial_cash=1_000_000, mode=None):
        """
        初始化
        Initialize
        """
        self.ts_code = ts_code
        self.initial_cash = initial_cash
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.features = ['ma5', 'ma10', 'ma20', 'ma60', 'ma120', 'ma250',
                         'boll_upper', 'boll_middle', 'boll_lower', 
                         'macd', 'macd_signal', 'macd_hist', 
                         'rsi6', 'rsi12', 'rsi24', 
                         'kdj_k', 'kdj_d', 'kdj_j']
        self.pro = self.init_tushare()
        self.mode = mode
        if self.mode == '1':
            self.max_iter = 80
        elif self.mode == '2':
            self.max_iter = 800
        else:
            raise ValueError
        
    def init_tushare(self):
        """
        初始化Tushare
        Initialize Tushare
        """
        ts.set_token('87b03794207c136ab0c30071b2058ccf125cb720cf4c15ecb2cdb503')
        return ts.pro_api()

    def fetch_data(self):
        """
        获取近10年数据
        Fetch last 10 years data
        """
        print('\n正在读取股票数据\n')
        df = self.pro.daily(ts_code=self.ts_code, start_date='20150622', end_date='20250621')
        print('股票数据读取成功\n')
        return df.sort_values('trade_date').reset_index(drop=True)

    def calc_indicators(self, df):
        """
        计算技术指标
        Calculate technical indicators
        """
        print('正在计算技术指标\n')
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
                labels[i - 1] = 1
                holding = True
            elif holding and df.loc[i, 'close'] < df.loc[i - 1, 'close']:
                labels[i - 1] = -1
                holding = False
        df['label'] = labels
        print('技术指标计算完成\n')
        return df

    def train_model(self, df):
        """
        训练模型
        Train model
        """
        X = df[self.features]
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        self.model.fit(X_train, y_train)
        return classification_report(y_test, self.model.predict(X_test))

    def backtest(self, df, model=None):
        """
        策略回测
        Backtest strategy
        """
        if model is None:
            model = self.model
        df.loc[:, 'ml_signal'] = model.predict(df[self.features])
        cash, position = self.initial_cash, 0
        cash_list, position_list, total_value_list = [], [], []

        for i in range(len(df)):
            price = df.iloc[i]['close']
            signal = df.iloc[i]['ml_signal']
            if signal == 1 and cash >= price * 100:
                volume = int(cash // 3 // price // 100) * 100
                if volume > 0:
                    position += volume
                    cash -= volume * price
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

    def calculate_return(self, df):
        return df['total_value'].iloc[-1] / self.initial_cash - 1

    def simulated_annealing(self, df, initial_temp=100.0, cooling_rate=0.995):
        """
        模拟退火：局部扰动 + 概率接受差解
        """
        print('正在迭代训练模型')
        start_time = time.time()
        max_iter = self.max_iter
        
        # 回测集设置（最后3年，实际2年）
        backtest_start = len(df) - 750
        backtest_df = df.iloc[backtest_start:].copy()
        backtest_df.reset_index(drop=True, inplace=True)
        
        min_len = 500
        max_start = len(df) - 500 - min_len  # 保证训练+验证不和回测重叠
        max_len = max_start


        # 初始解（随机）
        current_start = random.randint(0, max_start)
        current_len = min(max_len, random.choice([500, 750, 1000, 1250, 1500]))  # 初始长度
        best_start, best_len = current_start, current_len

        # 计算初始收益
        train_df = df.iloc[current_start:current_start + current_len]
        X = train_df[self.features]
        y = train_df['label']
        temp_model = RandomForestClassifier(n_estimators=100, random_state=42)
        temp_model.fit(X, y)
        
        test_X = backtest_df[self.features]
        backtest_df.loc[:, 'ml_signal'] = temp_model.predict(test_X)
        bt_df = self.backtest(backtest_df.copy(), model=temp_model)
        current_return = (bt_df['total_value'].iloc[-1] / self.initial_cash - 1) * 100
        best_return = current_return

        temp = initial_temp

        for iter in range(max_iter):
            # 邻域扰动
            new_start = current_start + random.randint(-10, 10)
            new_len = current_len + random.randint(-10, 10)

            # 保证范围合法
            new_start = max(0, min(new_start, max_start))
            new_len = max(min_len, min(new_len, max_len))

            # 重新划分训练数据
            train_df = df.iloc[new_start:new_start + new_len]
            X = train_df[self.features]
            y = train_df['label']

            # 重新训练
            temp_model = RandomForestClassifier(n_estimators=100, random_state=42)
            temp_model.fit(X, y)

            # 在固定回测集上回测
            test_X = backtest_df[self.features]
            backtest_df.loc[:, 'ml_signal'] = temp_model.predict(test_X)
            bt_df = self.backtest(backtest_df.copy(), model=temp_model)
            new_return = (bt_df['total_value'].iloc[-1] / self.initial_cash - 1) * 100

            # 计算收益差（delta）
            delta = new_return - current_return

            # Metropolis准则：接受新解 or 以一定概率接受差解
            if delta > 0 or random.random() < np.exp(delta / temp):
                current_start, current_len = new_start, new_len
                current_return = new_return

            # 更新历史最优解
            if new_return > best_return:
                best_return = new_return
                best_start, best_len = new_start, new_len

            # 每输出迭代信息
            elapsed = time.time() - start_time
            print(f"迭代进度 {iter+1}/{max_iter} - 本次收益: {current_return:.2f}% - 历史最佳收益: {best_return:.2f}% - 迭代运行时间: {elapsed:.2f}s")

            # 退火（温度衰减）
            temp *= cooling_rate

        print("模型训练完成\n")
        print(f"模型训练使用历史数据区间: {df.loc[best_start, 'trade_date']} 至 {df.loc[best_start+best_len, 'trade_date']}, 区间长度: {best_len}, 收益率: {best_return:.2f}%")

        return best_start, best_len

    def plot_results(self, df):
        """
        绘制结果
        Plot results
        """
        print('正在绘制回测收益曲线，标记回测买卖点\n')
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        fig, ax1 = plt.subplots(figsize=(14,6))
        ax1.set_xlabel('Trade Date')
        ax1.set_ylabel('Close Price')
        ax1.plot(df['trade_date'], df['close'], color='gray', label='Close Price')
        buy = df[df['ml_signal'] == 1]
        sell = df[df['ml_signal'] == -1]
        ax1.scatter(buy['trade_date'], buy['close'], marker='^', color='red', label='Buy')
        ax1.scatter(sell['trade_date'], sell['close'], marker='v', color='green', label='Sell')
        ax2 = ax1.twinx()
        ax2.set_ylabel('Strategy Return')
        ax2.plot(df['trade_date'], df['total_value']/self.initial_cash - 1, color='blue', label='Strategy Yield')
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate()
        fig.legend(loc='upper left')
        plt.title('Backtest Result')
        plt.tight_layout()
        print('回测收益曲线绘制完成')
        plt.show()
        

    def run(self):
        """
        全流程执行
        Run the whole process
        """
        df = self.fetch_data()
        df = self.calc_indicators(df)
        df = self.generate_labels(df)
        best_start, best_len = self.simulated_annealing(df)

        train_df = df.iloc[best_start:best_start + best_len]
        test_df = df[-500:]  # 后3年（实际2年）

        self.model.fit(train_df[self.features], train_df['label'])
        bt_df = self.backtest(test_df.copy())
        self.plot_results(bt_df)