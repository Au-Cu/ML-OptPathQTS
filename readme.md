# 基于机器学习最优路径回溯的量化交易信号
# ML-OptPathQTS：Machine Learning-based Optimal Path Backtracing Quantitative Trading Signals

## 项目简介 / Project Introduction

基于机器学习的最优路径回溯量化交易信号系统。  
该模型以“日”为交易频率，采用贪心算法对历史股票数据进行最优路径回溯，标记出历史最优买卖点。以多项技术指标（如 MA、KDJ、RSI、BOLL、MACD）为自变量，最优买卖信号为因变量，训练随机森林模型，反推出最佳买卖点时的指标共振特征。  
最终将训练好的模型应用于独立的回测数据集，自动生成日线买卖交易信号，实现实盘级别回测及收益可视化。

**A Machine Learning-Based Optimal Path Backtracing Quantitative Trading Signal System.**  
This model operates on a daily trading frequency, utilizing a greedy algorithm to perform optimal path backtracing on historical stock data and label the optimal historical buy/sell points.  
Multiple technical indicators (such as MA, KDJ, RSI, BOLL, MACD) are used as independent variables, and the optimal buy/sell signals are used as dependent variables to train a Random Forest model.  
The trained model is then applied to an independent backtesting dataset to automatically generate daily trading signals, enabling live-level backtesting and performance visualization.

---

## 功能 Features
- 🧠 贪心算法标注最优买卖点（Greedy Algorithm for Optimal Trade Points）
- 🌲 随机森林反推交易信号（Random Forest Signal Prediction）
- 📊 完整技术指标：MA, BOLL, MACD, RSI, KDJ
- 📈 策略收益曲线与股价曲线双轴可视化
- 🔄 训练集与回测集分离，避免“上帝视角”

---

## 模型局限性 / Model Limitations

1. **模型鲁棒性较差（Low Model Robustness）：**  
模型对历史数据的依赖较大，且对训练数据区间的长度与位置较为敏感，稳定性不足，在全新市场环境中的表现存在不确定性。

The model heavily relies on historical data and is highly sensitive to the length and specific range of the training dataset. As a result, its stability is limited, and its predictive performance in completely new or unforeseen market environments remains uncertain.

2. **资金管理简化（Simplified Capital Management）：**  
最优路径回溯仅考虑了涨跌方向，未考虑涨跌幅度，导致机器学习模型输出的买卖信号缺乏强弱区分。回测策略简化为固定的“三成仓交易”——买入操作始终为账户可用资金的1/3，卖出操作为持仓股数的1/3。此外，尚未纳入券商佣金、印花税、T+1卖出限制等实际交易细节。

The optimal path backtracing process only considers the price movement direction (up or down) and ignores the magnitude of the change. Consequently, the machine learning model's output signals lack a sense of intensity or confidence.  
During the backtesting phase, all transactions are simplified to a fixed "one-third position strategy": each buy operation uses exactly one-third of the available cash, while each sell operation sells one-third of the current stock holdings.  
Moreover, realistic trading costs such as brokerage commissions, stamp duty, and T+1 selling restrictions are not yet incorporated into the strategy simulation.


3. **模型可移植性较差（Limited Model Portability）：**  
模型虽对外暴露了交易标的与账户资金接口，但当前版本已固定标的及初始资金。不同股票需要重新指定训练数据区间，且当前训练/回测区间也为手动固定，无法根据实际时间动态调整或更新。

Although the model exposes interfaces for specifying the trading asset and account balance, the current version fixes these parameters (China Bank stock, 1 million CNY initial cash).  
To apply this model to different assets or time frames, the user must manually redefine the training data range and retrain the model accordingly.  
Additionally, the training and backtesting periods are hard-coded and cannot yet dynamically adjust based on the real trading date or market updates.

---

## 后续改进方向 / Future Improvements

1. **引入区间优化算法（Training Data Range Optimization）：**  
计划采用梯度下降、模拟退火等算法，针对不同股票及时间自动寻找使回测收益率最优的训练数据区间。

In the future, algorithms such as gradient descent and simulated annealing will be introduced to automatically search for the most suitable training data period that maximizes the backtest yield for each target stock.

2. **引入强化学习（Reinforcement Learning）：**  
结合每日真实涨跌幅与昨日模型预测结果进行偏差反馈，自动调整模型参数，提升模型自适应性。

Reinforcement learning techniques will be adopted to provide reward and penalty signals based on the deviation between the model’s daily prediction and the stock’s actual price movement. This mechanism aims to enable automatic, real-time model adjustment and optimization for improved adaptability to changing market conditions.


3. **信号强度细分（Signal Strength Quantification）：**  
将在模型输出中引入买卖信号强度指标（概率或置信区间），正负决定买卖方向，绝对值决定买卖强弱，并根据信号强弱动态调整仓位比例。

The model will output not only buy/sell decisions but also the confidence level or probability of each signal. Positive values will represent a buy signal, negative values a sell signal, and the absolute magnitude of these values will indicate the strength of the signal.  
The trading volume will be dynamically adjusted based on the signal's strength rather than being fixed at one-third of available cash or holdings.

4. **模块化与自动化（Modularization & Automation）：**  
开放标的与资金接口，自动爬取当日尾盘数据，计算当日技术指标，微调训练区间，实时训练与推理，最终给出具体买卖建议（含买卖量），供用户尾盘竞价参考。

The system will be fully modularized with open interfaces for trading symbols and account balances.  
It will automatically fetch intraday stock data near the market close, compute real-time technical indicators, adjust the training window, retrain the machine learning model if needed, and output specific trading instructions (including position size recommendations) for users to execute during the closing auction period.

---

## 备注 / Notes

本项目仍处于原型开发阶段，欢迎任何贡献与反馈。  
This project is still under prototype development. Contributions and feedback are highly welcome.

---


## 使用方法 Usage

1. 安装依赖 Install Dependencies
2. 运行主程序 Run Main Program

```bash
pip install -r requirements.txt
python main.py
