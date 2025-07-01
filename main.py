from ML_OptPathQTS1_1 import Stock

if __name__ == '__main__':
    mode = input('请选择运行模式\n\n快速模式：运行时间约为1分钟，但期望收益较低且波动较大\n精准模式：运行时间约为10分钟，但期望收益更高且波动更小\n\n快速模式请输入1,精准模式请输入2\n\n')
    strategy = Stock(mode=mode)
    strategy.run()