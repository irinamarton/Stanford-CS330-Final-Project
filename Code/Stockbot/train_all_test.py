import sys, os
sys.path.append('../python')
from util import *


ticker_dict, tickerSymbols = get_categorical_tickers()
start="2011-01-01"
end="2023-06-30"

tickerList = ticker_dict['all']
tickerList = [item for sublist in tickerList for item in sublist]

tickeranalysis = tickerList[0] # this should be a string
tickerList.remove(tickeranalysis)  #ticker removed from training
print(tickeranalysis in tickerList)

stockbot = LSTM_Model_MS(tickerSymbol = tickeranalysis, start = start, end = end, depth = 2, naive = False,
                       tickerSymbolList = tickerList, sameTickerTestTrain = False)

stockbot.full_workflow_and_plot()
stockbot.plot_bot_decision()

