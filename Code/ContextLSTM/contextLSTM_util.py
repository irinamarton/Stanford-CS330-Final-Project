import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def get_categorical_tickers():
    '''
    This Function returns a dictionary of tickers for different industry types
    :return:
    ticker_dict: Dictionary of 9 different industry types with over 8 tickers each
    tickerSymbols: Set of three tickers
    '''
    ticker_dict = {}
    all_tickers = []
    ticker_dict['test'] = ['AAPL']
    ticker_dict['utilities'] = ['AEP', 'CEG', 'EVRG', 'EXC', 'LNT', 'XEL']
    ticker_dict['energy'] = ['APA', 'CHKEW', 'FANG', 'PAA']
    ticker_dict['industrials'] = ['AXON', 'BKR', 'CSX', 'DASH', 'FOXA', 'FWONA', 'HON', 'JBHT', 'LECO', 'LIN', 'NDSN',
                                  'ODFL', 'PARAA', 'ROP', 'SAIA', 'STLD', 'TER', 'TRMB']
    ticker_dict['staples'] = ['CCEP', 'CELH', 'KDP', 'KHC', 'MDLZ', 'MNST', 'PEP', 'WBA']
    ticker_dict['discretionary'] = ['ABNB', 'AKAM', 'AMZN', 'BKNG', 'CASY', 'CDW', 'COST', 'CPRT', 'CTAS', 'DKNG',
                                    'DLTR', 'EBAY', 'EXPD', 'EXPE', 'FAST', 'GRAB', 'HTHT', 'JD', 'LI', 'LKQ', 'LULU',
                                    'MAR', 'MELI', 'NFLX', 'NWSA', 'ORLY', 'PAYX', 'PCAR', 'PDD', 'POOL', 'PYPL',
                                    'QRTEP', 'RIVN', 'ROST', 'RYAAY', 'SBUX', 'SIRI', 'TCOM', 'TSCO', 'TSLA', 'UAL',
                                    'ULTA', 'VRSK', 'WMG']
    ticker_dict['realestate'] = ['AGNCN', 'EQIX', 'GLPI', 'HST', 'REG', 'SBAC']
    ticker_dict['financials'] = ['ACGL', 'ARCC', 'BPYPP', 'CG', 'CINF', 'CME', 'COIN', 'CSGP', 'ERIE', 'FCNCA', 'FITB',
                                 'HBAN', 'LPLA', 'MORN', 'NDAQ', 'NTRS', 'PFG', 'SLMBP', 'TROW', 'TW', 'VLYPO',
                                 'WTW', 'XP']
    ticker_dict['healthcare'] = ['ALGN', 'ALNY', 'AMGN', 'ARGX', 'AZN', 'BGNE', 'BIIB', 'BMRN', 'BNTX', 'COO', 'DXCM',
                                 'EXAS', 'GILD', 'HOLX', 'ICLR', 'IDXX', 'ILMN', 'INCY', 'ISRG', 'LEGN', 'MRNA', 'NBIX',
                                 'PODD', 'REGN', 'RPRX', 'SGEN', 'SNY', 'UTHR', 'VTRS']
    ticker_dict['telecommunication'] = ['CHTR', 'CMCSA', 'CSCO', 'LBRDA', 'ROKU', 'TMUS', 'VOD', 'WBD']
    ticker_dict['tech'] = ['ADBE', 'ADI', 'ADP', 'ADSK', 'AMAT', 'AMD', 'ANSS', 'APP', 'ASML', 'AVGO', 'AZPN',
                           'BIDU', 'BSY', 'CDNS', 'CHKP', 'CRWD', 'CTSH', 'DDOG', 'EA', 'ENPH', 'ENTG', 'ERIC', 'FLEX',
                           'FSLR', 'FTNT', 'GEN', 'GFS', 'GOOGL', 'INTC', 'INTU', 'JKHY', 'KLAC', 'LOGI', 'LRCX',
                           'MANH', 'MCHP', 'MDB', 'META', 'MPWR', 'MRVL', 'MSFT', 'MU', 'NICE', 'NTAP', 'NTES', 'NVDA',
                           'NXPI', 'OKTA', 'ON', 'PANW', 'PTC', 'QCOM', 'SMCI', 'SNPS', 'SPLK', 'SSNC', 'STX', 'SWKS',
                           'TEAM', 'TTD', 'TTWO', 'TXN', 'VRSN', 'VRTX', 'WDAY', 'WDC', 'ZBRA', 'ZM', 'ZS']

    ticker_keys = []
    for key in ticker_dict.keys():
        ticker_keys.append(key)
        all_tickers.append(ticker_dict[key])
    ticker_dict['all'] = all_tickers
    tickerTest = ['BRK-A']
    return ticker_dict, tickerTest


def get_ticker_values(tickerSymbolList, start, end):
    '''
    Get ticker values in a list
    '''
    x_all = []
    for tickerSymbol in tickerSymbolList:
        tickerDf = yf.download(tickerSymbol, start=start, end=end)
        tickerDf = tickerDf['Adj Close']
        #print("tickerDf", len(tickerDf))
        data = tickerDf.values
        #print("ticker data", data)
        x_all.append(data)

    scaler = MinMaxScaler()

    all_data = np.concatenate([x.ravel() for x in x_all])
    scaler.fit(all_data.reshape(-1, 1))
    x_all_transformed = [scaler.transform(x.reshape(-1, 1)).ravel() for x in x_all]
    x_all_transformed = np.array(x_all_transformed)
    return x_all_transformed


def data_w_context(data, target):
    context_data = []
    for i in range(len(data)-1):
        x = np.vstack([data[i].tolist(), [[target[i]]], data[i+1].tolist()])
        context_data.append(x)
    context_target = target[1:]
    context_data = np.array(context_data)

    return context_data, context_target


def data_preprocess_HL_labels(dataset, iStart, iEnd, sHistory):
    data = []
    iStart += sHistory
    if iEnd is None:
        iEnd = len(dataset)
    for i in range(iStart, iEnd):
        indices = range(i - sHistory, i)  # set the order
        reshape_entity = np.asarray([])
        reshape_entity = np.append(reshape_entity, dataset[
            indices])  # Comment this out if there are multiple identifiers in the feature vector
        data.append(np.reshape(reshape_entity, (sHistory, 1)))  #
    data = np.array(data)

    indices_labels = range(iStart, iEnd)
    prices_labels = dataset[indices_labels]
    target = auto_label_data(prices_labels)
    target = np.array(target)

    return data, target


def auto_label_data(X, omega=0.025):
    """
    Auto-labeling data for CTL (Change of Trend Labeling)

    Parameters:
    X (list): Original Time Series Data [x1, x2, x3, ..., xN]
    omega (float): Proportion threshold parameter for trend definition

    Returns:
    list: Label vector Y [label1, label2, label3, ..., labelN]
    """

    N = len(X)
    FP = X[0]  # First Price
    xH = X[0]  # Highest Price
    HT = 0  # Time of Highest Price
    xL = X[0]  # Lowest Price
    LT = 0  # Time of Lowest Price
    Cid = 0  # Current Direction of Labeling
    FP_N = 0  # Index of initially obtained high or low point

    # First pass to find initial high/low point
    for i in range(N):
        if X[i] > FP + X[0] * omega:
            xH, HT, FP_N, Cid = X[i], i, i, 1
            break
        elif X[i] < FP - X[0] * omega:
            xL, LT, FP_N, Cid = X[i], i, i, -1
            break

    # Initialize label vector
    Y = [2] * N

    # Second pass to set labels
    for i in range(FP_N + 1, N):
        if Cid > 0:  # Upward trend
            if X[i] > xH:
                xH, HT = X[i], i
            if X[i] < xH - xH * omega and LT <= HT:
                for j in range(N):
                    if j > LT and j <= HT:
                        Y[j] = 0  # buy
                xL, LT, Cid = X[i], i, -1
        elif Cid < 0:  # Downward trend
            if X[i] < xL:
                xL, LT = X[i], i
            if X[i] > xL + xL * omega and HT <= LT:
                for j in range(N):
                    if j > HT and j <= LT:
                        Y[j] = 1  # sell
                xH, HT, Cid = X[i], i, 1

    return Y


def buy_and_sell_bot(val, controls):
    '''
    Returns the growth of investment over time as function of the input decision mask and the stock values
    :param val: np.array of the actual stock value over time
    :param controls: np.array of the control mask to make purchase/sell decisions
    :return: np.array of percentage growth value of the invested stock
    '''
    held = 0
    roi = 0.0
    buy_prices = []
    roi_returns = []

    for i in range(len(controls)):
        curr_price = val[i]
        curr_roi = 0.0
        if controls[i] == 0:  # buy
            for j in range(held):
                curr_roi += ((curr_price - buy_prices[j]) / buy_prices[j]) * 100
            held += 1
            buy_prices.append(curr_price)
            roi_returns.append(roi + curr_roi)
        elif (controls[i] == 1) and (held != 0): # sell
            h = held
            for j in range(h):
                curr_roi = ((curr_price - buy_prices[0])/buy_prices[0])*100
                held -= 1
                buy_prices.pop(0)
                roi += curr_roi
            roi_returns.append(roi)
        elif (controls[i] == 2) and (held != 0): # hold w stock
            for j in range(held):
                curr_roi += ((curr_price - buy_prices[j]) / buy_prices[j]) * 100
            roi_returns.append(roi + curr_roi)
        else: # hold no stock
            roi_returns.append(roi + curr_roi)

    return roi_returns


def plot_bot_decision(val, y, predictions):
    '''
    calculate roi from the inferred prediction value and plot the resulting growth
    '''
    bot_pred = buy_and_sell_bot(val, predictions)
    bot_ideal = buy_and_sell_bot(val, y)
    print("LEN BOTS", np.array(bot_ideal).size, np.array(bot_pred).size)
    print("actions Y", y)
    print("action PRED", predictions)
    print("val", val)
    plt.figure()
    plt.plot(bot_pred, label='From prediction (%.2f)'%bot_pred[-1])
    plt.plot(bot_ideal, label='Ideal case (%.2f)' % bot_ideal[-1])
    last = (val[-1]-val[0])/val[0]*100.0
    plt.plot((val - val[0]) / val[0] * 100.0, label='Buy and hold (%.2f)'%last)
    plt.xlabel("Days")
    plt.ylabel("Percentage ROI")
    plt.legend()
    plt.savefig('MSBot_prediction.png')
    plt.clf()