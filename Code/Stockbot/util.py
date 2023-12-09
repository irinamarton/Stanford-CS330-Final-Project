import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow import keras

#CEG util

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
    tickerSymbols = ['BRK-A']
    return ticker_dict, tickerSymbols


def get_control_vector(X):
    """
    Auto-labeling data for CTL (Change of Trend Labeling)

    Parameters:
    X (list): Original Time Series Data [x1, x2, x3, ..., xN]
    omega (float): Proportion threshold parameter for trend definition

    Returns:
    list: Label vector Y [label1, label2, label3, ..., labelN]
    """
    omega = 0.025
    N = len(X)
    FP = X[0]  # First Price
    xH = X[0]  # Highest Price
    HT = 0     # Time of Highest Price
    xL = X[0]  # Lowest Price
    LT = 0     # Time of Lowest Price
    Cid = 0    # Current Direction of Labeling
    FP_N = 0   # Index of initially obtained high or low point

    # First pass to find initial high/low point
    for i in range(N):
        if X[i] > FP + X[0] * omega:
            xH, HT, FP_N, Cid = X[i], i, i, 1
            break
        elif X[i] < FP - X[0] * omega:
            xL, LT, FP_N, Cid = X[i], i, i, -1
            break

    # Initialize label vector
    Y = [0] * N

    # Second pass to set labels
    for i in range(FP_N + 1, N):
        if Cid > 0:  # Upward trend
            if X[i] > xH:
                xH, HT = X[i], i
            if X[i] < xH - xH * omega and LT <= HT:
                for j in range(N):
                    if j > LT and j <= HT:
                        Y[j] = 1  # buy
                xL, LT, Cid = X[i], i, -1
        elif Cid < 0:  # Downward trend
            if X[i] < xL:
                xL, LT = X[i], i
            if X[i] > xL + xL * omega and HT <= LT:
                for j in range(N):
                    if j > HT and j <= LT:
                        Y[j] = -1  # sell
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
        if controls[i] > 0:  # buy
            for j in range(held):
                curr_roi += ((curr_price - buy_prices[j]) / buy_prices[j]) * 100
            held += 1
            buy_prices.append(curr_price)
            roi_returns.append(roi + curr_roi)
        elif (controls[i] < 0) and (held != 0):  # sell
            h = held
            for j in range(h):
                curr_roi = ((curr_price - buy_prices[0])/buy_prices[0])*100
                held -= 1
                buy_prices.pop(0)
                roi += curr_roi
            roi_returns.append(roi)
        elif (controls[i] == 0) and (held != 0):  # hold w stock
            for j in range(held):
                curr_roi += ((curr_price - buy_prices[j]) / buy_prices[j]) * 100
            roi_returns.append(roi + curr_roi)
        else: # hold no stock
            roi_returns.append(roi + curr_roi)

    return roi_returns


class LSTM_Model_MS():
    '''
    Class to train and infer stock price for a model trained on multiple stocks of a given industry. The
    list of tickers can be separately supplied to train beyond tickers from one industry.
    '''
    def __init__(self,tickerSymbol, start, end,
                 past_history = 10, forward_look = 1, train_test_split = 0.8, batch_size = 64,
                 epochs = 10, steps_per_epoch = 200, validation_steps = 50, verbose = 1, infer_train = True,
                 depth = 2, naive = False, values = 250, plot_values = True, plot_bot = True,
                 tickerSymbolList = None, sameTickerTestTrain = True):
        '''
        Initialize parameters for the class
        :param tickerSymbol: String of Ticker symbol to train on
        :param start: String of start date of time-series data
        :param end: String of end date of time-series data
        :param past_history: Int of past number of days to look at
        :param forward_look: Int of future days to predict at a time
        :param train_test_split: Float of fraction train-test split
        :param batch_size: Int of mini-batch size
        :param epochs: Int of total number of epochs in training
        :param steps_per_epoch: Int for total number of mini-batches to run over per epoch
        :param validation_steps: Int of total number of steps to use while validating with the dev set
        :param verbose: Int to decide to print training stage results
        :param infer_train: Flag to carry out prediction on training set
        :param depth: Int to decide depth of stacked LSTM
        :param naive: Flag for deciding if we need a Vanila model
        :param values: Int for number of days to predict for by iteratively updating the time-series histroy
        :param plot_values: Flag to plot
        :param plot_bot: Flag to plot the investment growth by the decision making bot
        :param tickerSymbolList: List of tickers to train the model on
        :param sameTickerTestTrain: Falg, for model containing the ticker on which predictions are made
        '''
        self.tickerSymbol = tickerSymbol
        self.start = start
        self.end = end
        self.past_history = past_history
        self.forward_look = forward_look
        self.train_test_split = train_test_split
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.verbose = verbose
        self.values = values
        self.depth = depth
        self.naive = naive
        self.custom_loss = False
        self.plot_values = plot_values
        self.plot_bot = plot_bot
        self.infer_train = infer_train
        self.sameTickerTestTrain = sameTickerTestTrain
        if tickerSymbolList == None:
            self.tickerSymbolList = [tickerSymbol]
        else:
            self.tickerSymbolList = tickerSymbolList
        tf.random.set_seed(1728)

    def data_preprocess(self, dataset, iStart, iEnd, sHistory, forward_look=1):
        '''
        Preprocess the data to make either the test set or the train set
        :param dataset: np.array of time-series data
        :param iStart: int of index start
        :param iEnd: int of index end
        :param sHistory: int number of days in history that we need to look at
        :param forward_look: int of number of days in the future that needs to predicted
        :return: returns a list of test/train data
        '''
        data = []
        target = []
        iStart += sHistory
        if iEnd is None:
            iEnd = len(dataset) - forward_look + 1
        for i in range(iStart, iEnd):
            indices = range(i - sHistory, i)  # set the order
            if forward_look > 1:
                fwd_ind = range(i, i + forward_look)
                fwd_entity = np.asarray([])
                fwd_entity = np.append(fwd_entity, dataset[fwd_ind])
            reshape_entity = np.asarray([])
            reshape_entity = np.append(reshape_entity, dataset[
                indices])  # Comment this out if there are multiple identifiers in the feature vector
            data.append(np.reshape(reshape_entity, (sHistory, 1)))  #
            if forward_look > 1:
                target.append(np.reshape(fwd_entity, (forward_look, 1)))
            else:
                target.append(dataset[i])
        data = np.array(data)
        target = np.array(target)
        return data, target


    def get_ticker_values(self):
        '''
        Get ticker values in a list
        '''
        self.y_all = []
        for tickerSymbol in self.tickerSymbolList:
            tickerData = yf.Ticker(tickerSymbol)
            tickerDf = yf.download(tickerSymbol, start=self.start, end=self.end)
            tickerDf = tickerDf['Adj Close']
            data = tickerDf
            self.y_all.append(data.values)
            self.maxTestValues = len(data.values) - int(len(data.values) * self.train_test_split)
        if self.sameTickerTestTrain == False: # This indicates self.tickerSymbol is the test ticker and self.tickerSymbolList is the training set
            tickerData = yf.Ticker(self.tickerSymbol)
            tickerDf = yf.download(self.tickerSymbol, start=self.start, end=self.end)
            tickerDf = tickerDf['Adj Close']
            data = tickerDf
            self.ytestSet = data.values
            self.maxTestValues = len(data.values) - int(len(data.values) * self.train_test_split)

        scaler = MinMaxScaler()
        all_data = np.concatenate([y.ravel() for y in self.y_all] + [self.ytestSet.ravel()])
        scaler.fit(all_data.reshape(-1, 1))
        self.y_all = [scaler.transform(y.reshape(-1,1)).ravel() for y in self.y_all]
        self.ytestSet = scaler.transform(self.ytestSet.reshape(-1, 1)).ravel()


    def prepare_test_train(self):
        '''
        Create the dataset from the extracted time-series data
        '''
        self.y_size = 0
        if self.sameTickerTestTrain == True: # For each ticker, split data into train and test set. Test and validation are the same
            self.xtrain = []
            self.ytrain = []
            self.xtest = []
            self.ytest = []
            for y in self.y_all:
                training_size = int(y.size * self.train_test_split)
                data, target = self.data_preprocess(y, 0, training_size, self.past_history, forward_look = self.forward_look)
                self.xtrain.append(data)
                self.ytrain.append(target)
                data, target = self.data_preprocess(y, training_size, None, self.past_history, forward_look = self.forward_look)
                self.xtest.append(data)
                self.ytest.append(target)
                self.y_size += y.size
            self.xtrain = np.concatenate(self.xtrain)
            self.ytrain = np.concatenate(self.ytrain)
            self.xtest = np.concatenate(self.xtest)
            self.ytest = np.concatenate(self.ytest)
            self.xt = self.xtest.copy()
            self.yt = self.ytest.copy()
        else: # For each ticker, data into train set only. Split test ticker data into validation and test sets
            self.xtrain = []
            self.ytrain = []
            self.xtest = []
            self.ytest = []
            for y in self.y_all:
                if y.size < self.values:
                    continue
                y = y[:-self.values]
                training_size = int((y.size-self.values) * (self.train_test_split))
                data, target = self.data_preprocess(y, 0, None, self.past_history, forward_look=self.forward_look)
                self.xtrain.append(data)
                self.ytrain.append(target)
                data_val = data[training_size:]
                target_val = target[training_size:]
                self.xtest.append(data_val)
                self.ytest.append(target_val)
                self.y_size += y.size
            self.xtrain = np.concatenate(self.xtrain)
            self.ytrain = np.concatenate(self.ytrain)
            self.xtest = np.concatenate(self.xtest)
            self.ytest = np.concatenate(self.ytest)

            y = self.ytestSet[-(self.values)-self.past_history:]
            print("Y SIZE", np.array(y).shape, self.past_history)
            data, target = self.data_preprocess(y, 0, None, self.past_history, forward_look=self.forward_look)
            self.xt = data
            self.yt = target
            print('TEST SHAPe', self.xt.shape, self.yt.size)


    def create_p_test_train(self):
        '''
        Prepare shuffled train and test data
        '''
        BATCH_SIZE = self.batch_size
        BUFFER_SIZE = self.y_size
        p_train = tf.data.Dataset.from_tensor_slices((self.xtrain, self.ytrain))
        self.p_train = p_train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).repeat()
        p_test = tf.data.Dataset.from_tensor_slices((self.xtest, self.ytest))
        self.p_test = p_test.batch(BATCH_SIZE).repeat()

    def model_LSTM(self):
        '''
        Create the stacked LSTM model and train it using the shuffled train set
        '''
        self.model = tf.keras.models.Sequential()
        if self.naive:
            self.model.add(tf.keras.layers.LSTM(20, input_shape = self.xtrain.shape[-2:]))
        else:
            self.model.add(tf.keras.layers.LSTM(20, return_sequences=True, input_shape = self.xtrain.shape[-2:]))
        for i in range(self.depth):
            self.model.add(tf.keras.layers.LSTM(20, return_sequences=True))
        if self.naive is False:
            self.model.add(tf.keras.layers.LSTM(20))
        self.model.add(tf.keras.layers.Dense(self.forward_look))

        self.model.compile(optimizer='Adam',
                      loss='mse', metrics=['mse'])
        self.create_p_test_train()
        self.hist = self.model.fit(self.p_train, epochs = self.epochs, steps_per_epoch = self.steps_per_epoch,
                  validation_data = self.p_test, validation_steps = self.validation_steps,
                  verbose = self.verbose)

    def infer_values(self, xtest, ytest, ts = None):
        '''
        Infer values by using the test set
        :param xtest: test dataset
        :param ytest: actual value dataset
        :param ts: tikcer symbol
        :return: model variables that store predicted data
        '''
        self.pred = []
        self.pred_update = []
        self.usetest = xtest.copy()
        if self.infer_train:
            self.pred_train = []
            self.pred_update_train = []
            self.usetest_train = self.xtrain.copy()
        # for i in range(self.values):  #  first_values
        for i in range(self.values):  # last_values
            # self.y_pred = self.model.predict(xtest[i,:,:].reshape(1,xtest.shape[1],xtest.shape[2]))[0][:]  # first_values
            self.y_pred = self.model.predict(xtest[i, :, :].reshape(1, xtest.shape[1], xtest.shape[2]))[0][:]  # last_values
            self.y_pred_update = self.model.predict(self.usetest[i,:,:].reshape(1,xtest.shape[1],xtest.shape[2]))[0][:]
            self.pred.append(self.y_pred)
        self.pred = np.array(self.pred)
        self.pred_update = np.array(self.pred_update)
        if self.infer_train:
            self.pred = np.array(self.pred)
            self.pred_update = np.array(self.pred_update)

    def plot_test_values(self):
        '''
        Plot predicted values against actual values
        '''
        plt.figure()
        plt.plot(self.pred[:50], label='predicted (%s)' % self.ts)
        # plt.plot(self.yt[:self.values-1],label='actual (%s)'%self.ts)  # first
        plt.plot(self.yt[:50], label='actual (%s)' % self.ts)  # last
        plt.xlabel("Days")
        plt.ylabel("Normalized stock price")
        plt.legend()
        plt.savefig('MultiStock_prediction_%d_%s_%d.png' % (self.past_history, self.ts, self.values))


    def full_workflow(self, model = None):
        self.get_ticker_values()
        self.prepare_test_train()
        self.model_LSTM()
        print(self.model.summary())
        if model is None:
            self.ts = self.tickerSymbol
        else:
            self.xt = model.xtest
            self.yt = model.ytest
            self.ts = model.tickerSymbol
        if self.sameTickerTestTrain == True:
            self.ts = 'Ensemble'

        self.infer_values(self.xt, self.yt, self.ts)


    def full_workflow_and_plot(self, model = None):
        '''
        Workflow to carry out the entire process end-to-end
        :param model: Choose which model to use to predict inferred values
        :return:
        '''
        self.full_workflow(model = model)
        self.plot_test_values()

    def plot_bot_decision(self):
        '''
        calculate roi from the inferred prediction value and plot the resulting growth
        '''
        # ideal = self.yt[:self.values - 1]  # [-self.values-1:-1] y_test[-300:], predicted_classed[-300:]  first
        # ideal = self.yt  # last
        ideal = self.ytestSet[-self.values:]
        ideal = ideal[:50]
        # pred = np.asarray(self.pred[1:]).reshape(-1,)  # first
        pred = np.array(self.pred).reshape(-1)  # last
        pred = pred[:50]
        print(ideal)
        control_ideal = get_control_vector(ideal)
        control_pred = get_control_vector(pred)
        print("control vectors", control_pred, control_ideal)
        bot_ideal = buy_and_sell_bot(ideal, control_ideal)
        bot_pred = buy_and_sell_bot(ideal, control_pred)
        plt.figure()
        plt.plot(bot_pred, label='From prediction (%.2f)'%bot_pred[-1])
        plt.plot(bot_ideal, label='Ideal case (%.2f)' % bot_ideal[-1])
        last = (ideal[-1]-ideal[0])/ideal[0]*100.0
        plt.plot((ideal - ideal[0]) / ideal[0] * 100.0, label='Buy and hold (%.2f)'%last)
        plt.xlabel("Days")
        plt.ylabel("Percentage ROI")
        plt.legend()
        plt.savefig('MSBot_250days_prediction_%d_%s_%d.png' % (self.past_history, self.ts, self.values))
        plt.clf()
