import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, DataLoader
import util_MetaContext as util

NUM_TRAIN_TICKERS = 168
NUM_VAL_TICKERS = 47
NUM_TEST_TICKERS = 1
NUM_SAMPLES_PER_CLASS = 20
CONTEXT_DAYS = 10


class StocksDataset(Dataset):
    """Stocks dataset for meta-learning.

    Each element of the dataset is a task. A task is specified with a key,
    which is a tuple of class indices (no particular order). The corresponding
    value is the instantiated task, which consists of sampled (image, label)
    pairs.
    """

    def __init__(self, tickers, split, start_date, end_date, num_support, num_query):
        """
        Inits StockMarketDataset.
        Args:
            tickers (list): List of ticker symbols.
            start_date (str): Start date for the data.
            end_date (str): End date for the data.
            num_support (int): Number of support examples per ticker.
            num_query (int): Number of query examples per ticker.
        """
        super().__init__()

        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

        data = util.get_ticker_values(tickers, start_date, end_date)

        if split == 'train':
            self.data = data[:NUM_TRAIN_TICKERS]
        elif split == 'val':
            self.data = data[NUM_TRAIN_TICKERS:NUM_TRAIN_TICKERS + NUM_VAL_TICKERS]
        elif split == 'test':
            self.data = data[NUM_TRAIN_TICKERS + NUM_VAL_TICKERS:]
        else:
            raise ValueError("Invalid split type. Choose from 'train', 'val', 'test'.")

        # shuffle tickers
        np.random.default_rng(0).shuffle(self.data)

        # check problem arguments
        assert num_support + num_query <= NUM_SAMPLES_PER_CLASS
        self.num_support = num_support
        self.num_query = num_query
        self.num_samples = self.num_support+self.num_query

    def __getitem__(self, ticker_idxs):
        """
                Constructs a task for stock market prediction.
                Args:
                    ticker_idxs (list[int]): Ticker indices that comprise the task.
                Returns:
                    Tuple of tensors: support data, support labels, query data, query labels
                """

        data_support, data_query = [], []
        labels_support, labels_query = [], []

        random = False

        for _, ticker_idx in enumerate(ticker_idxs):
            ticker_data = self.data[ticker_idx][:-250]  # train
            # ticker_data = self.data[ticker_idx][-250-CONTEXT_DAYS-1:]  #inc learning
            # ticker_data = ticker_data[:25+CONTEXT_DAYS+1]  #inc learning

            data, labels = util.data_preprocess_HL_labels(ticker_data, 0, None, CONTEXT_DAYS)
            context_data, context_labels = util.data_w_context(data, labels)

            if random is False:
                ### TASK B

                query_idx = np.random.randint(3, len(context_labels))

                data_support.extend([context_data[query_idx - 3],
                                     context_data[query_idx - 2],
                                     context_data[query_idx - 1]])
                labels_support.extend([context_labels[query_idx - 3],
                                       context_labels[query_idx - 2],
                                       context_labels[query_idx - 1]])

                q_data = [context_data[query_idx]]
                q_label = context_labels[query_idx]
                data_query.extend(q_data)
                labels_query.append(q_label)
            else:
                ### TASK A

                query_idx = np.random.randint(3, len(context_labels) - 1)
                data_query.extend([context_data[query_idx]])
                labels_query.append(context_labels[query_idx])

                context_data.pop(query_idx)
                context_labels = np.delete(context_labels, query_idx)

                idx_0 = [i for i, x in enumerate(context_labels) if x == 0]
                idx_1 = [i for i, x in enumerate(context_labels) if x == 1]
                idx_2 = [i for i, x in enumerate(context_labels) if x == 2]

                if (len(idx_0) == 0) or (len(idx_1) == 0) or (len(idx_2) == 0):
                    print("NO EXAMPLES LEFT", len(idx_0), len(idx_1), len(idx_2))
                    raise ValueError("!!!MISSING CLASS EXAMPLES!!!")

                list_of_0_data = [context_data[i] for i in idx_0]
                list_of_1_data = [context_data[i] for i in idx_1]
                list_of_2_data = [context_data[i] for i in idx_2]

                if len(idx_0) == 1:
                    supp_idx_0 = 0
                else:
                    supp_idx_0 = np.random.randint(0, len(idx_0))
                if len(idx_1) == 1:
                    supp_idx_1 = 0
                else:
                    supp_idx_1 = np.random.randint(0, len(idx_1))
                if len(idx_2) == 1:
                    supp_idx_2 = 0
                else:
                    supp_idx_2 = np.random.randint(0, len(idx_2))

                data_support.extend([list_of_0_data[supp_idx_0], list_of_1_data[supp_idx_1], list_of_2_data[supp_idx_2]])
                labels_support.extend([0, 1, 2])

        data_support = torch.stack(data_support)
        labels_support = torch.tensor(labels_support)
        data_query = torch.stack(data_query)
        labels_query = torch.tensor(labels_query)

        return (data_support, labels_support, data_query, labels_query)

    def __len__(self):
        return self.num_samples


class StocksSampler(Sampler):
    """
    Samples task specification keys for a StockMarketDataset with context.
    """

    def __init__(self, num_tickers, num_way, num_tasks):
        """
        Inits StockMarketSampler.
        Args:
            num_tickers (int): Total number of tickers.
            num_way (int): Number of tickers per task.
            num_tasks (int): Number of tasks to sample.
        """
        super().__init__(None)
        self.num_tickers = num_tickers
        self.num_way = num_way
        self.num_tasks = num_tasks

    def __iter__(self):
        return (np.random.choice(self.num_tickers, 1, replace=False)
                for _ in range(self.num_tasks))

    def __len__(self):
        return self.num_tasks


def get_stock_market_dataloader(
        batch_size,
        tickers,
        split,
        start_date,
        end_date,
        num_way,
        num_support,
        num_query,
        num_tasks_per_epoch,
        num_workers=2):
    """
    Returns a DataLoader for Stock Market Data with context.
    Args:
        tickers (list): List of ticker symbols.
        split (str): one of 'train', 'val', 'test'
        start_date (str): Start date for the data.
        end_date (str): End date for the data.
        batch_size (int): Number of tasks per batch.
        num_way (int): Number of tickers per task.
        num_support (int): Number of support examples per ticker.
        num_query (int): Number of query examples per ticker.
        num_tasks_per_epoch (int): Number of tasks before DataLoader is exhausted.
    """
    if split == 'train':
        split_tickers = tickers[:NUM_TRAIN_TICKERS]
    elif split == 'val':
        split_tickers = tickers[NUM_TRAIN_TICKERS:NUM_TRAIN_TICKERS + NUM_VAL_TICKERS]
    elif split == 'test':
        split_tickers = tickers[NUM_TRAIN_TICKERS + NUM_VAL_TICKERS:]
    else:
        raise ValueError("Invalid split type. Choose from 'train', 'val', 'test'.")

    dataset = StocksDataset(tickers, split, start_date, end_date, num_support, num_query)
    sampler = StocksSampler(len(split_tickers), num_way, num_tasks_per_epoch)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

### TEST DATASET LOADING ---------------------------------------------------------
