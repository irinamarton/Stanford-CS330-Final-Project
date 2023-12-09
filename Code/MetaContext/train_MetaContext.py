"""Implementation of model-agnostic meta-learning for stock data classification."""
import sys
sys.path.append('..')
import argparse
import os

import numpy as np
import torch

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch import autograd
from torch.utils import tensorboard
from google_drive_downloader import GoogleDriveDownloader as gdd
from sklearn.metrics import classification_report

import stocks
import util_MetaContext as util

NUM_INPUT_CHANNELS = 1
KERNEL_SIZE = 3
SUMMARY_INTERVAL = 10
SAVE_INTERVAL = 50
LOG_INTERVAL = 10
VAL_INTERVAL = LOG_INTERVAL * 5
NUM_TEST_TASKS = 250
CONTEXT_DAYS = 10

class MAMLLSTM:
    """ MAML model for time series forecasting with LSTM """

    def __init__(
            self,
            num_outputs,
            num_inner_steps,
            inner_lr,
            learn_inner_lrs,
            outer_lr,
            log_dir,
            device
    ):
        """Inits MAML.

        The network consists of four convolutional blocks followed by a linear
        head layer. Each convolutional block comprises a convolution layer, a
        batch normalization layer, and ReLU activation.

        Note that unlike conventional use, batch normalization is always done
        with batch statistics, regardless of whether we are training or
        evaluating. This technically makes meta-learning transductive, as
        opposed to inductive.

        Args:
            num_outputs (int): dimensionality of output, i.e. number of classes
                in a task
            num_inner_steps (int): number of inner-loop optimization steps
            inner_lr (float): learning rate for inner-loop optimization
                If learn_inner_lrs=True, inner_lr serves as the initialization
                of the learning rates.
            learn_inner_lrs (bool): whether to learn the above
            outer_lr (float): learning rate for outer-loop optimization
            log_dir (str): path to logging directory
            device (str): device to be used
        """
        meta_parameters = {}
        self.device = device
        self.num_outputs = num_outputs


        # LSTM specific parameters
        self.hidden_size = 20
        self.input_size = 1
        self.num_layers = 4

        # Initialize LSTM parameters
        for i in range(self.num_layers):
            # Adjust the input size for subsequent layers
            layer_input_size = self.hidden_size if i > 0 else self.input_size

            w_ih = f'weight_ih_l{i}'
            w_hh = f'weight_hh_l{i}'
            b_ih = f'bias_ih_l{i}'
            b_hh = f'bias_hh_l{i}'

            meta_parameters[w_ih] = nn.init.xavier_uniform_(
                torch.empty(
                    4 * self.hidden_size,
                    layer_input_size,
                    requires_grad=True,
                    device=self.device))
            meta_parameters[w_hh] = nn.init.xavier_uniform_(
                torch.empty(
                    4 * self.hidden_size,
                    self.hidden_size,
                    requires_grad=True,
                    device=self.device))
            meta_parameters[b_ih] = nn.init.zeros_(
                torch.empty(
                    4 * self.hidden_size,
                    requires_grad=True,
                    device=self.device))
            meta_parameters[b_hh] = nn.init.zeros_(
                torch.empty(
                    4 * self.hidden_size,
                    requires_grad=True,
                    device=device))

        # construct linear layer
        meta_parameters['linear_w'] = nn.init.xavier_uniform_(
            torch.empty(
                self.num_outputs,
                self.hidden_size,
                requires_grad=True,
                device=self.device))
        meta_parameters['linear_b'] = nn.init.zeros_(
            torch.empty(
                self.num_outputs,
                requires_grad=True,
                device=self.device))


        self._meta_parameters = meta_parameters
        self._num_inner_steps = num_inner_steps
        self._inner_lrs = {
            k: torch.tensor(inner_lr, requires_grad=learn_inner_lrs, device=device)
            for k in self._meta_parameters.keys()
        }

        self._outer_lr = outer_lr

        # edited optimizer
        self._optimizer = torch.optim.Adam(
            list(self._meta_parameters.values()) +
            list(self._inner_lrs.values()),
            lr=self._outer_lr
        )

        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)

        self._start_train_step = 0

    def _forward(self, data, parameters):
        """Computes predicted classification logits.

        Args:
            data (Tensor): batch of data
                shape (num_ways, sequence_len, 1)
            parameters (dict[str, Tensor]): parameters to use for
                the computation

        Returns:
            a Tensor consisting of a batch of logits
                shape (num_images, classes)
        """
        num_ways, sequence_length, _ = data.size()

        # Initialize LSTM cell
        lstm_layers = nn.LSTM(input_size=self.input_size,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers)
        lstm_layers.to(self.device)

        for name, param_x in lstm_layers.named_parameters():
            #print(name, param_x)
            param_x.data = parameters[name]

        data_reshaped = data.transpose(0, 1)  # shape needs to be (sequence_length, num_ways, input_size)

        # Pass data through the LSTM layer
        lstm_output, _ = lstm_layers(data_reshaped)
        out1 = lstm_output[-1]
        if out1.size(0) > 1:
            out1 = F.batch_norm(out1, None, None, training=True)
        out2 = F.relu(out1)
        out3 = F.linear(out1, parameters['linear_w'], parameters['linear_b'])
        out = F.softmax(out3)

        return out

    def _inner_loop(self, data, labels, train):
        """Computes the adapted network parameters via the MAML inner loop.

        Args:
            images (Tensor): task support set inputs
                shape (num_images, channels, height, width)
            labels (Tensor): task support set outputs
                shape (num_images,)
            train (bool): whether we are training or evaluating

        Returns:
            parameters (dict[str, Tensor]): adapted network parameters
            accuracies (list[float]): support set accuracy over the course of
                the inner loop, length num_inner_steps + 1
            gradients(list[float]): gradients computed from auto.grad, just needed
                for autograders, no need to use this value in your code and feel to replace
                with underscore
        """
        accuracies = []
        parameters = {
            k: torch.clone(v)
            for k, v in self._meta_parameters.items()
        }
        gradients = None
        ### START CODE HERE ###
        # TODO: finish implementing this method.
        # This method computes the inner loop (adaptation) procedure
        # over the course of _num_inner_steps steps for one
        # task. It also scores the model along the way.
        # Make sure to populate accuracies and update parameters.
        # Use F.cross_entropy to compute classification losses.
        # Use util.score to compute accuracies.
        logits = self._forward(data, parameters)
        accuracy = util.score(logits, labels)
        accuracies.append(accuracy)

        for step in range(self._num_inner_steps):
            logits = self._forward(data, parameters)
            loss = F.cross_entropy(logits, labels)

            if train:
                gradients = autograd.grad(loss, list(parameters.values()), create_graph=True, allow_unused=True)
            else:
                gradients = autograd.grad(loss, list(parameters.values()), create_graph=False, allow_unused=True)

            updated_param = {}

            for (param_name, param), grad in zip(parameters.items(), gradients):
                #print(f"Gradient for {param_name}")
                if grad is None:
                    #print(f"NO GRAD: {param_name}")
                    updated_param[param_name] = param
                    continue
                updated_param[param_name] = param - self._inner_lrs[param_name] * grad

            parameters = updated_param

            logits = self._forward(data, parameters)
            accuracy = util.score(logits, labels)
            accuracies.append(accuracy)


        ### END CODE HERE ###
        return parameters, accuracies, gradients

    def _outer_step(self, task_batch, train):
        """Computes the MAML loss and metrics on a batch of tasks.

        Args:
            task_batch (tuple): batch of tasks from an DataLoader
            train (bool): whether we are training or evaluating

        Returns:
            outer_loss (Tensor): mean MAML loss over the batch, scalar
            accuracies_support (ndarray): support set accuracy over the
                course of the inner loop, averaged over the task batch
                shape (num_inner_steps + 1,)
            accuracy_query (float): query set accuracy of the adapted
                parameters, averaged over the task batch
        """
        outer_loss_batch = []
        accuracies_support_batch = []
        accuracy_query_batch = []
        predictions = []

        for task in task_batch:

            data_support, labels_support, data_query, labels_query = task
            data_support = data_support.to(self.device)
            labels_support = labels_support.to(self.device)
            data_query = data_query.to(self.device)
            labels_query = labels_query.to(self.device)
            ### START CODE HERE ###
            # TODO: finish implementing this method.
            # For a given task, use the _inner_loop method to adapt for
            # _num_inner_steps steps, then compute the MAML loss and other
            # metrics. Reminder you can replace gradients with _ when calling
            # _inner_loop.
            # Use F.cross_entropy to compute classification losses.
            # Use util.score to compute accuracies.
            # Make sure to populate outer_loss_batch, accuracies_support_batch,
            # and accuracy_query_batch.
            # support accuracy: The first element (index 0) should be the accuracy before any steps are taken.

            parameters, accuracies, _ = self._inner_loop(data_support, labels_support, train)

            q_logits = self._forward(data_query, parameters)
            predictions.extend(torch.clone(q_logits).detach())

            q_accuracy = util.score(q_logits, labels_query)
            maml_loss = F.cross_entropy(q_logits, labels_query)

            outer_loss_batch.append(maml_loss)
            accuracy_query_batch.append(q_accuracy)
            accuracies_support_batch.append(accuracies)

            ### END CODE HERE ###
        outer_loss = torch.mean(torch.stack(outer_loss_batch))
        accuracies_support = np.mean(
            accuracies_support_batch,
            axis=0
        )
        accuracy_query = np.mean(accuracy_query_batch)

        return outer_loss, accuracies_support, accuracy_query, predictions

    def train(self, dataloader_meta_train, dataloader_meta_val, writer):
        """Train the MAML.

        Consumes dataloader_meta_train to optimize MAML meta-parameters
        while periodically validating on dataloader_meta_val, logging metrics, and
        saving checkpoints.

        Args:
            dataloader_meta_train (DataLoader): loader for train tasks
            dataloader_meta_val (DataLoader): loader for validation tasks
            writer (SummaryWriter): TensorBoard logger
        """
        print(f'Starting training at iteration {self._start_train_step}.')
        for i_step, task_batch_wrong in enumerate(
                dataloader_meta_train,
                start=self._start_train_step
        ):
            data_support_list, labels_support_list, data_query_list, labels_query_list = task_batch_wrong

            # Combine the individual tasks into the correct format
            task_batch = []
            for i in range(args.batch_size):  #
                data_support = data_support_list[i]
                labels_support = labels_support_list[i]
                data_query = data_query_list[i]
                labels_query = labels_query_list[i]
                task_batch.append((data_support, labels_support, data_query, labels_query))

            self._optimizer.zero_grad()
            outer_loss, accuracies_support, accuracy_query, _ = (
                self._outer_step(task_batch, train=True)
            )
            outer_loss.backward()
            self._optimizer.step()

            if i_step % LOG_INTERVAL == 0:
                print(
                    f'Iteration {i_step}: '
                    f'loss: {outer_loss.item():.3f}, '
                    f'pre-adaptation support accuracy: '
                    f'{accuracies_support[0]:.3f}, '
                    f'post-adaptation support accuracy: '
                    f'{accuracies_support[-1]:.3f}, '
                    f'post-adaptation query accuracy: '
                    f'{accuracy_query:.3f}'
                )
                writer.add_scalar('loss/train', outer_loss.item(), i_step)
                writer.add_scalar(
                    'train_accuracy/pre_adapt_support',
                    accuracies_support[0],
                    i_step
                )
                writer.add_scalar(
                    'train_accuracy/post_adapt_support',
                    accuracies_support[-1],
                    i_step
                )
                writer.add_scalar(
                    'train_accuracy/post_adapt_query',
                    accuracy_query,
                    i_step
                )

            if i_step % VAL_INTERVAL == 0:
                losses = []
                accuracies_pre_adapt_support = []
                accuracies_post_adapt_support = []
                accuracies_post_adapt_query = []
                for val_task_batch_wrong in dataloader_meta_val:
                    data_support_list, labels_support_list, data_query_list, labels_query_list = val_task_batch_wrong

                    # Combine the individual tasks into the correct format
                    val_task_batch = []
                    for i in range(args.batch_size):  #
                        data_support = data_support_list[i]
                        labels_support = labels_support_list[i]
                        data_query = data_query_list[i]
                        labels_query = labels_query_list[i]
                        val_task_batch.append((data_support, labels_support, data_query, labels_query))

                    outer_loss, accuracies_support, accuracy_query, _ = (
                        self._outer_step(val_task_batch, train=False)
                    )
                    losses.append(outer_loss.item())
                    accuracies_pre_adapt_support.append(accuracies_support[0])
                    accuracies_post_adapt_support.append(accuracies_support[-1])
                    accuracies_post_adapt_query.append(accuracy_query)
                loss = np.mean(losses)
                accuracy_pre_adapt_support = np.mean(
                    accuracies_pre_adapt_support
                )
                accuracy_post_adapt_support = np.mean(
                    accuracies_post_adapt_support
                )
                accuracy_post_adapt_query = np.mean(
                    accuracies_post_adapt_query
                )
                print(
                    f'Validation: '
                    f'loss: {loss:.3f}, '
                    f'pre-adaptation support accuracy: '
                    f'{accuracy_pre_adapt_support:.3f}, '
                    f'post-adaptation support accuracy: '
                    f'{accuracy_post_adapt_support:.3f}, '
                    f'post-adaptation query accuracy: '
                    f'{accuracy_post_adapt_query:.3f}'
                )
                writer.add_scalar('loss/val', loss, i_step)
                writer.add_scalar(
                    'val_accuracy/pre_adapt_support',
                    accuracy_pre_adapt_support,
                    i_step
                )
                writer.add_scalar(
                    'val_accuracy/post_adapt_support',
                    accuracy_post_adapt_support,
                    i_step
                )
                writer.add_scalar(
                    'val_accuracy/post_adapt_query',
                    accuracy_post_adapt_query,
                    i_step
                )

            if i_step % SAVE_INTERVAL == 0:
                self._save(i_step)

    def test(self, dataloader_test):
        """Evaluate the MAML on test tasks.

        Args:
            dataloader_test (DataLoader): loader for test tasks
        """
        accuracies = []
        predictions = []
        _, _, accuracy_query, preds_test = self._outer_step(dataloader_test, train=False)
        predictions.extend(preds_test)
        accuracies.append(accuracy_query)
        mean = np.mean(accuracies)
        std = np.std(accuracies)
        mean_95_confidence_interval = 1.96 * std / np.sqrt(NUM_TEST_TASKS)
        print(
            f'Accuracy over {NUM_TEST_TASKS} test tasks: '
            f'mean {mean:.3f}, '
            f'95% confidence interval {mean_95_confidence_interval:.3f}'
        )
        predicted_classes = [torch.argmax(tensor).item() for tensor in predictions]
        start = "2011-01-01"
        end = "2023-06-30"
        ticker_dict, tickerTest = util.get_categorical_tickers()
        tickerList = ticker_dict['all']
        tickerList = [item for sublist in tickerList for item in sublist]
        prices_raw = util.get_ticker_values(tickerList, start, end)
        # print("prices raw shape", prices_raw.shape)
        prices_test = prices_raw[-1].copy()
        _, ideal = util.data_preprocess_HL_labels(prices_test[-250-CONTEXT_DAYS:], 0, None, CONTEXT_DAYS)

        print("PREDICTED", len(predicted_classes), predicted_classes)

        print("CLASS REP FIRST")
        #print(classification_report(labels[:50], predicted_classes[:50]))
        print("lens", len(ideal), len(predicted_classes))
        print(classification_report(ideal[:50], predicted_classes[:50]))

        util.plot_bot_decision(prices_test[-250:][:50], ideal[:50], predicted_classes[:50])
        #util.plot_bot_decision(prices_test[-250:], ideal, predicted_classes)


    def load(self, checkpoint_step):
        """Loads a checkpoint.

        Args:
            checkpoint_step (int): iteration of checkpoint to load

        Raises:
            ValueError: if checkpoint for checkpoint_step is not found
        """
        target_path = (
            f'{os.path.join(self._log_dir, "state")}'
            f'{checkpoint_step}.pt'
        )
        if os.path.isfile(target_path):
            state = torch.load(target_path)
            self._meta_parameters = state['meta_parameters']
            self._inner_lrs = state['inner_lrs']
            self._optimizer.load_state_dict(state['optimizer_state_dict'])
            self._start_train_step = checkpoint_step + 1
            print(f'Loaded checkpoint iteration {checkpoint_step}.')
        else:
            raise ValueError(
                f'No checkpoint for iteration {checkpoint_step} found.'
            )

    def _save(self, checkpoint_step):
        """Saves parameters and optimizer state_dict as a checkpoint.

        Args:
            checkpoint_step (int): iteration to label checkpoint with
        """
        optimizer_state_dict = self._optimizer.state_dict()
        torch.save(
            dict(meta_parameters=self._meta_parameters,
                 inner_lrs=self._inner_lrs,
                 optimizer_state_dict=optimizer_state_dict),
            f'{os.path.join(self._log_dir, "state")}{checkpoint_step}.pt'
        )
        print('Saved checkpoint.')


def main(args):

    print(args)

    if args.device == "gpu" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("true 1")
        # on MPS the derivative for aten::linear_backward is not implemented ... Waiting for PyTorch 2.1.0
        # DEVICE = "mps"

        # Due to the above, default for now to cpu
        DEVICE = "cpu"
    elif args.device == "gpu" and torch.cuda.is_available():
        print("true 2")
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    print("Using device: ", DEVICE)

    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f'./logs/maml-lstm/stocks.way_{args.num_way}.support_{args.num_support}.query_{args.num_query}.inner_steps_{args.num_inner_steps}.inner_lr_{args.inner_lr}.learn_inner_lrs_{args.learn_inner_lrs}.outer_lr_{args.outer_lr}.batch_size_{args.batch_size}'  # pylint: disable=line-too-long
    print(f'log_dir: {log_dir}')
    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    maml = MAMLLSTM(
        args.num_way,
        args.num_inner_steps,
        args.inner_lr,
        args.learn_inner_lrs,
        args.outer_lr,
        log_dir,
        DEVICE
    )

    if args.checkpoint_step > -1:
        maml.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')

    ticker_dict, tickerTest = util.get_categorical_tickers()
    tickerList = ticker_dict['all']
    tickerList = [item for sublist in tickerList for item in sublist]

    start = "2011-01-01"
    end = "2023-06-30"

    if not args.test:
        num_training_tasks = args.batch_size * (args.num_train_iterations -
                                                args.checkpoint_step - 1)
        print(
            f'Training on {num_training_tasks} tasks with composition: '
            f'num_way={args.num_way}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        print("STARTING DOWNLOAD")
        dataloader_meta_train = stocks.get_stock_market_dataloader(
            args.batch_size,
            tickerList,
            'train',
            start,
            end,
            args.num_way,
            args.num_support,
            args.num_query,
            num_training_tasks,
            args.num_workers
        )
        print("FINISH DOWNLOAD TRAIN")
        print("DATALOADER TRAIN")

        dataloader_meta_val = stocks.get_stock_market_dataloader(
            args.batch_size,
            tickerList,
            'val',
            start,
            end,
            args.num_way,
            args.num_support,
            args.num_query,
            args.batch_size * 4,
            args.num_workers
        )
        print("FINISH DOWNLOAD VAL")

        maml.train(
            dataloader_meta_train,
            dataloader_meta_val,
            writer
        )
    else:
        print(
            f'Testing on tasks with composition '
            f'num_way={args.num_way}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        print("START DOWNLOAD TEST")

        data = util.get_ticker_values(tickerList, start, end)
        ticker_data = data[-1].copy()

        print("GETING ITEMS")
        data_support, data_query = [], []
        labels_support, labels_query = [], []
        no_days = 251

        print("DATA shape", np.array(ticker_data).shape)
        ticker_data = ticker_data[-no_days-CONTEXT_DAYS-3:]
        print("ticker", np.array(ticker_data).shape)
        data, labels = util.data_preprocess_HL_labels(ticker_data, 0, None, CONTEXT_DAYS)
        print("DATA+LABELS", np.array(data).shape, np.array(labels).shape)
        context_data, context_labels = util.data_w_context(data, labels)
        print("CONTEXT DATA+LABELS", np.array(context_data).shape, np.array(context_labels).shape)

        for i in range(3, len(context_labels)):
            data_query.extend([context_data[i]])
            labels_query.append(context_labels[i])

            data_support.extend(
                [context_data[i - 3], context_data[i - 2], context_data[i - 1]])  # most recent
            labels_support.extend([context_labels[i - 3], context_labels[i - 2], context_labels[i - 1]])

        data_support = torch.stack(data_support)
        labels_support = torch.tensor(labels_support)
        data_query = torch.stack(data_query)
        labels_query = torch.tensor(labels_query)
        print("FINAL DATA+LABELS", data_support.shape, len(labels_support), data_query.shape)
        print("LEN TEST", len(data_query))

        dataloader_test = [[data_support, labels_support, data_query, labels_query]]
        print("FINISH Loading TEST")
        maml.test(dataloader_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a MAML!')
    parser.add_argument('--log_dir', type=str, default='logs/maml-lstm/stocks.way_3.support_1.query_1.inner_steps_1.inner_lr_0.4.learn_inner_lrs_True.outer_lr_1e-05.batch_size_16',
                        help='directory to save to or load from')
    parser.add_argument('--num_way', type=int, default=3,
                        help='number of classes in a task')
    parser.add_argument('--num_support', type=int, default=1,
                        help='number of support examples per class in a task')
    parser.add_argument('--num_query', type=int, default=1,
                        help='number of query examples per class in a task')
    parser.add_argument('--num_inner_steps', type=int, default=1,
                        help='number of inner-loop updates')
    parser.add_argument('--inner_lr', type=float, default=0.4,
                        help='inner-loop learning rate initialization')
    parser.add_argument('--learn_inner_lrs', default=True, action='store_true',
                        help='whether to optimize inner-loop learning rates')
    parser.add_argument('--outer_lr', type=float, default=0.00001,
                        help='outer-loop learning rate')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='number of tasks per outer-loop update')
    parser.add_argument('--num_train_iterations', type=int, default=1001,
                        help='number of outer-loop updates to train for MAKE IT > CHECKPOINT FOR FINE')
    parser.add_argument('--test', default=True, action='store_true',
                        help='train or test')
    parser.add_argument('--checkpoint_step', type=int, default=600,
                        help=('checkpoint iteration to load for resuming '
                              'training, or for evaluation (-1 is ignored)'))
    parser.add_argument('--num_workers', type=int, default=8,
                        help=('needed to specify dataloader'))
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--device', type=str, default='gpu')

    args = parser.parse_args()
    main(args)