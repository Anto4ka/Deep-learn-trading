import random
from collections import deque
from typing import List
import numpy as np
import stock_exchange
from experts.obscure_expert import ObscureExpert
from framework.vote import Vote
from framework.period import Period
from framework.portfolio import Portfolio
from framework.stock_market_data import StockMarketData
from framework.interface_expert import IExpert
from framework.interface_trader import ITrader
from framework.order import Order, OrderType
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from framework.company import Company
# from framework.order import Cpycompany
from framework.utils import save_keras_sequential, load_keras_sequential
from framework.logger import logger
import os
import random


class State:
    def __init__(self, last_stock_data_a, last_stock_data_b, vote_a, vote_b):
        self.last_stock_data_a = last_stock_data_a
        self.last_stock_data_b = last_stock_data_b

        def convertVote(vote):
            if vote == Vote.BUY:
                return 3
            elif vote == Vote.HOLD:
                return 2
            elif vote == Vote.SELL:
                return 1
            else:
                exit("Wrong vote")

        self.vote_a = convertVote(vote_a)
        self.vote_b = convertVote(vote_b)

        def getDiff(vals):

            if (len(vals) != 2):
                return 0.0
            else:

                val1 = vals[0][-1]
                val2 = vals[1][-1]
                ratio = val2 / val1

                if (ratio) > 1.0:
                    return 3
                elif (ratio == 1.0):
                    return 2
                else:
                    return 1

        self.aDiff = getDiff(last_stock_data_a)
        self.bDiff = getDiff(last_stock_data_b)


class DeepQLearningTrader(ITrader):
    """
    Implementation of ITrader based on Deep Q-Learning (DQL).
    """
    RELATIVE_DATA_DIRECTORY = 'traders/dql_trader_data'

    def __init__(self, expert_a: IExpert, expert_b: IExpert, load_trained_model: bool = True,
                 train_while_trading: bool = False, color: str = 'black', name: str = 'dql_trader', ):
        """
        Constructor
        Args:
            expert_a: Expert for stock A
            expert_b: Expert for stock B
            load_trained_model: Flag to trigger loading an already trained neural network
            train_while_trading: Flag to trigger on-the-fly training while trading
        """

        # Save experts, training mode and name
        super().__init__(color, name)
        assert expert_a is not None and expert_b is not None
        self.expert_a = expert_a
        self.expert_b = expert_b
        self.train_while_trading = train_while_trading

        # Parameters for neural network
        self.state_size = 4
        self.action_size = 5
        self.hidden_size = 50

        # Parameters for deep Q-learning
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.min_size_of_memory_before_training = 1000  # should be way bigger than batch_size, but smaller than memory
        self.memory = deque(maxlen=2000)

        # Attributes necessary to remember our last actions and fill our memory with experiences
        self.last_state = None
        self.last_action_a = None
        self.last_action_b = None
        self.last_portfolio_value = None
        self.last_order = None
        self.last_input = None

        # Create main model, either as trained model (from file) or as untrained model (from scratch)
        self.model = None
        if load_trained_model:
            self.model = load_keras_sequential(self.RELATIVE_DATA_DIRECTORY, self.get_name())
            logger.info(f"DQL Trader: Loaded trained model")
        if self.model is None:  # loading failed or we didn't want to use a trained model
            self.model = Sequential()
            self.model.add(Dense(self.hidden_size * 2, input_dim=self.state_size, activation='relu'))
            self.model.add(Dense(self.hidden_size, activation='relu'))
            self.model.add(Dense(self.action_size, activation='linear'))
            logger.info(f"DQL Trader: Created new untrained model")
        assert self.model is not None
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

    def save_trained_model(self):
        """
        Save the trained neural network under a fixed name specific for this traders.
        """
        save_keras_sequential(self.model, self.RELATIVE_DATA_DIRECTORY, self.get_name())
        logger.info(f"DQL Trader: Saved trained model")

    def trade(self, portfolio: Portfolio, stock_market_data: StockMarketData) -> List[Order]:

        def pVal(portfolio, stock_data_a, stock_data_b):
            a_val = portfolio.get_stock(Company.A) * stock_data_a.get_last()[-1]
            b_val = portfolio.get_stock(Company.B) * stock_data_b.get_last()[-1]

            buf = portfolio.cash + a_val + b_val

            return np.float(portfolio.cash + a_val + b_val)

        """
        Generate action to be taken on the "stock marketf"

        Args:
          portfolio : current Portfolio of this traders
          stock_market_data : StockMarketData for evaluation

        Returns:
          A OrderList instance, may be empty never None
        """
        assert portfolio is not None
        assert stock_market_data is not None
        assert stock_market_data.get_companies() == [Company.A, Company.B]

        # TODO Compute the current state

        stock_data_a = None
        stock_data_b = None
        last_stock_data_a = None
        last_stock_data_b = None

        company_list = stock_market_data.get_companies()
        for company in company_list:
            if company == Company.A:
                stock_data_a = stock_market_data[Company.A]
                last_stock_data_a = stock_data_a.get_from_offset(-2)
            elif company == Company.B:
                stock_data_b = stock_market_data[Company.B]
                last_stock_data_b = stock_data_b.get_from_offset(-2)
            else:
                assert False

        vote_a = self.expert_a.vote(stock_data_a)
        vote_b = self.expert_b.vote(stock_data_b)

        state = State(last_stock_data_a, last_stock_data_b, vote_a, vote_b)

        # TODO Q-Learning
        nn_input = np.array([np.array([state.aDiff, state.vote_a,
                                       state.bDiff, state.vote_b])])

        action_vals = self.model.predict(nn_input)

        # TODO Store state as experience (memory) and train the neural network only if trade() was called before at least once

        # TODO Create actions for current state and decrease epsilon for fewer random actions
        actions = [[Order(OrderType.BUY, Company.A, int((portfolio.cash / 2) // stock_data_a.get_last()[-1])),
                    Order(OrderType.BUY, Company.B, int((portfolio.cash / 2) // stock_data_b.get_last()[-1]))],
                   [Order(OrderType.BUY, Company.A, int((portfolio.cash) // stock_data_a.get_last()[-1])),
                    Order(OrderType.SELL, Company.B, portfolio.get_stock(Company.B))],
                   [Order(OrderType.SELL, Company.A, portfolio.get_stock(Company.A)),
                    Order(OrderType.BUY, Company.B, int(portfolio.cash // stock_data_b.get_last()[-1]))],
                   [Order(OrderType.SELL, Company.A, portfolio.get_stock(Company.A)),
                    Order(OrderType.SELL, Company.B, portfolio.get_stock(Company.B))],
                   [Order(OrderType.SELL, Company.A, 0), Order(OrderType.SELL, Company.B, 0)]]

        if not self.train_while_trading:
            self.epsilon = 0.0
        else:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            else:
                self.epsilon = self.epsilon_min

        # randomize action
        if random.random() < self.epsilon:
            next_action = random.choice(list(range(self.action_size)))
        else:
            next_action = np.argmax(action_vals[0])

        order_list = actions[next_action]
        # portfolio_value = pVal(portfolio, stock_data_a, stock_data_b)
        portfolio_value = portfolio.get_value(stock_market_data, stock_market_data.get_most_recent_trade_day())

        if (self.last_state != None and self.train_while_trading):

            def reward(oldVal, newVal):
                neg = -100.0
                pos = 100.0

                qTrash = 1.000

                q = newVal / oldVal

                if q < 1:
                    return neg
                elif q == 1:
                    return -10
                else:
                    print("Q: ", q)
                    if q > qTrash:
                        return pos * oldVal / newVal
                    else:
                        return 50 *  oldVal / newVal

            r = reward(self.last_portfolio_value, portfolio_value)
            # r = portfolio_value - self.last_portfolio_value
            # r = portfolio_value

            action_vals[0][self.last_order] = r

            self.memory.append([self.last_input, action_vals])

            if (len(self.memory) > self.min_size_of_memory_before_training):
                sample = random.sample(self.memory, self.batch_size)
                trainSample = list()
                testSample = list()

                for [sampleIn, sampleOut] in sample:
                    trainSample.append(sampleIn[0])
                    testSample.append(sampleOut[0])

                self.model.fit(np.array(trainSample), np.array(testSample), self.batch_size)

        # Save created state, actions and portfolio value for the next call of trade()

        self.last_input = nn_input
        self.last_state = state
        self.last_order = next_action
        self.last_portfolio_value = portfolio_value

        print(next_action, action_vals, portfolio.cash, portfolio.get_stock(Company.A), portfolio.get_stock(Company.B))
        return order_list


# This method retrains the traders from scratch using training data from TRAINING and test data from TESTING
EPISODES = 5
if __name__ == "__main__":
    # Create the training data and testing data
    # Hint: You can crop the training data with training_data.deepcopy_first_n_items(n)
    training_data = StockMarketData([Company.A, Company.B], [Period.TRAINING])
    testing_data = StockMarketData([Company.A, Company.B], [Period.TESTING])

    # Create the stock exchange and one traders to train the net
    stock_exchange = stock_exchange.StockExchange(10000.0)
    training_trader = DeepQLearningTrader(ObscureExpert(Company.A), ObscureExpert(Company.B), False, True)

    # Save the final portfolio values per episode
    final_values_training, final_values_test = [], []

    for i in range(EPISODES):
        logger.info(f"DQL Trader: Starting training episode {i}")

        # train the net
        stock_exchange.run(training_data, [training_trader])
        training_trader.save_trained_model()
        final_values_training.append(stock_exchange.get_final_portfolio_value(training_trader))

        # test the trained net
        testing_trader = DeepQLearningTrader(ObscureExpert(Company.A), ObscureExpert(Company.B), True, False)
        stock_exchange.run(testing_data, [testing_trader])
        final_values_test.append(stock_exchange.get_final_portfolio_value(testing_trader))

        logger.info(f"DQL Trader: Finished training episode {i}, "
                    f"final portfolio value training {final_values_training[-1]} vs. "
                    f"final portfolio value test {final_values_test[-1]}")

    from matplotlib import pyplot as plt

    plt.figure()
    plt.plot(final_values_training, label='training', color="black")
    plt.plot(final_values_test, label='test', color="green")
    plt.title('final portfolio value training vs. final portfolio value test')
    plt.ylabel('final portfolio value')
    plt.xlabel('episode')
    plt.legend(['training', 'test'])
    plt.show()
