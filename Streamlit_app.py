import optuna
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import streamlit as st

# %matplotlib inline
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent

from finrl import config_tickers
from finrl.config import INDICATORS
import warnings
import itertools
import pandas as pd

warnings.filterwarnings('ignore')

##############################################################################################################################################

## %% Data download

if datetime.now().month >= 10:
    End_date = str(datetime.now().year) + '-' + str(datetime.now().month) + '-' + str(datetime.now().day)
else:
    End_date = str(datetime.now().year) + '-0' + str(datetime.now().month) + '-' + str(datetime.now().day)

Ticker_list = list(config_tickers.DOW_30_TICKER)

TRAIN_START = st.text_input("TRAIN_START", "2005-01-01")
TRAIN_END = st.text_input("TRAIN_END", "2020-12-31")
TRADE_START = st.text_input("TRADE_START", "2021-01-01")
TRADE_END = st.text_input("TRADE_END", End_date)
load_data = st.checkbox("LOAD DATA")


@st.cache_data
def download_data(TRAIN_START_DATE: str, TRADE_END_DATE: str, Tickers: list):
    df = YahooDownloader(start_date=TRAIN_START_DATE,
                         end_date=TRADE_END_DATE,
                         ticker_list=Ticker_list).fetch_data()

    print('\nData loaded!')

    return df


###################################################################################################################################################

## %% Preporcessing
@st.cache_data
def preprocess_data(df):
    fe = FeatureEngineer(use_technical_indicator=True,
                         tech_indicator_list=INDICATORS,
                         use_vix=True,
                         use_turbulence=True,
                         user_defined_feature=False)

    df_process = fe.preprocess_data(df)

    list_ticker = df_process["tic"].unique().tolist()
    list_date = list(pd.date_range(df_process['date'].min(), df_process['date'].max()).astype(str))
    combination = list(itertools.product(list_date, list_ticker))

    df_process = pd.DataFrame(combination, columns=["date", "tic"]).merge(df_process, on=["date", "tic"], how="left")
    df_process = df_process[df_process['date'].isin(df_process['date'])]
    df_process = df_process.sort_values(['date', 'tic'])

    df_process = df_process.ffill()
    print('Preprocessing done!')

    return df_process


df_process = None

if load_data and all([TRAIN_START, TRAIN_END, TRADE_START, TRADE_END]):
    # Call the download_data and process_data functions here
    # Display first 5 samples of data in a DataFrame
    df = download_data(TRAIN_START_DATE=TRAIN_START, TRADE_END_DATE=TRADE_END, Tickers=Ticker_list)
    df_process = preprocess_data(df=df)
    Ticker_list = list(df_process['tic'].unique())
    st.dataframe(df.head(5), use_container_width=True, hide_index=True)
#################################################################################################################################################

# Create subplots with 2 rows; top for candlestick price, and bottom for bar volume
Use_ticker = st.selectbox("Select Ticker", Ticker_list)
Plot_data = st.button("PLOT DATA")


@st.cache_data
def choose_ticker(load_data, Use_ticker, df_process):
    df_unique = None
    if load_data is True and Use_ticker is not None:
        # Use_ticker = random.choice(Ticker_list)
        df_unique = df_process[df_process['tic'] == Use_ticker]
    return df_unique


df_unique = None
if load_data is True and Use_ticker is not None:
    df_unique = choose_ticker(load_data, Use_ticker, df_process)


if st.button('PLOT DATA') and load_data is True and Use_ticker is not None:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=(Use_ticker, 'Volume'),
                        vertical_spacing=0.1,
                        row_width=[0.2, 0.7])

    # ----------------
    # Candlestick Plot
    fig = fig.add_trace(go.Candlestick(x=df_unique['date'],
                                 open=df_unique['open'],
                                 high=df_unique['high'],
                                 low=df_unique['low'],
                                 close=df_unique['close'], showlegend=False,
                                 name='candlestick'),
                  row=1, col=1)

    # Moving Average
    fig = fig.add_trace(go.Scatter(x=df_unique['date'],
                             y=df_unique['close_30_sma'],
                             line_color='black',
                             name='sma'),
                  row=1, col=1)

    # RSI
    fig = fig.add_trace(go.Scatter(x=df_unique['date'],
                             y=df_unique['rsi_30'],
                             line_color='green',
                             name='rsi'),
                  row=1, col=1)

    # MACD
    fig = fig.add_trace(go.Scatter(x=df_unique['date'],
                             y=df_unique['macd'],
                             line_color='blue',
                             name='macd'),
                  row=1, col=1)
    # Vix
    fig = fig.add_trace(go.Scatter(x=df_unique['date'],
                             y=df_unique['vix'],
                             line_color='red',
                             name='vix'),
                  row=1, col=1)

    # Upper Bound
    fig = fig.add_trace(go.Scatter(x=df_unique['date'],
                             y=df_unique['boll_ub'],
                             line_color='gray',
                             name='upper band',
                             opacity=0.5),
                  row=1, col=1)

    # Lower Bound fill in between with parameter 'fill': 'tonexty'
    fig = fig.add_trace(go.Scatter(x=df_unique['date'],
                             y=df_unique['boll_lb'],
                             line_color='gray',
                             fill='tonextx',
                             name='lower band',
                             opacity=0.2),
                  row=1, col=1)

    # ----------------
    # Volume Plot
    fig = fig.add_trace(go.Bar(x=df_unique['date'], y=df_unique['volume'], showlegend=True),
                  row=2, col=1)

    # Remove range slider; (short time frame)
    fig = fig.update(layout_xaxis_rangeslider_visible=False)

    fig = fig.update_layout(height=900, width=1100)

    #fig.show()
    st.plotly_chart(fig,use_container_width=True)
else:
    pass


####################################################################################################################################################

@st.cache_data
def dataset_splitting(df_unique, TRAIN_START, TRAIN_END, TRADE_START, TRADE_END):
    train = data_split(df_unique, TRAIN_START, TRAIN_END)
    trade = data_split(df_unique, TRADE_START, TRADE_END)

    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }
    e_train_gym = StockTradingEnv(df=train, **env_kwargs)
    e_trade_gym = StockTradingEnv(df=trade, turbulence_threshold=None, **env_kwargs)

    env_train, _ = e_train_gym.get_sb_env()

    print('Environment Created!')

    return env_train, e_trade_gym


env_train = None
e_trade_gym = None
if df_unique is not None:
    env_train, e_trade_gym = dataset_splitting(df_unique, TRAIN_START, TRAIN_END, TRADE_START, TRADE_END)
    agent = DRLAgent(env = env_train)

# Add a dropdown for selecting the action (Train agents, Fine Tune agents)
action = st.selectbox("Select Action", ["Train Agent", "FineTune Agent"])
selected_agent = None


if action == "Train Agent" or 'FineTune Agent':
    # Add a dropdown for selecting the agent
    selected_agent = st.selectbox("Select Agent", ["A2C", "DDPG", "PPO", "TD3", "SAC"])


@st.cache_resource
def train_agent(df_unique, selected_agent, timesteps):
    # tmp_path = RESULTS_DIR + f'/{model_name}'
    env_train, e_trade_gym = dataset_splitting(df_unique, TRAIN_START, TRAIN_END, TRADE_START, TRADE_END)
    agent = DRLAgent(env=env_train)
    model = agent.get_model(selected_agent.lower())

    # Set new logger
    # new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # model.set_logger(new_logger)

    # trained_agent = None

    with st.spinner("Training has started... Please wait!"):
        trained_agent = agent.train_model(model=model,
                                          tb_log_name=selected_agent.lower(),
                                          total_timesteps=timesteps)

    # trained_agent.save(TRAINED_MODEL_DIR + f'/agent_{model_name}.pth')
    print('{} training done!'.format(selected_agent.lower()))
    return trained_agent


def sample_params(trial: optuna.Trial, algorithm: str):
    params = {}

    buffer_size = trial.suggest_categorical("buffer_size", [int(1e2), int(1e3), int(1e4)])
    learning_rate = trial.suggest_categorical('learning_rate',[1e-5,1e-4,1e-3])
    batch_size = trial.suggest_categorical("batch_size", [ 64, 256, 512])

    if algorithm == "a2c":
        normalize_advantage = trial.suggest_categorical("normalize_advantage", [False, True])
        params.update({"learning_rate": learning_rate, "normalize_advantage": normalize_advantage})
    elif algorithm == "ddpg":
        params.update({"buffer_size": buffer_size, "learning_rate": learning_rate, "batch_size": batch_size})
    elif algorithm == "ppo":
        params.update({"learning_rate": learning_rate, "batch_size": batch_size})
    elif algorithm in ["td3", "sac"]:
        params.update({"buffer_size": buffer_size, "learning_rate": learning_rate, "batch_size": batch_size})

    return params


def calculate_sharpe(df):
  df['daily_return'] = df['account_value'].pct_change(1)
  if df['daily_return'].std() !=0:
    sharpe = (252**0.5)*df['daily_return'].mean()/ \
          df['daily_return'].std()
    return sharpe
  else:
    return 0

class LoggingCallback:
    def __init__(self,threshold,trial_number,patience):

      self.threshold = threshold
      self.trial_number  = trial_number
      self.patience = patience
      self.cb_list = [] #Trials list for which threshold is reached
    def __call__(self,study:optuna.study, frozen_trial:optuna.Trial):
      #Setting the best value in the current trial
      study.set_user_attr("previous_best_value", study.best_value)

      #Checking if the minimum number of trials have pass
      if frozen_trial.number >self.trial_number:
          previous_best_value = study.user_attrs.get("previous_best_value",None)
          #Checking if the previous and current objective values have the same sign
          if previous_best_value * study.best_value >=0:
              #Checking for the threshold condition
              if abs(previous_best_value-study.best_value) < self.threshold:
                  self.cb_list.append(frozen_trial.number)
                  #If threshold is achieved for the patience amount of time
                  if len(self.cb_list)>self.patience:
                      print('The study stops now...')
                      print('With number',frozen_trial.number ,'and value ',frozen_trial.value)
                      print('The previous and current best values are {} and {} respectively'
                              .format(previous_best_value, study.best_value))
                      study.stop()


def objective(_trial: optuna.Trial, algorithm: str):

    # Define hyperparameters based on the algorithm
    if algorithm == "a2c":
        hyperparameters = sample_params(_trial,algorithm)
    elif algorithm == "ddpg":
        hyperparameters = sample_params(_trial,algorithm)
    elif algorithm == "ppo":
        hyperparameters = sample_params(_trial,algorithm)
    elif algorithm == "td3":
        hyperparameters = sample_params(_trial,algorithm)
    elif algorithm == "sac":
        hyperparameters = sample_params(_trial,algorithm)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    model = agent.get_model(algorithm, model_kwargs=hyperparameters)

    # You can increase it for better comparison
    trained_model = agent.train_model(model=model, tb_log_name=algorithm, total_timesteps=5000)

    #model_save_path = f'trained_models/{algorithm}_{trial.number}.pth'
    #trained_model.save(model_save_path)

    # For the given hyperparameters, determine the account value in the trading period
    df_account_value, df_actions = DRLAgent.DRL_prediction(model=trained_model, environment=e_trade_gym)

    # Calculate Sharpe ratio from the account value
    sharpe = calculate_sharpe(df_account_value)

    return sharpe

@st.cache_data
def run_optimization(algorithm: str, study_name: str, n_trials: int):
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(study_name=study_name, direction='maximize', sampler=sampler, pruner=optuna.pruners.HyperbandPruner())
    # if study.best_value is not None:
    #     st.write(f"Best Trial Number: {study.best_trial.number}")
    #     st.write(f"Best Sharpe Ratio: {study.best_value}")
    logging_callback = LoggingCallback(threshold=1e-3, patience=5, trial_number=5)

    study.optimize(lambda trial: objective(trial, algorithm), n_trials=n_trials, catch=(ValueError,), callbacks=[logging_callback])
    #Get the best hyperparamters
    print('Hyperparameters after tuning',study.best_params)

    return study


timesteps = None
trained_agent = None
if selected_agent is not None:
    # Add a scrollbar to select the timesteps
    timesteps = st.slider("Select timestep", min_value=1000, max_value=100000, step=2000)
df_account = None
if st.button("Train Agent") and action == 'Train Agent':
    trained_agent = train_agent(df_unique, selected_agent.lower(), timesteps)
    df_account, _ = DRLAgent.DRL_prediction(model=trained_agent, environment=e_trade_gym)
    print(df_account)

elif st.button('FineTune Agent') and action == 'FineTune Agent':

    a2c_params = ['learning_rate','normalize_advantage']
    ppo_params = ['learning_rate','batch_size']
    others_params = ['buffer_size','learning_rate','batch_size']

    model = None
    num_trials = 20
    study_name = f"{selected_agent.lower()}_study"
    with st.spinner("Fine Tuning has started... Please wait!"):
        study = run_optimization(selected_agent.lower(), study_name, 20)


    if study is not None:
        agent = DRLAgent(env=env_train)
        if selected_agent == 'A2C':
            model = agent.get_model('a2c', model_kwargs={a2c_params[0]: study.best_params['learning_rate'],
                                                         a2c_params[1]: study.best_params['normalize_advantage']})
        elif selected_agent == 'PPO':
            model = agent.get_model('ppo', model_kwargs={ppo_params[0]: study.best_params['learning_rate'],
                                                         ppo_params[1]: study.best_params['batch_size']})
        else:
            model = agent.get_model(selected_agent.lower(),
                                    model_kwargs={others_params[1]: study.best_params['learning_rate'],
                                                  others_params[0]: study.best_params['buffer_size'],
                                                  others_params[2]: study.best_params['batch_size']})

        with st.spinner("Training has started (with tuned parameters)... Please wait!"):
            tuned_agent = agent.train_model(model=model,
                                            tb_log_name=selected_agent.lower(),
                                            total_timesteps=timesteps)
        df_account, _ = DRLAgent.DRL_prediction(model=tuned_agent, environment=e_trade_gym)
        print(df_account)
if st.button('Clear All'):
    st.cache_data.clear()
    st.cache_resource.clear()


# df_account = df_account
# df1 = deepcopy(df_account)
# print(df1)
# if df_account is not None:
#     test_returns = get_daily_return(df1, value_col_name='account_value')
#     test_returns = pd.DataFrame(test_returns)
#     test_returns['date'] = test_returns.index
#     test_returns = test_returns.reset_index(drop=True)
#     test_returns.index = pd.to_datetime(test_returns['date'])
#     test_returns['Close'] = test_returns['daily_return']
#     test_returns.drop('daily_return', axis=1, inplace=True)
#     test_returns.index = test_returns.index.tz_convert('utc')
#     test_returns.drop('date', axis=1, inplace=True)
#     ret1 = pd.Series(test_returns['Close'], dtype='float32')
#     print(ret1)
#     tearsheet = pf.create_returns_tear_sheet(ret1, live_start_date='2021-01-03')
#
#     st.markdown(tearsheet)


#################################################################################################################################################



