import optuna
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import streamlit as st

# %matplotlib inline
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3 import DDPG,PPO,A2C,TD3,SAC

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
from FinGPT.finnhub_date_range import Finnhub_Date_Range
from tqdm import tqdm
from transformers import pipeline
from statistics import mode, mean
import torch.nn.functional as F

from finrl import config_tickers
from finrl.config import INDICATORS
import warnings
import itertools
import pandas as pd

from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd
warnings.filterwarnings('ignore')

##############################################################################################################################################
st.title('Financial Stock Analysis')
## %% Data download

if datetime.datetime.now().month >= 10:
    End_date = str(datetime.datetime.now().year) + '-' + str(datetime.datetime.now().month) + '-' + str(datetime.datetime.now().day)
else:
    End_date = str(datetime.datetime.now().year) + '-0' + str(datetime.datetime.now().month) + '-' + str(datetime.datetime.now().day)

Ticker_list = list(config_tickers.DOW_30_TICKER)

TRAIN_START = '2005-01-01'
TRAIN_END = End_date
TRADE_START = str(st.sidebar.date_input("TRADE_START", datetime.date(2021,1,1)))
TRADE_END = str(st.sidebar.date_input("TRADE_END", datetime.date(2023,1,1)))
st.success('Please load data!')
load_data = st.checkbox("LOAD DATA")


@st.cache_data(max_entries=1,show_spinner='Downloading stocks data ...')
def download_data(TRAIN_START_DATE: str, TRADE_END_DATE: str, Tickers: list):
    df = YahooDownloader(start_date=TRAIN_START_DATE,
                             end_date=TRADE_END_DATE,
                             ticker_list=Ticker_list).fetch_data()

    return df


###################################################################################################################################################

## %% Preporcessing
@st.cache_data(max_entries=1,show_spinner='Pre-processing data ...')
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
Use_ticker = st.sidebar.selectbox("Select Ticker", Ticker_list)
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


@st.cache_data
def plot_data(df_unique, Use_ticker):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=(Use_ticker, 'Volume'),
                        vertical_spacing=0.1,
                        row_width=[0.1, 0.8])

    # ----------------
    # Candlestick Plot
    fig.add_trace(go.Candlestick(x=df_unique['date'],
                                 open=df_unique['open'],
                                 high=df_unique['high'],
                                 low=df_unique['low'],
                                 close=df_unique['close'], showlegend=True,
                                 name='candlestick'),
                  row=1, col=1)

    # Moving Average
    fig.add_trace(go.Scatter(x=df_unique['date'],
                             y=df_unique['close_30_sma'],
                             line_color='violet',
                             name='sma'),
                  row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df_unique['date'],
                             y=df_unique['rsi_30'],
                             line_color='green',
                             name='rsi'),
                  row=1, col=1)

    # MACD
    fig.add_trace(go.Scatter(x=df_unique['date'],
                             y=df_unique['macd'],
                             line_color='blue',
                             name='macd'),
                  row=1, col=1)
    # Vix
    fig.add_trace(go.Scatter(x=df_unique['date'],
                             y=df_unique['vix'],
                             line_color='red',
                             name='vix'),
                  row=1, col=1)

    # Upper Bound
    fig.add_trace(go.Scatter(x=df_unique['date'],
                             y=df_unique['boll_ub'],
                             line_color='gray',
                             name='upper band',
                             opacity=0.5),
                  row=1, col=1)

    # Lower Bound fill in between with parameter 'fill': 'tonexty'
    fig.add_trace(go.Scatter(x=df_unique['date'],
                             y=df_unique['boll_lb'],
                             line_color='gray',
                             fill='tonextx',
                             name='lower band',
                             opacity=0.2),
                  row=1, col=1)

    # ----------------
    # Volume Plot
    fig.add_trace(go.Bar(x=df_unique['date'], y=df_unique['volume'], showlegend=True),
                  row=2, col=1)

    # Remove range slider; (short time frame)
    fig.update(layout_xaxis_rangeslider_visible=False)

    fig.update_layout(height=600, width=900)

    #fig.show()
    return fig


if df_unique is not None and Plot_data:
    fig = plot_data(df_unique, Use_ticker)
    st.plotly_chart(fig)
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
if env_train is not None:
    st.sidebar.success('Choose between Training and Fine Tuning !')
action = st.sidebar.selectbox("Select Action", ["Train Agent", "FineTune Agent"])
selected_agent = None


if action == "Train Agent" :
    # Add a dropdown for selecting the agent
    selected_agent = st.sidebar.selectbox("Select Agent", ["A2C", "DDPG", "PPO", "TD3", "SAC"])
elif action == 'FineTune Agent':
    selected_agent = st.sidebar.selectbox("Select Agent", ["A2C", "DDPG", "PPO"])


@st.cache_resource(max_entries=1,show_spinner="Training has started... Please wait!")
def train_agent(df_unique, selected_agent, timesteps):
    # tmp_path = RESULTS_DIR + f'/{model_name}'
    env_train, e_trade_gym = dataset_splitting(df_unique, TRAIN_START, TRAIN_END, TRADE_START, TRADE_END)
    agent = DRLAgent(env=env_train)
    model = agent.get_model(selected_agent.lower())

    # Set new logger
    # new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # model.set_logger(new_logger)

    # trained_agent = None

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

@st.cache_data(max_entries=1,show_spinner="Fine Tuning has started... Please wait!")
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

@st.cache_data(max_entries=1)
def get_baseline(start, end,ticker):
    df_dji = YahooDownloader(
        start_date=start, end_date=end, ticker_list=[ticker]).fetch_data()
    dji1 = df_dji[["date", "close"]]
    fst_day = dji1["close"][0]
    dji = pd.merge(
        dji1["date"],
        dji1["close"].div(fst_day).mul(1000000),
        how="outer",
        left_index=True,
        right_index=True,
    ).set_index("date")
    return df_dji,dji


@st.cache_data(max_entries=1)
def get_daily_return(df, value_col_name="account_value"):
    df["daily_return"] = df[value_col_name].pct_change(1)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True, drop=True)
    df.index = df.index.tz_localize("UTC")
    return pd.Series(df["daily_return"], index=df.index)


@st.cache_data(max_entries=1)
def plot_values(dji,account_val):
    fig = go.Figure()

    # Plot cumulative returns as line plots
    fig.add_trace(go.Line(x=dji.index, y=dji, mode='lines', name='Actual Value'))
    fig.add_trace(go.Line(x=account_val.index, y=account_val, mode='lines', name='Predicted Value'))

    fig.update_layout(title='Actual Vs Predicted',xaxis_title='Date',yaxis_title='Close Returns',width=1100,height=500)

    st.plotly_chart(fig)
@st.cache_data(max_entries=1)
def plot_returns(test_returns, baseline_returns):
    # Create subplots
    fig = go.Figure()

    # Plot cumulative returns as line plots
    fig.add_trace(go.Line(x=test_returns.index, y=test_returns, mode='lines', name='Test Asset'))
    fig.add_trace(go.Line(x=baseline_returns.index, y=baseline_returns, mode='lines', name='Baseline Asset'))

    # Calculate monthly returns
    test_monthly = test_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    baseline_monthly = baseline_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

    # Create subplots
    fig.add_trace(go.Bar(x=test_monthly.index, y=test_monthly, name='Test Asset (Monthly)', yaxis="y2"))
    fig.add_trace(go.Bar(x=baseline_monthly.index, y=baseline_monthly, name='Baseline Asset (Monthly)', yaxis="y2"))

    fig.update_layout(
        title="Returns Analysis",
        xaxis_title="Date",
        yaxis_title="Cumulative Returns",
        yaxis2=dict(title="Monthly Returns", overlaying="y", side="right"),
        width=900,  # Adjust the width as needed
        height=600    # Adjust the height as needed
    )

    st.plotly_chart(fig)

timesteps = None
trained_agent = None
if selected_agent is not None:
    # Add a scrollbar to select the timesteps
    timesteps = st.slider("Select timestep", min_value=1000, max_value=100000, step=2000)
df_account = None
if st.button("Train Agent") and action == 'Train Agent':
    trained_agent = train_agent(df_unique, selected_agent.lower(), timesteps)
    df_account, _ = DRLAgent.DRL_prediction(model=trained_agent, environment=e_trade_gym)

    test_returns = get_daily_return(df_account, value_col_name='account_value')
    test_returns = pd.DataFrame(test_returns)
    test_returns['date'] = test_returns.index
    test_returns = test_returns.reset_index(drop=True)
    test_returns.index = pd.to_datetime(test_returns['date'])

    baseline_df,baseline_account = get_baseline(TRADE_START, TRADE_END,ticker=Use_ticker)
    print(baseline_df)
    baseline_df = baseline_df.fillna(method="ffill").fillna(method="bfill")
    baseline_returns = get_daily_return(baseline_df, value_col_name="close")

    # Plot cumulative and monthly returns
    plot_values(baseline_account['close'],df_account['account_value'].fillna(method='ffill'))
    st.dataframe(baseline_df.head(5), use_container_width=True)
    st.dataframe(df_account.head(5), use_container_width=True)
    plot_returns(df_account['daily_return'],baseline_df['daily_return'])



elif st.button('FineTune Agent') and action == 'FineTune Agent':

    # a2c_params = ['learning_rate','normalize_advantage']
    # ppo_params = ['learning_rate','batch_size']
    # others_params = ['buffer_size','learning_rate','batch_size']

    # model = None
    # num_trials = 10
    # study_name = f"{selected_agent.lower()}_study"
    # study = run_optimization(selected_agent.lower(), study_name, 10)


    # if study is not None:
    #     agent = DRLAgent(env=env_train)
    #     if selected_agent == 'A2C':
    #         model = agent.get_model('a2c', model_kwargs={a2c_params[0]: study.best_params['learning_rate'],
    #                                                      a2c_params[1]: study.best_params['normalize_advantage']})
    #     elif selected_agent == 'PPO':
    #         model = agent.get_model('ppo', model_kwargs={ppo_params[0]: study.best_params['learning_rate'],
    #                                                      ppo_params[1]: study.best_params['batch_size']})
    #     else:
    #         model = agent.get_model(selected_agent.lower(),
    #                                 model_kwargs={others_params[1]: study.best_params['learning_rate'],
    #                                               others_params[0]: study.best_params['buffer_size'],
    #                                               others_params[2]: study.best_params['batch_size']})

        # with st.spinner("Training has started (with tuned parameters)... Please wait!"):
        #     tuned_agent = agent.train_model(model=model,
        #                                     tb_log_name=selected_agent.lower(),
        #                                     total_timesteps=timesteps)
    
    model_path = 'models/{0}.pth'.format(selected_agent)
    if selected_agent == 'A2C':
        tuned_agent = A2C.load(model_path, env=env_train)
    elif selected_agent == 'DDPG':
        tuned_agent = DDPG.load(model_path, env=env_train)
    elif selected_agent == 'PPO':
        tuned_agent = PPO.load(model_path, env=env_train)

    df_account, _ = DRLAgent.DRL_prediction(model=tuned_agent, environment=e_trade_gym)

    test_returns = get_daily_return(df_account, value_col_name='account_value')
    test_returns = pd.DataFrame(test_returns)
    test_returns['date'] = test_returns.index
    test_returns = test_returns.reset_index(drop=True)
    test_returns.index = pd.to_datetime(test_returns['date'])

    baseline_df,baseline_account = get_baseline(TRADE_START, TRADE_END,ticker=Use_ticker)
    baseline_df = baseline_df.fillna(method="ffill").fillna(method="bfill")
    baseline_returns = get_daily_return(baseline_df, value_col_name="close")

    # Plot cumulative and monthly returns
    plot_values(baseline_account['close'],df_account['account_value'].fillna(method='ffill'))
    st.dataframe(baseline_df.head(5), use_container_width=True)
    st.dataframe(df_account.head(5), use_container_width=True)
    plot_returns(df_account['daily_return'],baseline_df['daily_return'])
    

st.title('🦙 Llama Banker')

model_sent = "ProsusAI/finbert"
model_sum = "Falconsai/text_summarization"
tokens = 2048


@st.cache_resource
def sentiment_model(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

@st.cache_resource(max_entries=1,show_spinner='Loading Language Model...Please wait!')
def summary_model(model_name):
    summarizer = pipeline("summarization", model=model_name)
    return summarizer


@st.cache_data(max_entries=1)
def split_news(news, max_tokens):
    news_items = news.split('\n')
    split_news_list = []
    current_item = ""

    for item in tqdm(news_items):
        if len(current_item) + len(item) <= max_tokens:
            current_item += item + '\n'
        else:
            split_news_list.append(current_item.strip())  # Remove trailing newline
            current_item = item + '\n'

    # Add the last item
    split_news_list.append(current_item.strip())

    return split_news_list


# Step 1: Create text fields for start date and end date with default values
# st.success('Note: Loading News takes time...Choose dates wisely!')
# start_date = str(st.date_input("Start Date", datetime.date(2023,1,1)))
# end_date = str(st.date_input("End Date", datetime.date(2023,1,8)))
# News_list = []


# @st.cache_data(max_entries=1,show_spinner='Downloading News ....Please wait!')
# def download_news(start_date, end_date):
#     config = {
#         "use_proxy": "us_free",
#         "max_retry": 5,
#         "proxy_pages": 5,
#         "token": "ckc09r1r01qjeja48ougckc09r1r01qjeja48ov0"
#     }
#     stock = Use_ticker
#     news_downloader = Finnhub_Date_Range(config)
#     news_downloader.download_date_range_stock(str(start_date), str(end_date), stock=stock)
#     news_downloader.gather_content()
#     df = news_downloader.dataframe
#     df["date"] = df.datetime.dt.date
#     df["date"] = df["date"].astype("str")
#     df = df.sort_values("datetime")
#     news_list = list(df['headline'])

#     st.success(f"Downloaded {len(news_list)} news articles.")
#     return news_list

finviz_url = 'https://finviz.com/quote.ashx?t='
def get_news(ticker):
    url = finviz_url + ticker
    req = Request(url=url,headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'}) 
    response = urlopen(req)    
    # Read the contents of the file into 'html'
    html = BeautifulSoup(response,features='lxml')
    # Find 'news-table' in the Soup and load it into 'news_table'
    news_table = html.find(id='news-table')
    return news_table

# parse news into dataframe
def parse_news(news_table):
    parsed_news = []
    
    for x in news_table.findAll('tr'):
        # read the text from each tr tag into text
        # get text from a only
        text = x.a.get_text() if x.a else ''
        # splite text in the td tag into a list 
        date_scrape = x.td.text.split()
        # if the length of 'date_scrape' is 1, load 'time' as the only element

        if len(date_scrape) == 1:
            time = date_scrape[0]
            
        # else load 'date' as the 1st element and 'time' as the second    
        else:
            date = date_scrape[0]
            time = date_scrape[1]
        
        # Append ticker, date, time and headline as a list to the 'parsed_news' list
        parsed_news.append([date, time, text])        
        # Set column names
        columns = ['date', 'time', 'headline']
        # Convert the parsed_news list into a DataFrame called 'parsed_and_scored_news'
        parsed_news_df = pd.DataFrame(parsed_news, columns=columns)        
        # Create a pandas datetime object from the strings in 'date' and 'time' column
        #parsed_news_df['datetime'] = pd.to_datetime(parsed_news_df['date'] + ' ' + parsed_news_df['time'])
        
    return parsed_news_df

@st.cache_data
def download_news(ticker):
    news_table= get_news(ticker)
    parsed_news_df = parse_news(news_table)
    headlines = '. \n'.join(list(parsed_news_df['headline']))
    return headlines

news = None
split_news_list = None
load_news = st.checkbox("Load News")
# Step 2: Create a button to download news
if load_news:
    st.cache_data.clear()
    st.cache_resource.clear()
    news= download_news(Use_ticker)

# Step 3: Create a button to perform sentiment analysis and summarization
if load_news and news is not None:
    sentiments = []
    probabilities = []
    summaries = []
    split_news_list = split_news(news,max_tokens=tokens)
    model, tokenizer = sentiment_model(model_sent)
    summarizer = summary_model(model_name=model_sum)

    for i, news_item in enumerate(split_news_list):
        inputs = tokenizer(news_item, return_tensors="pt")
        outputs = model(**inputs)

        probs = F.softmax(outputs.logits, dim=-1)
        sentiment = model.config.id2label[probs.argmax().item()]
        probability = probs.max().item()
        sentiments.append(sentiment)
        probabilities.append(probability)
    
        summary = summarizer(news_item,max_length=200,min_length=30,do_sample=False)[0]["summary_text"]
        if summary[-1] == '.':
            pass
        else:
            summary += '.'
        summary += "\n"

        summaries.append(summary)
    sentimode = mode(sentiments)
    probmean = mean(probabilities)
    result_summary = "\n".join(summaries)
    st.subheader("Headlines")
    st.success(result_summary)
    st.subheader("Sentiment Analysis")
    st.success(f'Sentiment: {sentimode.upper()}')
    st.success(f'Score: {probmean:.2f}')
    if st.button("Clear All"):
        st.cache_data.clear()
        st.cache_resource.clear()

else:
    pass
    
    

#################################################################################################################################################



