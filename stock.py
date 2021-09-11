import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from matplotlib import *
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.font_manager import FontProperties
from matplotlib import gridspec
import json
from urllib.parse import quote
import requests
from pandas.io.json import json_normalize
try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen
#import seaborn as sns
#sns.set(style='darkgrid', context='talk', palette='Dark2')
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
   
# function to read data through API and reurn as dataframe
def json_data(url):
    response = urlopen(url)
    elevations = response.read()
    data = json.loads(elevations)
    # to dataframe
    df = pd.json_normalize(data['results'])
    return df
    
# Calculates simple interest
def sint(principle, rate, time):
    interest = (principle*rate*time)/100
    print("simple interest  = ", interest)
    print("Total Simple Return  = ", interest+principle)

# Calculates compound interest
def cint(principle, rate, time):
    Amount = principle * (pow((1 + rate / 100), time))
    CI = Amount - principle
    PC = Amount # principle + compound
    print("Compound Interest  = ", CI)
    print("Total Compund Return  = ", PC)

# Calculates percentage
def percent(interest, principle, time):      
    rate = interest*100/(principle*time)
    print("Percentage  = ",rate, "%")
    
def loss_percent(buy_price, selling_price):
    if(buy_price > selling_price ):
        loss = ((buy_price-selling_price)/buy_price)*100
        print("Loss Percentage = ", loss, "%")
    else:
        print("Error Input(selling_price is high!)")
    
def profit_percent(buy_price, selling_price):
    if(selling_price>buy_price):
        profit = ((selling_price-buy_price)/buy_price)*100
        print("Profit Percentage = ", profit, "%")
    else:
         print("Error Input(selling_price is low!)")
    

# simple functions to calculate basic needs
# Relative strength index from exponential moving average
def rsi_ema(df, periods):
    close_prime = df['Close'].diff()    
    #  lower close and higher close
    up = close_prime.clip(lower=0)
    down = -1 * close_prime.clip(upper=0)
    ema_up = up.ewm(com = periods - 1, min_periods = periods).mean()
    ema_down = down.ewm(com = periods - 1, min_periods = periods).mean()
        
    rsi = ema_up / ema_down
    rsi = 100 - (100/(1 + rsi))
    return rsi

# Relative strength index from simple moving average
def rsi_dma(df, periods):
    close_prime = df['Close'].diff()    
    # lower closes and higher closes
    up = close_prime.clip(lower=0)
    down = -1 * close_prime.clip(upper=0)
    sma_up = up.rolling(window = periods).mean()
    sma_down = down.rolling(window = periods).mean()       
    rsi = sma_up /sma_down
    rsi = 100 - (100/(1 + rsi))
    return rsi

def candlestick(df):
    t = df["Date"]
    o = df["Open"] 
    h = df["High"]
    l = df["Low"]
    c = df["Close"]
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    years_fmt = mdates.DateFormatter('%Y')
    fig, ax = plt.subplots(figsize=(9, 4),dpi=700)
    #plt.title('(closing price - 200 DMA) %', fontsize=12)
    plt.xticks(fontsize=10)
    color = ["green" if close_price > open_price else "red" for close_price, open_price in zip(c, o)]
    ax.bar(x=t, height=np.abs(o-c), bottom=np.min((o,c), axis=0), width=1, color=color)
    ax.bar(x=t, height=h-l, bottom=l, width=0.5, color=color)
    ax.grid(b=None, which='major', axis='both')
    ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_locator(months)
    ax.set_ylabel('Price')
    #ax.legend(loc='center left', ncol =2, bbox_to_anchor=(.1, 0.9))
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

def analysis(df):
    l1 = len(df)
    print("OUTPUT: ANALYSIS BASED ON -> BASE PRICE & DMA")
    print('Maximum close =',max(df['Close']), 'in', l1, 'Days')
    print('Minimum close =',min(df['Close']), 'in', l1, 'Days')
    print()
    # Base Price = mean value of closing prices about 3 years
    BP = df["Close"].mean()
    print("Base Price = ", BP, "from", l1, "Days")
    print()
    #tail_200 = df['Close'].tail(200)
    Last_DMA200 = np.mean(df['Close'].tail(200))
    Last_DMA100 = np.mean(df['Close'].tail(100))

    print("LAST 200 DMA = ", Last_DMA200)
    print("LAST 100 DMA = ", Last_DMA100)
    print()

    last_20_min = min(df['Close'].tail(20))
    last_20_max = max(df['Close'].tail(20))
    print("Last 20 Days Min. = ", last_20_min)
    print("Last 20 Days Max. = ", last_20_max)
    print()
    current_price = df['Close'].iloc[-1]
    print("Current  Price = ", current_price)
    print()

    year_min = min(df['Close'].tail(365))
    year_max = max(df['Close'].tail(365))

    yr_ratio = year_max/year_min

    print("Last 365 Days Min. = ", year_min)
    print("Last 365 Days Max. = ", year_max)
    print("Last 365 Days Max/Min = ", yr_ratio, '(less than 2% is GOOD)')
    print()

    base_price = BP
    #print('Maximum close =',max(df['Close']))
    if(current_price > base_price):
        loss_percent(current_price,base_price)
    elif(current_price < base_price):
        profit_percent(current_price,base_price)
    
    print()
    print("WARNING: BUY ONLY IF LOSS IS BELOW 20% OR PROFIT 15% HIGH")
    print()

# visualize the moving average
    date_array  = df['Date']
    close_array = df['Close']
# 
    short_roll = df.rolling(window=20).mean()
    long_roll = df.rolling(window=50).mean()
    long_std = df.rolling(window=50).std()

    print("long_roll closing price ", long_roll['Close'].iloc[-1])
    print("short_roll closing price", short_roll ['Close'].iloc[-1])
    
    # candle stick plot
    o = df["Open"] 
    h = df["High"]
    l = df["Low"]
    c = df["Close"]
    
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    years_fmt = mdates.DateFormatter('%Y')
#
    fig = plt.subplots(figsize=(11, 10), dpi=900)
    gs = gridspec.GridSpec(3, 1) # , width_ratios=[3, 1]
    ax0 = plt.subplot(gs[0])
    color = ["green" if close_price > open_price else "red" for close_price, open_price in zip(c, o)]
    ax0.bar(df["Date"], height=np.abs(o-c), bottom=np.min((o,c), axis=0), width=0.7, color=color)
    ax0.bar(df["Date"], height=h-l, bottom=l, width=0.25, color=color)
    #ax0.plot(date_array, close_array, linestyle='-', color='blue', alpha=0.7)
    ax0.plot(date_array, short_roll['Close'], color='tab:green',  label = '20 days MA') 
    ax0.plot(date_array, long_roll['Close'], color='tab:red',  alpha=0.7, label = '50 days MA') 
    plt.grid(b=None, which='major', axis='both')
    #ax0.plot(date_array, long_std['Close'], color='k',  alpha=0.7, label = 'every 50 days std') 
    ax0.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax0.xaxis.set_major_locator(months)
    ax0.set_ylabel('Price')
    ax0.legend(loc='center left', ncol =2, bbox_to_anchor=(.1, 0.9))
    for tick in ax0.get_xticklabels():
        tick.set_rotation(90)
    #
    ax1 = plt.subplot(gs[1])
    plt.title('Stock Volume')
    ax1.bar(df['Date'], df['Volume'], linestyle='-', color='blue', alpha=0.7)
    ax1.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax1.xaxis.set_major_locator(months)
    plt.grid(b=None, which='major', axis='both')
    ylabel('No. of stocks')
    for tick in ax1.get_xticklabels():
        tick.set_rotation(90)
        #
    ax2 = plt.subplot(gs[2])  
    plt.title('Histogram of closing price')   
    ax2.set_xlabel('Price')
    df['Close'].plot.hist(bins=10)   
    plt.tight_layout()

# 
def macd_strategy(prices, data):    
    buy_price = []
    sell_price = []
    macd_signal = []
    signal = 0

    for i in range(len(data)):
        if data['macd'][i] > data['signal'][i]:
            if signal != 1:
                buy_price.append(prices[i])
                sell_price.append(np.nan)
                signal = 1
                macd_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                macd_signal.append(0)
        elif data['macd'][i] < data['signal'][i]:
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(prices[i])
                signal = -1
                macd_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                macd_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            macd_signal.append(0)
            
    return buy_price, sell_price, macd_signal    
    
#
def MACD_RSI_BB(df,period_rsi, period_bb): 
    # calculate MACD
    exp1 = df['Close'].ewm(span = 12, adjust = False).mean()
    exp2 = df['Close'].ewm(span = 26, adjust = False).mean()
    macd = pd.DataFrame(exp1 - exp2).rename(columns = {'Close':'macd'})
    
    signal = pd.DataFrame(macd.ewm(span = 9, adjust = False).mean()).rename(columns = {'macd':'signal'})
    hist = pd.DataFrame(macd['macd'] - signal['signal']).rename(columns = {0:'hist'})
    frames =  [macd, signal, hist]
    df_macd = pd.concat(frames, join = 'inner', axis = 1)   
    
    buy_price, sell_price, macd_signal  = macd_strategy(df['Close'], df_macd)
    months = mdates.MonthLocator()  # every month
    print("MACD > SIGNAL LINE : BUY")
    print()
    print("MACD < SIGNAL LINE : SELL")
    # calculate RSI
    rsi = rsi_dma(df, period_rsi)
    # calculate Bollinger Bands
    multiplier = 2
    sma = df["Close"].rolling(window=period_bb).mean()
    df['UpperBand'] = df['Close'].rolling(period_bb).mean() + df['Close'].rolling(period_bb).std() * multiplier
    df['LowerBand'] = df['Close'].rolling(period_bb).mean() - df['Close'].rolling(period_bb).std() * multiplier  
  
    #------------------PLOT--------------------------
    fig, ax = plt.subplots(figsize=(11, 12), dpi=400)
    gs = gridspec.GridSpec(4, 1)   
    ax0 = plt.subplot(gs[0])
    ax0.plot(df['Date'], df['Close'], color = 'blue', linewidth = 1.0, alpha=0.7, label = 'Close Price')
    ax0.plot(df['Date'], buy_price, marker = '^', color = 'green', markersize = 10, label = 'BUY SIGNAL', linewidth = 0)
    ax0.plot(df['Date'], sell_price, marker = 'v', color = 'r', markersize = 10, label = 'SELL SIGNAL', linewidth = 0)
    ax0.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax0.xaxis.set_major_locator(months)
    leg1 = legend(loc='center left', ncol =3, handlelength = .9, columnspacing = 0.3, 
       frameon = True, fancybox=True, shadow=False, bbox_to_anchor=(0.01, 0.93))
    plt.grid(b=None, which='major', axis='both')
    plt.xticks(fontsize=10)
    ylabel('Price')
    for tick in ax0.get_xticklabels():
        tick.set_rotation(90)   
    #---------------------------
    ax1 = plt.subplot(gs[1])
    plt.title('Moving Average Covergence Divergence', fontsize=12)
    ax1.plot(df['Date'], df_macd['macd'], color = 'blue', linewidth = 1.0, alpha=0.7, label = 'MACD')
    ax1.plot(df['Date'], df_macd['signal'], color = '#FF4500', linewidth = 1.0, alpha=0.7, label = 'SIGNAL')
    leg2 = legend(loc='center left', ncol =2, handlelength = .9, columnspacing = 0.3, 
       frameon = True, fancybox=True, shadow=False, bbox_to_anchor=(0.01, 0.93))
    plt.grid(b=None, which='major', axis='both')
    for i in range(len(df['Close'])):
        if str(df_macd['hist'][i])[0] == '-':
            ax1.bar(df['Close'].index[i], df_macd['hist'][i], color = 'red')
        else:
            ax1.bar(df['Close'].index[i], df_macd['hist'][i], color = 'green')
    ax1.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax1.xaxis.set_major_locator(months)
    plt.xticks(fontsize=10)
    ylabel('MACD')
    for tick in ax1.get_xticklabels():
        tick.set_rotation(90)
   #---------------------------
    ax2 = plt.subplot(gs[2])    
    plt.title('Relative Strength Index', fontsize=12)
    ax2.plot(df['Date'], rsi, linestyle='-', color='blue', alpha=0.7, linewidth = 1.0)
    plt.axhline(40, color='r', linestyle='-', linewidth = 1.0 )
    plt.axhline(50, color='k', linestyle='--',linewidth = 1.0)
    plt.axhline(60, color='g', linestyle='-',linewidth = 1.0)
    plt.grid(b=None, which='major', axis='both')
    ax2.axhspan(40, 60, color='tab:blue',alpha=0.34)
    ax2.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax2.xaxis.set_major_locator(months)
    plt.xticks(fontsize=10)
    ylabel('RSI')
    for tick in ax2.get_xticklabels():
        tick.set_rotation(90)    
    #---------------------------------------   
    ax3 = plt.subplot(gs[3]) 
    plt.title('Bollinger Bands', fontsize=12)
    ax3.plot(df['Date'], sma, linestyle='-', color='blue', alpha=0.7, label='SMA', linewidth = 1.0)
    ax3.plot(df['Date'], df['UpperBand'], linestyle='-', color='tab:green', alpha=0.9,label='Upper Band', linewidth = 1.0)
    ax3.plot(df['Date'], df['LowerBand'], linestyle='-', color='tab:red', alpha=0.9,label='Lower Band', linewidth = 1.0)
    plt.fill_between(df['Date'], df['LowerBand'], y2=df['UpperBand'], color='tab:blue', alpha=0.25)
    plt.grid(b=None, which='major', axis='both')
    ax3.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    plt.xticks(fontsize=10)
    ax3.xaxis.set_major_locator(months)
    leg1 = legend(loc='center left', ncol =3, handlelength = .9, columnspacing = 0.3, 
       frameon = True, fancybox=True, shadow=False, bbox_to_anchor=(0.01, 0.9))
    ylabel('Price')
    for tick in ax3.get_xticklabels():
        tick.set_rotation(90)
    plt.tight_layout() 
    
def MACD_HOLD_strategy(df):
    date = df['Date']
    exp1 = df['Close'].ewm(span = 12, adjust = False).mean()
    exp2 = df['Close'].ewm(span = 26, adjust = False).mean()
    macd = pd.DataFrame(exp1 - exp2).rename(columns = {'Close':'macd'})
    
    signal = pd.DataFrame(macd.ewm(span = 9, adjust = False).mean()).rename(columns = {'macd':'signal'})
    hist = pd.DataFrame(macd['macd'] - signal['signal']).rename(columns = {0:'hist'})
    frames =  [macd, signal, hist]
    df_macd = pd.concat(frames, join = 'inner', axis = 1)   
    buy_price, sell_price, macd_signal = macd_strategy(df['Close'],df_macd)

    position = []
    for i in range(len(macd_signal)):
        if macd_signal[i] > 1:
            position.append(0)
        else:
            position.append(1)
        
    for i in range(len(df['Close'])):
        if macd_signal[i] == 1:
            position[i] = 1
        elif macd_signal[i] == -1:
            position[i] = 0
        else:
            position[i] = position[i-1]
        
    macd = df_macd['macd']
    signal = df_macd['signal']
    close_price = df['Close']
    macd_signal = pd.DataFrame(macd_signal).rename(columns = {0:'macd_signal'}).set_index(df.index)
    position = pd.DataFrame(position).rename(columns = {0:'macd_position'}).set_index(df.index)

    frames = [date, close_price, macd, signal, macd_signal, position]
    strategy = pd.concat(frames, join = 'inner', axis = 1)
    strategy.tail(10)
    return strategy    
          
# MOVING AVERAGE CONVERGENCE DIVERGENCE and RELATIVE STRENGTH INDEX
def MACD_RSI_old(df,periods):
   # y = df["Close"]
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = exp1-exp2
    exp3 = macd.ewm(span=9, adjust=False).mean()
    date_array1 = df['Date']
   # ---------- CALCULATE RSI FROM EMA --------------
    ema = rsi_ema(df, periods)
    dma = rsi_dma(df,  periods)
   #-------------------------------------------------  
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    years_fmt = mdates.DateFormatter('%Y')
    # plot
    print("If MCDA > SIGNAL LINE : BUY")
    fig = plt.subplots(figsize=(10, 7), dpi=400)
    gs = gridspec.GridSpec(2, 1)      
    ax1 = plt.subplot(gs[0])
    plt.title('Relative Strength Index', fontsize=12)
    #ax1.plot(date_array1, ema, linestyle='--', color='blue', alpha=0.7, label = 'EMA')
    ax1.plot(date_array1, dma, linestyle='-', color='blue', alpha=0.7, linewidth = 1.0)
    plt.axhline(40, color='r', linestyle='-', linewidth = 1.0 )
    plt.axhline(50, color='k', linestyle='--',linewidth = 1.0)
    plt.axhline(60, color='g', linestyle='-',linewidth = 1.0)
    ax1.axhspan(40, 60, color='tab:blue',alpha=0.34)
    ax1.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax1.xaxis.set_major_locator(months)
    plt.xticks(fontsize=10)
    ylabel('Change')
    for tick in ax1.get_xticklabels():
        tick.set_rotation(90)   
#   
    ax0 = plt.subplot(gs[1])
    plt.title('Moving Average Covergence Divergence', fontsize=12)
    ax0.plot(date_array1, macd, linestyle='-', color='blue', alpha=0.7, label = 'MACD')
    ax0.plot(date_array1, exp3, linestyle='-', color='#FF4500', alpha=0.7, label = 'signal')
    plt.axhline(0, color='k', linestyle='--',linewidth = 1.0)
    ax0.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax0.xaxis.set_major_locator(months)
    plt.xticks(fontsize=10)
    leg1 = legend(loc='center left', ncol =2, handlelength = .9, columnspacing = 0.3, 
       frameon = True, fancybox=True, shadow=False, bbox_to_anchor=(0.2, 0.9))
    ylabel('Change')
    for tick in ax0.get_xticklabels():
        tick.set_rotation(90)
    plt.tight_layout()   
        
                
def BB(df,period):    
    multiplier = 2
    sma = df["Close"].rolling(window=period).mean()
    df['UpperBand'] = df['Close'].rolling(period).mean() + df['Close'].rolling(period).std() * multiplier
    df['LowerBand'] = df['Close'].rolling(period).mean() - df['Close'].rolling(period).std() * multiplier
    date_array = df['Date']
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    years_fmt = mdates.DateFormatter('%Y')
    fig, ax = plt.subplots(figsize=(10, 4), dpi=400)
    plt.title('Bollinger Bands', fontsize=12)
    ax.plot(date_array, sma, linestyle='-', color='blue', alpha=0.7, label='SMA', linewidth = 1.5)
    ax.plot(date_array, df['UpperBand'], linestyle='-', color='tab:green', alpha=0.9,label='Upper Band', linewidth = 1.5)
    ax.plot(date_array, df['LowerBand'], linestyle='-', color='tab:red', alpha=0.9,label='Lower Band', linewidth = 1.5)
    plt.fill_between(date_array, df['LowerBand'], y2=df['UpperBand'], color='tab:blue', alpha=0.25)
    ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    plt.xticks(fontsize=10)
    ax.xaxis.set_major_locator(months)
    leg1 = legend(loc='center left', ncol =3, handlelength = .9, columnspacing = 0.3, 
       frameon = True, fancybox=True, shadow=False, bbox_to_anchor=(0.01, 0.9))
    ylabel('Price')
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

# HELPS TO DO REVERSE SAFE TRADING
# build a 200 DMA and calculate change with CMP and calculate change percentage
# include everything in a data file 
def dma_analysis(df):
    df1 = df[['Date', 'Close']]
    DMA_200 = df1['Close'].rolling(window=200).mean()
    df1['200 DMA'] = DMA_200
    change = df1['Close']- df1['200 DMA']
    df1['Change'] = change
    loss = ((df1['Close'] - df1['200 DMA'])/df1['Close'])*100
    df1['Percentage'] = loss

# clean the data : replace inf and inf and nan
    df1.replace([np.inf, -np.inf], np.nan, inplace=True)
    df1 = df1.fillna(0)

    date_array2 = df1['Date']
    print("SELL: 1 part of stock for every 5% fall in 200DMA")
    print()
    print("BUY :  1 part of stock for every 5% rise in 200DMA")
    print(df1.tail(20))
# save data with new columns
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    years_fmt = mdates.DateFormatter('%Y')
# plot percenrtage
    fig, ax = plt.subplots(figsize=(10, 4),dpi=400)
    plt.title('(closing price - 200 DMA) %', fontsize=12)
    plt.xticks(fontsize=10)
    ax.plot(date_array2, df1['Percentage'], linestyle='-', color='blue', alpha=0.7)
    plt.axhline(0, color='tab:green', linestyle='--',linewidth = 1.0)
    plt.grid(b=None, which='major', axis='both')
    ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_locator(months)
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)    
    return df1
   
def dma_analysis_period(df,period1, period2, period3, period4):
    df = df[['Date', 'Close']]
    DMA_200 = df['Close'].rolling(window=period1).mean()
    DMA_100 = df['Close'].rolling(window=period2).mean()
    DMA_50 = df['Close'].rolling(window=period3).mean()
    DMA_20 = df['Close'].rolling(window=period4).mean()
    df['200 DMA'] = DMA_200
    df['100 DMA'] = DMA_100
    df['50 DMA'] = DMA_50
    df['20 DMA'] = DMA_20
    change1 = df['Close']- df['200 DMA']
    change2 = df['Close']- df['100 DMA']
    change3 = df['Close']- df['50 DMA']
    change4 = df['Close']- df['20 DMA']
    df['Change1'] = change1
    df['Change2'] = change2
    df['Change3'] = change3
    df['Change4'] = change4
    loss1 = ((df['Close'] - df['200 DMA'])/df['Close'])*100
    loss2 = ((df['Close'] - df['100 DMA'])/df['Close'])*100
    loss3 = ((df['Close'] - df['50 DMA'])/df['Close'])*100
    loss4 = ((df['Close'] - df['20 DMA'])/df['Close'])*100  
    df['Percentage1'] = loss1
    df['Percentage2'] = loss2
    df['Percentage3'] = loss3
    df['Percentage4'] = loss4
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(0)
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    years_fmt = mdates.DateFormatter('%Y')
# plot percenrtage
    fig, ax = plt.subplots(figsize=(10, 3),dpi=400)
    plt.title('(closing price - 200 DMA) %', fontsize=12)
    plt.xticks(fontsize=10)
    ax.plot(df['Date'], df['Percentage1'], linestyle='-', color='tab:green', alpha=0.7, label='200 DMA')
    ax.plot(df['Date'], df['Percentage2'], linestyle='-', color='tab:blue', alpha=0.7, label='100 DMA')
    ax.plot(df['Date'], df['Percentage3'], linestyle='-', color='blue', alpha=0.7, label='50 DMA')
    ax.plot(df['Date'], df['Percentage4'], linestyle='-', color='tab:red', alpha=0.7, label='20 DMA')
    plt.axhline(0, color='k', linestyle='--',linewidth = 1.5)
    plt.grid(b=None, which='major', axis='both')
    ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_locator(months)
    leg1 = legend(loc='center left', ncol =4, handlelength = .9, columnspacing = 0.3, 
       frameon = True, fancybox=True, shadow=False, bbox_to_anchor=(0.01, 0.9))
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)    
    return df
 
def Volatility(df):
    df['log returns'] = (np.log( df['Close']/df['Close'].shift(1)))
     
# calculate daily standard deviation of returns
    daily_std = np.std(df['log returns'])
    l = len(df) # 252 trading days per year
# annualized daily standard deviation
   
    voltatility = daily_std*l** 0.5 # l = 252 if trading days for 1 year
    str_vol = str(round(voltatility, 4)*100)
    df['20 day Historical Volatility'] = 100*df['log returns'].rolling(window=20).std()
    
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    years_fmt = mdates.DateFormatter('%Y')
    fig = plt.subplots(figsize=(7, 6), dpi=400)
    gs = gridspec.GridSpec(2, 1)      
    ax1 = plt.subplot(gs[0])
    plt.title('Historical Volatility', fontsize=12)
    ax1.plot(df['Date'], df['20 day Historical Volatility'], linestyle='-', color='blue', alpha=0.7, linewidth = 1.0) 
    ax1.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax1.xaxis.set_major_locator(months)
    plt.grid(b=None, which='major', axis='both')
    plt.xticks(fontsize=10)
    ylabel('20 day Historical Volatility')
    for tick in ax1.get_xticklabels():
        tick.set_rotation(90)   
#   
    ax0 = plt.subplot(gs[1])
    plt.title('Volatility', fontsize=12)
    df['log returns'].hist(ax=ax0, bins=50, alpha=0.6, color='blue')
    ax0.set_xlabel('Log return')
    ax0.set_ylabel('Freq of log return')
    ax0.set_title('volatility:'  + str_vol + '%')
    plt.tight_layout()   
