from numpy import *
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from matplotlib import *
from matplotlib.pyplot import *
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from matplotlib import gridspec
#from nsepy import get_history
from tabulate import tabulate
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


def Recovery_Trauma(capital):
    Draw_down = np.linspace(5,95,25)
    Start_Capital = np.empty(size(Draw_down))
    Efforts = np.empty(size(Draw_down))
    Start_Capital = capital*(1 - .01*Draw_down)
    Efforts = np.around(100*(Capital - Start_Capital)/(Start_Capital),1)
    d = dict(A=Draw_down, B=Start_Capital,C=Efforts)
    Rec_Trau = pd.DataFrame(dict([ (pd.Series(k)) for k in d.items() ]))
    Rec_Trau.columns = ["Draw down %","Starting Capital(Rs)","Efforts %"] 
    return Rec_Trau


# Position sizing
def Core_Eq_Model(Capital,Position_size, N_trade):
    Trade = np.linspace(1,10,N_trade).astype(int)
    Available_Equity = np.empty(len(Trade))
    Trade_exposure = np.empty(len(Trade))
    Core_Equity = np.empty(len(Trade))
    Available_Equity[0] = Capital     #(1-(Position_size/100))*Capital
    Trade_exposure[0] = Capital - (1-(Position_size/100))*Capital #Available_Equity[0]
    Core_Equity[0] =  Available_Equity[0] - Trade_exposure[0]

    for i in range(1,len(Trade)):
        Available_Equity[i] = Core_Equity[i-1]
        Trade_exposure[i] = Available_Equity[i]/Position_size
        Core_Equity[i] = Available_Equity[i] - Trade_exposure[i]
    d = dict(A=Trade, B=np.around(Available_Equity,0), C=np.around(Trade_exposure,0), D=np.around(Core_Equity,0))
    CE = pd.DataFrame(dict([ (pd.Series(k)) for k in d.items() ]))
    CE.columns = ["Trade", "Available Equity","Trade_exposure","Core_Equity"] 
    return CE


# Percentage of invest amount from total capital for fixed 2% volatility 
# using Average True Range 
def Percentage_Volatility_buy(df,capital):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    
    atr = true_range.rolling(14).sum()/14
    ATR_value = atr.iloc[-1]
    voltality = (2/100)*capital
    No_shares = voltality/ATR_value
    Exposure = No_shares*df["Close"].iloc[-1]
    print("Average True Range = ", round(atr.iloc[-1],2),"Rs" )
    print("Volatility = ", int(voltality) )
    print("No of shares to buy = ", int(No_shares))
    print("Exposure = ", round(Exposure,2), "Rs")
    print("Capital available from Reduced Total Equity= ", round(capital-Exposure,2), "Rs")