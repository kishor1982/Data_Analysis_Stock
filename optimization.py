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
import scipy.optimize as sco
import scipy.interpolate as sci
from scipy import stats
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


def Variance(df,W):
    dfn = df.drop(['Date'], axis=1)
    Rt = dfn.pct_change()*100
    Rt_Av = round((Rt-np.mean(Rt)), 2)
    Rt_Av = Rt_Av.fillna(0)
    AR_mat = Rt_Av.to_numpy()
    XTX = np.matmul(AR_mat.T/100,AR_mat/100)
    C_mat = np.matmul(AR_mat.T/100,AR_mat/100)/len(dfn)
    #  Equity Daily return
    E_Rt_std = Rt.std()
    X = [E_Rt_std]
    E_Rt_std1 = np.array(X).T
    SD_mat = np.matmul(E_Rt_std1/100,E_Rt_std1.T/100)
    # Correlation Matrix
    CM = C_mat/SD_mat
    WSD = X*W/100
    TWSDC = np.matmul(CM,WSD.T/100)
    MM = np.matmul(WSD/100,TWSDC/100)
    Port_Var = np.sqrt(MM)
    print('Portfolio Variance =', Port_Var*100)
     # Save column names in to dictionary and save in to dataframe
    df_dic = [{c: i for i, c in enumerate(dfn.columns)}]
    dfc = pd.DataFrame.from_dict(df_dic)   
    df_CM = pd.DataFrame(CM)
    df_CM.columns=dfc.columns.values
    df_CM.index = df_CM.columns
    # return Correlation Matrix
    print("Correlation between the Equities:")
    #df_CM
    return df_CM 


# inputs : df, Capital =100, W-> Individual Weight in array, No_Days = 252 (in a year)
# Returns Daily Average and Yearly Reutrn
def Portfolio_SD_Return(df,Capital,W,No_Days):   
    dfn = df.drop(['Date'], axis=1)
    Rt = dfn.pct_change()*100
    Wt = np.empty([len(Rt), len(Rt.columns)])
    Inv = np.empty([len(Rt)])
    Inv_Day_Rt = np.empty(len(Rt))

    for j in range(0,len(Rt.columns)):
        Wt[0,j] = W[j]
        Wt[1,j] = W[j]*(Capital+Rt.iloc[1,j])/100
    

    for i in range(2,len(Rt)):
        for j in range(0,len(Rt.columns)):
            Wt[i,j] = Wt[i-1,j]*(Capital+Rt.iloc[:,j][i])/100
           
    
    Inv = np.array([sum(x) for x in zip(Wt)])
    Inv_Day_Rt[0] = 0
    for i in range(1,len(df)):
        Inv_Day_Rt[i] = round((Inv[i]/Inv[i-1] - 1)*100,2)
    DA = Rt.mean()
    Yr_Rt = DA*No_Days
    Exp_Ret = round(sum(W*Yr_Rt),2)
    
    STD_portfolio = np.std(Inv_Day_Rt)
    Ann_Var = STD_portfolio*np.sqrt(No_Days)
  
    print("Standard Deviation of Portfolio =",round(STD_portfolio,3),"%")
    print()
    print("Based on 1st STD (68% chances)")
    print("---------------------------------")
    print("Expected PF Return =", round(Exp_Ret/100,2),"%")
    print("Annaul PF Variance  =", round(Ann_Var,3),"%")
    print("---------------------------------")
    print("Lower Bound Profit = ",round(Exp_Ret/100,2)-round(Ann_Var,3),"%")
    print("Upper Bound Profit = ",round(Exp_Ret/100,2)+round(Ann_Var,3),"%")
    print("---------------------------------")
    print("Based on 2nd STD (95% chances)")
    print("---------------------------------")
    print("Annaul PF Variance  =", round(2*Ann_Var,3),"%")
    print("---------------------------------")
    print("Lower Bound Profit = ",round(Exp_Ret/100,2)-round(2*Ann_Var,3),"%")
    print("Upper Bound Profit = ",round(Exp_Ret/100,2)+round(2*Ann_Var,3),"%")
    print("---------------------------------")
    print("Based on 3rd STD (99% chances)")
    print("---------------------------------")
    print("Annaul PF Variance  =", (round(2*Ann_Var,3)),"%")
    print("---------------------------------")
    print("Lower Bound Profit = ",round(Exp_Ret/100,2)-round(3*Ann_Var,3),"%")
    print("Upper Bound Profit = ",round(Exp_Ret/100,2)+round(3*Ann_Var,3),"%")
    # return Daily Avg and Yearly Return
    
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    years_fmt = mdates.DateFormatter('%Y')
    fig, ax = plt.subplots(figsize=(9, 4),dpi=240)

    plt.xticks(fontsize=10)
    plt.title("Daily Change in Portfolio")
    ax.plot(df["Date"], Inv)
#ax.grid(b=None, which='major', axis='both')
    ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_locator(months)
    ax.set_ylabel('Price')
#ax.legend(loc='center left', ncol =2, bbox_to_anchor=(.1, 0.9))
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
        '''
    print("-----------------------------------")
    print("Daily Reurn %")
    print("-----------------------------------")
    print(round(DA,2))
    print("-----------------------------------")
    print("Yearly Reurn %")
    print("-----------------------------------")
    print(round(Yr_Rt))
    print("-----------------------------------")
    '''
    df_DY = pd.concat([round(DA,2), round(Yr_Rt,2)], axis=1)
    df_DY.columns = ['Daily Avg. Rt %', 'Yearly Return %']
    return df_DY #DA, Yr_Rt

def frequency(data, bins):
    # work with local sorted copy of bins for performance
    bins = bins[:]
    bins.sort()
    freqs = [0] * (len(bins)+1)
    for item in data:
        for i, bin_val in enumerate(bins):
            if item <= bin_val:
                freqs[i] += 1
                break
        else:
            freqs[len(bins)] += 1
    return freqs

def drange(Minimum, Maximum, Bin_Width):
    r = Minimum
    while r <= Maximum:
        yield r
        r += Bin_Width
                
def Frequency_Return(df,Capital,W):   
    dfn = df.drop(['Date'], axis=1)
    Rt = dfn.pct_change()*100
    Wt = np.empty([len(Rt), len(Rt.columns)])
    Inv = np.empty([len(Rt)])
    Inv_Day_Rt = np.empty(len(Rt))

    for j in range(0,len(Rt.columns)):
        Wt[0,j] = W[j]
        Wt[1,j] = W[j]*(Capital+Rt.iloc[1,j])/100
    

    for i in range(2,len(Rt)):
        for j in range(0,len(Rt.columns)):
             Wt[i,j] = Wt[i-1,j]*(Capital+Rt.iloc[:,j][i])/100
           
    
    Inv = np.array([sum(x) for x in zip(Wt)])
    Inv_Day_Rt[0] = 0

    for i in range(1,len(df)):
        Inv_Day_Rt[i] = round((Inv[i]/Inv[i-1] - 1)*100,2)
        Maximum = Inv_Day_Rt.max()
    Minimum = Inv_Day_Rt.min()

    No_of_obs = 50
    Bin_Width = (Maximum - (Minimum))/(No_of_obs)
    Low_Rt = Minimum+Bin_Width
    print("Bin Width = ", Bin_Width)
    order_ret = Inv_Day_Rt
    d = dict(A=order_ret, B=np.sort(order_ret)[::-1] )
    df_order = pd.DataFrame(dict([ (pd.Series(k)) for k in d.items() ]))
    df_order.columns = ["Portfolio Return ","Reordered return"] 
    
    BA = drange(Minimum, Maximum, Bin_Width)
    Bin_Array = np.array([ x for x in BA])
    data = Inv_Day_Rt.tolist()
    bins = Bin_Array.tolist()
    Frequency = np.array(frequency(data, bins))

    d = dict(A=np.around(Bin_Array,2), B=Frequency[0:len(Bin_Array)])
    df_bin = pd.DataFrame(dict([ (pd.Series(k)) for k in d.items() ]))
    df_bin.columns = ["Bin Array", "Frequency"] 
    plt.title("Frequency Return for the Portfolio")
    plt.bar(df_bin["Bin Array"], df_bin["Frequency"], width=0.2)
    return df_order,df_bin


def DMA_200(df):
    df1 = df.drop(['Date'], axis=1)
    nRows = len(df1)
    nCols = len(df1.columns)
    DMA_200 =  df1.rolling(window=200).mean()
    Change  = df1 - DMA_200
    Percentage = round( (Change/df1)*100, 2)
    df2 = Percentage.fillna(0) 
    df2.insert(loc=0, column='Date', value=df["Date"])
    df3 = pd.DataFrame(df2, columns=df2.columns)
    ax = df2.plot(x="Date", y=df1.columns, kind="line",figsize=(12, 4))
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5)) 
    plt.axhline(0, color='k', linestyle='--',linewidth = 1.0)
    return df2




