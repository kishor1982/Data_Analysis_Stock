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
from scipy.stats import norm
from tabulate import tabulate
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


# https://stackoverflow.com/questions/56987200/sum-along-diagonal-and-anti-diagonal-lines-in-2d-array-numpy-python    
def left_diagonal_sum(a):
    n = len(a)
    N = 2*n-1
    R = np.arange(N)
    r = np.arange(n)
    mask = (r[:,None] <= R) & (r[:,None]+n > R)
    b = np.zeros(mask.shape,dtype=a.dtype)
    b[mask] = a.ravel()
    return b.sum(0)
    
      
# MAXIMUM PAIN    
def Max_Pain(df):   
    df1 = df[["STRIKE PRICE","Call OI", "Put OI"]]
    df1["St_Diff"]  = (df1['STRIKE PRICE'].shift(-1) - df1['STRIKE PRICE'].iloc[1]).fillna(0) 
    CT = np.empty((len(df),len(df)))
    CC = np.array((len(df)))
    for i in range(len(df)):
        for j in range(len(df)):
            CT[i,j] = df1['Call OI'].iloc[j]*df1["St_Diff"].iloc[i]
            CC =  left_diagonal_sum(CT)
    df1["Cumulative Call"] = pd.DataFrame(CC)
    #----------------------------------
    PT = np.empty((len(df),len(df)))
    PC1 = np.array((len(df)))
    PC = np.array((len(df)))

    for i in range(len(df1)):
        for j in range(len(df1)):
            PT[i,j] = df1["Put OI"].iloc[j]*df1["St_Diff"].iloc[i]
            PC1 = left_diagonal_sum(PT[::-1])[::-1]  # making right diagonal sum
            PC = PC1[0:len(df)]          
    df1["Cumulative Put"] = pd.DataFrame(PC[::-1])
    N1 = len(df)-1
    
    df1["Total"] = df1["Cumulative Call"] + df1["Cumulative Put"] 
    df1["Total"].iloc[0]  = df1["Cumulative Call"].iloc[N1] + df1["Cumulative Put"].iloc[0]
    
    fig, ax = plt.subplots(figsize=(7, 3),dpi=400)
    plt.xticks(fontsize=10)
    ax.bar(df1["STRIKE PRICE"], df1["Total"], color='tab:red', alpha=0.75, width=25)   
    
    CS = np.sum(df1["Call OI"])
    PS = np.sum(df1["Put OI"])
    PCR = PS/CS

    if PCR > 1:
        print("More Puts are bought than Calls: Market is extremly Bearish ")
    elif PCR < 0.5:
        print("More Calls are bought than Puts: Market is extremly Bullish ")
    #else:
    #    print(" Market is Regular")
    N = df1['Total'].idxmin()
    St = df1['STRIKE PRICE'].iloc[N]
    print("Strike Price where option Writer meet the Minimum Pain (loss) = ", St, "Rs")
    print("Put Call Ratio (PCR) = ", round(PCR,4))
    
    chg_5p = (St*5)/100
    chg_2p5 = (St*2.5)/100
    chg_3p5 = (St*3.5)/100
    print("\nMinimum Pain Strike Price with 5% correction = ", chg_5p+St, "Rs")
    print("\nMinimum Pain Strike Price with 3.5% correction = ", chg_3p5+St, "Rs")
    print("\nMinimum Pain Strike Price with 2.5% correction = ", chg_2p5+St, "Rs")
    return df1
