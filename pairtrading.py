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
import scipy
import sys
from sklearn.linear_model import LinearRegression
#from pyfinance import ols
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    
    
def Density_Curve_Trade_signal():
    d3 = [
     ["0.16", "-1 SD", "65%"],
     [".025", "-2 SD", "95%"],
     [".003", "-3 SD", "99.7%"] ,
     [".84", "+1 SD", "65%" ], 
     [".974", "+2 SD", "95%"],
     [".997", "+3 SD", "99.7%"]]  
    print(tabulate(d3, headers=["  Density Curve Value", "STD Away", "Prob. reverse to mean"]))
    # Trade set up
    d4 = [
     ["Long", "Between 0.025  & 0.003", "Between 2nd and 3rd", "0.25 or Lower", "0.003 or Higher"],
     ["Short", "Between 0.975 & 0.997", "Between 2nd & 3rd", "0.975 or Lower", "0.997 or Higher"]]  
    print(tabulate(d4, headers=["  Trade Type", "Trigger(Density Curve)", "STD", "Target", "Stoploss"]))
    d5 = [
     ["Long", "Buy A & Sell B"],
     ["----------", "-------------------"],
     ["Short", "Sell A & Buy B"]]  
    print(tabulate(d5, headers=["Position", "Action"]))

    
def Density_Curve(df):
    dfn = df.drop(['Date'], axis=1)
    cp = dfn.diff()
    Spread = cp.iloc[:,0] - cp.iloc[:,1]
    Differential = dfn.iloc[:,0] - dfn.iloc[:,1]
    Ratio = round(dfn.iloc[:,0]/dfn.iloc[:,1],3)
    mean_ratio = np.mean(Ratio)
    std_ratio = np.std(Ratio)
    
    # Density Curve: Cumulative Distribution Function
    Cum_Dist_Fun = scipy.stats.norm.cdf(Ratio, mean_ratio, std_ratio)
    d = dict(A=df["Date"], B=dfn.iloc[:,0], C=cp.iloc[:,0], D=dfn.iloc[:,1], E=cp.iloc[:,1], F=Spread, G=Differential, H=Ratio, I=Cum_Dist_Fun)
    df_pair = pd.DataFrame(dict([ (pd.Series(k)) for k in d.items() ]))
    df_pair.columns = ["Date","Close1", "Close_diff1", "Close2", "Close_diff2", "Spread", "Differential","Ratio","Density Curve(Ratio)"] 
    df_pair = df_pair.fillna(0)
    
    mean_sp = df_pair["Spread"].mean()
    mean_diff = df_pair["Differential"].mean()
    mean_r = df_pair["Ratio"].mean()

    med_sp = df_pair["Spread"].median()
    med_diff = df_pair["Differential"].median()
    med_r = df_pair["Ratio"].median()


    mode_sp = df_pair.mode()['Spread'][0]
    mode_diff = df_pair.mode()['Differential'][0] 
    mode_r = df_pair.mode()['Ratio'][0]

# Standard Deviation
    std_sp =  df_pair["Spread"].std()
    std_diff =  df_pair["Differential"].std()
    std_r =  df_pair["Ratio"].std()

# Absolute Deviation
    absd_sp =  df_pair["Spread"].mad()
    absd_diff =  df_pair["Differential"].mad()
    absd_r =  df_pair["Ratio"].mad()
    d1 = [
     ["Mean", mean_sp, mean_diff, mean_r],
     ["Median", med_sp, med_diff, med_r],
     ["Mode", mode_sp, mode_diff, mode_r] ,
     ["Standard Dev.",std_sp,std_diff,std_r ], 
     ["Abs. Dev.", absd_sp, absd_diff, absd_r]]  
    print(tabulate(d1, headers=["      ", "Spread", "Differential", "Ratio"]))   
    print("------------------------------------------------------")
    spread_1st_std = mean_sp + std_sp
    spread_2nd_std = mean_sp + (2*std_sp)
    spread_3rd_std = mean_sp + (3*std_sp)

    spread_1st_std_below = mean_sp - std_sp
    spread_2nd_std_below = mean_sp - (2*std_sp)
    spread_3rd_std_below = mean_sp - (3*std_sp)


    diff_1st_std = mean_diff + std_diff
    diff_2nd_std = mean_diff + (2*std_diff)
    diff_3rd_std = mean_diff + (3*std_diff)

    diff_1st_std_below = mean_diff - std_diff
    diff_2nd_std_below = mean_diff - (2*std_diff)
    diff_3rd_std_below = mean_diff - (3*std_diff)


    mode_1st_std = mean_r + std_r
    mode_2nd_std = mean_r + (2*std_r)
    mode_3rd_std = mean_r + (3*std_r)

    mode_1st_std_below = mean_r - std_r
    mode_2nd_std_below = mean_r - (2*std_r)
    mode_3rd_std_below = mean_r - (3*std_r)

    d2 = [
     ["3", spread_3rd_std, diff_3rd_std, mode_3rd_std],
     ["2", spread_2nd_std, diff_2nd_std, mode_2nd_std],
     ["1", spread_1st_std, diff_1st_std, mode_1st_std] ,
     ["Mean",mean_sp,mean_diff,mean_r ], 
     ["-1", spread_1st_std_below, diff_1st_std_below, mode_1st_std_below],
     ["-2", spread_2nd_std_below, diff_2nd_std_below, mode_2nd_std_below],
     ["-3", spread_3rd_std_below, diff_3rd_std_below, mode_3rd_std_below]]  
    print(tabulate(d2, headers=["  STD    ", "Spread", "Differential", "Ratio"]))
    
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    years_fmt = mdates.DateFormatter('%Y')

    fig, ax = plt.subplots(figsize=(11, 12), dpi=400)
    gs = gridspec.GridSpec(4, 1)   
    ax0 = plt.subplot(gs[0])
    ax0.plot(df["Date"], df_pair["Spread"], linestyle='-', color='blue', alpha=0.7, label='Spread', linewidth = 1.5)
    plt.axhline(y = mean_sp, color = 'g', linestyle = '-')
    ax0.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    plt.xticks(fontsize=10)
#ax0.xaxis.set_major_locator(months)
    ylabel('Spread (Rs)')
    for tick in ax0.get_xticklabels():
        tick.set_rotation(90)
    ax1 = plt.subplot(gs[1])
    ax1.plot(df["Date"], df_pair["Differential"], linestyle='-', color='blue', alpha=0.7, label='Differential', linewidth = 1.5)
    plt.axhline(y = mean_diff, color = 'g', linestyle = '-')
    ax1.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    plt.xticks(fontsize=10)
    ylabel('Differential (Rs)')
    for tick in ax1.get_xticklabels():
        tick.set_rotation(90)
    
    ax2 = plt.subplot(gs[2])
    ax2.plot(df["Date"], df_pair["Ratio"], linestyle='-', color='blue', alpha=0.7, label='Ratio', linewidth = 1.5)
    plt.axhline(y = mean_r, color = 'g', linestyle = '-')
    ax2.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ylabel('Ratio')
    for tick in ax2.get_xticklabels():
        tick.set_rotation(90)  
    ax3 = plt.subplot(gs[3])
    ax3.plot(df["Date"], df_pair["Density Curve(Ratio)"], linestyle='-', color='blue', alpha=0.7, label='Ratio', linewidth = 1.5)
    ax3.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ylabel('Density Curve(Ratio)')
    for tick in ax3.get_xticklabels():
        tick.set_rotation(90)  
    fig.tight_layout()
    
    return df_pair


'''
def Relative_X(df):
    x = df.iloc[:,1]
    y = df.iloc[:,2]
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x,y)

    d1 = [
     ["Slope", slope],
     ["Intercept ", intercept],
     ["Multiple R ", r_value],
     [" R Square ", r_value**2],
     [" P-Value  ", p_value],
     ["Std_Err ", std_err] ]  
    print(tabulate(d1, headers=["Regression Statistics"]))
    
    x = np.array(x).reshape(-1,1)
    model = LinearRegression().fit(x, y)

    y_pred = model.intercept_ + model.coef_ * x


    res1 = np.array(y) - y_pred
    residuals1 = np.diagonal(res1)
    y_pred =y_pred.reshape(len(residuals1,))  

    ma = np.array([y_pred,residuals1])

    d2 = dict(A=y_pred, B=residuals1)
    df1 = pd.DataFrame(dict([(pd.Series(k)) for k in d2.items() ]))
    df1.columns = ["Predicted Y",  "Residuals1"] 
    df1 = df1.fillna(0)
    print(" Standard Error = ", df1["Residuals1"].std())

    model_new = ols.OLS(y=y, x=x)
    std_err_intercept = model_new.se_alpha
    print(" Standard Error of Intercept = ", std_err_intercept)
    print("Error Ratio = ", std_err_intercept/df1["Residuals1"].std())
    return df1


def Relative_Y(df):
    y = df.iloc[:,1]
    x = df.iloc[:,2]
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x,y)

    d1 = [
     ["Slope", slope],
     ["Intercept ", intercept],
     ["Multiple R ", r_value],
     [" R Square ", r_value**2],
     [" P-Value  ", p_value],
     ["Std_Err ", std_err] ]  
    print(tabulate(d1, headers=["Regression Statistics"]))
    
    x = np.array(x).reshape(-1,1)
    model = LinearRegression().fit(x, y)

    y_pred = model.intercept_ + model.coef_ * x


    res1 = np.array(y) - y_pred
    residuals1 = np.diagonal(res1)
    y_pred =y_pred.reshape(len(residuals1,))  

    ma = np.array([y_pred,residuals1])

    d2 = dict(A=y_pred, B=residuals1)
    df1 = pd.DataFrame(dict([(pd.Series(k)) for k in d2.items() ]))
    df1.columns = ["Predicted X",  "Residuals1"] 
    df1 = df1.fillna(0)
    print(" Standard Error = ", df1["Residuals1"].std())

    model_new = OLS(y=y, x=x)
    std_err_intercept = model_new.se_alpha
    print(" Standard Error of Intercept = ", std_err_intercept)
    print("Error Ratio = ", std_err_intercept/df1["Residuals1"].std())
    return df1

'''