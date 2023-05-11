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
    
#---------------- OPTION PAY OFF ----------------    
# Long_Call Option Pay Off
def Long_Call(MEP,SR,Premium_Paid):
    LSIV = np.where(MEP > SR, MEP - SR, 0)

# PREMIUM PAID
    PP = [-Premium_Paid]*len(MEP)
    PP1 = [Premium_Paid]*len(MEP)

# PAY OFF
    LSPO = LSIV-PP1 
# Break Even 
    BE = SR + Premium_Paid
    d = dict(A=MEP, B=LSIV, C=PP, D=LSPO)
    df = pd.DataFrame(dict([ (pd.Series(k)) for k in d.items() ]))
    df.columns = ["Market Expiry","Long Call SR-IV", "PP", "LC Payoff"] 
    print("Break Even (Rs) = ", BE)
    fig, ax = plt.subplots(figsize=(7, 3),dpi=400)
    plt.title('Long Call Pay Off', fontsize=12)
    plt.xticks(fontsize=10)
    ax.plot(df["Market Expiry"], df["LC Payoff"], linestyle='-', color='tab:green', alpha=0.7, linewidth=1.5)
    plt.axhline(0, color='k', linestyle='-',linewidth = 0.5)
    plt.axvline(BE, color='tab:red', linestyle='-',linewidth = 1)
    plt.grid(b=None, which='major', axis='both')
    ax.set_xlabel("Market Expiry Price (Rs)")
    ax.set_ylabel("Profit/Loss (Rs)")
    xloc = MultipleLocator(50)
    yloc = MultipleLocator(20)
    ax.xaxis.set_minor_locator(xloc)
    ax.yaxis.set_minor_locator(yloc)
    return df
#---------------------------------------------------------
# Short Call Option Pay Off    
def Short_Call(MEP,SR,Premium_Received):
    LSIV = np.where(MEP > SR, MEP - SR, 0)

# PREMIUM PAID
    PR = [Premium_Received]*len(MEP)

# PAY OFF
    LSPO = PR - LSIV 
# Break Even 
    BE = SR + Premium_Received
    d = dict(A=MEP, B=LSIV, C=PR, D=LSPO)
    df = pd.DataFrame(dict([ (pd.Series(k)) for k in d.items() ]))
    df.columns = ["Market Expiry","Short Call SR-IV", "PP", "Short Call Payoff"] 
    
    print("Break Even (Rs) = ", BE)
    fig, ax = plt.subplots(figsize=(7, 3),dpi=400)
    plt.title('Short Call Pay Off', fontsize=12)
    plt.xticks(fontsize=10)
    ax.plot(df["Market Expiry"], df["Short Call Payoff"], linestyle='-', color='tab:green', alpha=0.7, linewidth=1.5)
    plt.axhline(0, color='k', linestyle='-',linewidth = 0.5)
    plt.axvline(BE, color='tab:red', linestyle='-',linewidth = 1)
    plt.grid(b=None, which='major', axis='both')
    ax.set_xlabel("Market Expiry Price (Rs)")
    ax.set_ylabel("Profit/Loss (Rs)")
    xloc = MultipleLocator(50)
    yloc = MultipleLocator(20)
    ax.xaxis.set_minor_locator(xloc)
    ax.yaxis.set_minor_locator(yloc)
    
    return df    
#---------------------------------------------------------
# Long Put Pay off
def Long_Put(MEP,SR,Premium_Paid):
    LSIV = np.where(MEP < SR, SR-MEP , 0)

# PREMIUM PAID
    PP = [-Premium_Paid]*len(MEP)
    PP1 = [Premium_Paid]*len(MEP)

# PAY OFF
    LSPO = LSIV-PP1 
# Break Even 
    BE = SR - Premium_Paid
    d = dict(A=MEP, B=LSIV, C=PP, D=LSPO)
    df = pd.DataFrame(dict([ (pd.Series(k)) for k in d.items() ]))
    df.columns = ["Market Expiry","Long Put SR-IV", "PP", "Long Put Payoff"] 
    print("Break Even (Rs) = ", BE)
    fig, ax = plt.subplots(figsize=(7, 3),dpi=400)
    plt.title('Long Put Pay Off', fontsize=12)
    plt.xticks(fontsize=10)
    ax.plot(df["Market Expiry"], df["Long Put Payoff"], linestyle='-', color='tab:green', alpha=0.7, linewidth=1.5)
    plt.axhline(0, color='k', linestyle='-',linewidth = 0.5)
    plt.axvline(BE, color='tab:red', linestyle='-',linewidth = 1)
    plt.grid(b=None, which='major', axis='both')
    ax.set_xlabel("Market Expiry Price (Rs)")
    ax.set_ylabel("Profit/Loss (Rs)")
    xloc = MultipleLocator(50)
    yloc = MultipleLocator(20)
    ax.xaxis.set_minor_locator(xloc)
    ax.yaxis.set_minor_locator(yloc)
    
    return df
#---------------------------------------------------------
# Short Put Pay off  
def Short_Put(MEP,SR,Premium_Received):
# Long call intrinsic
    LSIV = np.where(MEP < SR, SR-MEP, 0)

# PREMIUM PAID
    PR = [Premium_Received]*len(MEP)

# PAY OFF
    LSPO = PR - LSIV 
# Break Even 
    BE = SR - Premium_Received
    d = dict(A=MEP, B=LSIV, C=PR, D=LSPO)
    df = pd.DataFrame(dict([ (pd.Series(k)) for k in d.items() ]))
    df.columns = ["Market Expiry","Short Put SR-IV", "PP", "Short Put Payoff"] 
    print("Break Even (Rs) = ", BE)
    fig, ax = plt.subplots(figsize=(7, 3),dpi=400)
    plt.title('Short Put Pay Off', fontsize=12)
    plt.xticks(fontsize=10)
    ax.plot(df["Market Expiry"], df["Short Put Payoff"], linestyle='-', color='tab:green', alpha=0.7, linewidth=1.5)
    plt.axhline(0, color='k', linestyle='-',linewidth = 0.5)
    plt.axvline(BE, color='tab:red', linestyle='-',linewidth = 1)
    plt.grid(b=None, which='major', axis='both')
    ax.set_xlabel("Market Expiry Price (Rs)")
    ax.set_ylabel("Profit/Loss (Rs)")
    xloc = MultipleLocator(50)
    yloc = MultipleLocator(20)
    ax.xaxis.set_minor_locator(xloc)
    ax.yaxis.set_minor_locator(yloc)
    
    return df
#---------------------------------------------------------
    
def daily_avg(df):
    df['Daily Rt'] = (np.log( df['Close']/df['Close'].shift(1)))*100
    Daily_Average = df['Daily Rt'].mean()
    Daily_SD = np.std(df['Daily Rt'])
    print("Daily Average =", df['Daily Rt'].mean())
    print()
    print("Daily SD = ", Daily_SD,"%")
    print()
    
def avg_return(df, price, period):
    #CMP = df['Close'].iloc[-1]
    df['Daily Rt'] = (np.log( df['Close']/df['Close'].shift(1)))*100
    Daily_Average = df['Daily Rt'].mean()
    Daily_SD = np.std(df['Daily Rt'])
    
    period_avg = (Daily_Average*period)
    period_SD = (Daily_SD*np.sqrt(period))
   
    range_up = (period_avg+period_SD)/100
    range_low = abs((period_avg-period_SD))/100
    price_up = price*(1.0+range_up )
    price_low = price*(1.0-range_low )
    print(period, "Days Average = ",period_avg ,"%")
    print()
    print(period, "Days SD = ",period_SD ,"%")
    print()
    print(period, "days Up range = ", range_up*100, "%")
    print()
    print(period, "days Low range = ", range_low*100 ,"%")
    print()
    print("Upper Price for",period, "days range = ", price_up)
    print()
    print("Lower Price for",period, "days range = ", price_low)
    print()
    print("-----------Suggestions-----------:")
    print("Sell Call Option Above",price_up )
    print()
    print("Sell Put Option Below",price_low )
    print("----------------------------------:")

def sd_price(df, lm):
    if len(df) == 252:
        l = 252
        dfn = df
    elif len(df) > 252:
        dfn = df.iloc[-252:]
        l = len(df)
    else:
        dfn = df
        l = len(df)
        
    Spot_price = dfn['Close'].iloc[-1]
    dfn['Close'] = dfn['Close'].tail(l)
    dfn['log returns'] = (np.log( dfn['Close']/dfn['Close'].shift(1)))
#
    Daily_avg = abs(np.mean(((dfn['Close'] - dfn['Close'].shift(1))/dfn['Close'].shift(1))*100))
    Daily_std = round(np.std(dfn['log returns']), 4)*100
#
# annual numbers
    yearly_avg = Daily_avg*l
    yearly_std = Daily_std*np.sqrt(l)
    Up_range_1SD = yearly_avg + yearly_std 
    Low_range_1SD = yearly_avg - yearly_std 

    UP_1SD = Spot_price*(1+(Up_range_1SD/100))
    Low_1SD = Spot_price*(1-(Low_range_1SD/100))
#
    Up_range_2SD = yearly_avg + 2*yearly_std 
    Low_range_2SD = yearly_avg - 2*yearly_std
#
    UP_2SD = Spot_price*(1+(Up_range_2SD/100))
    Low_2SD = Spot_price*(1-(Low_range_2SD/100))
#
# Monthly numbers
    #lm = 30
    Mon_Average = Daily_avg*lm
    Mon_Std_dev = Daily_std*np.sqrt(lm)
#
    Mon_Up_range_1SD = Mon_Average + Mon_Std_dev 
    Mon_Low_range_1SD = Mon_Average - Mon_Std_dev
    Mon_UP_1SD = Spot_price*(1+ (Mon_Up_range_1SD/100)) 
    Mon_Low_1SD = Spot_price*(1-(Mon_Low_range_1SD/100)) 
#
    Mon_Up_range_2SD = Mon_Average + 2*Mon_Std_dev 
    Mon_Low_range_2SD = Mon_Average - 2*Mon_Std_dev
    Mon_UP_2SD = Spot_price*(1+ (Mon_Up_range_2SD/100)) 
    Mon_Low_2SD = Spot_price*(1-(Mon_Low_range_2SD/100)) 

# annual Probability 
    print("Number of days = " ,l)
    print("Daily Average/Mean    = ", round(abs(Daily_avg),4), "%")
    print("Daily Std./Volatility = ", round(Daily_std,4), "%")
    print("Annual Std./Volatility = ", round(yearly_std,4), "%")

    print("Current Market Price = ", round(Spot_price,2), "Rs")
    print()
    print("1SD: 68% Probability in a year")
    print("1Yr UP from 1SD   = ",round(UP_1SD,2), "Rs")
    print("1Yr Low from 1SD   = ", round(Low_1SD,2), "Rs"  )
    print()
    print("2SD: 95% Probability in a year")
    print("1Yr UP from 2SD    = ",round(UP_2SD,2), "Rs")
    print("1Yr Low from 2SD   = ", round(Low_2SD,2), "Rs" )
    print()
    print("1SD: 68% Probability in ", lm, "Days")
    print(lm ," days UP from 1SD   = ",round(Mon_UP_1SD,2), "Rs")
    print(lm ," days Low from 1SD   = ", round(Mon_Low_1SD,2), "Rs" )
    print()
    print("2SD: 95% Probability in ", lm, "Days")
    print(lm ," days UP from 2SD    = ",round(Mon_UP_2SD,2), "Rs")
    print(lm ," days Low from 2SD   = ", round(Mon_Low_2SD,2), "Rs" )
#
def vol_hist(df):
    df['Daily Rt'] = (np.log( df['Close']/df['Close'].shift(1)))*100
    daily_std = np.std(np.log( df['Close']/df['Close'].shift(1)))
    str_vol = str(round(daily_std, 4)*100)
    #df['20 day Volatility'] = 100*df['Daily Rt'].rolling(window=20).std() 
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    years_fmt = mdates.DateFormatter('%Y')
    fig = plt.subplots(figsize=(5,4), dpi=400)
    gs = gridspec.GridSpec(1, 1)       
    ax0 = plt.subplot(gs[0])
    df['Daily Rt'].hist(ax=ax0, bins=26, alpha=0.75, color='blue')
    ax0.set_title('volatility:'  + str_vol + '%')
    ax0.set_xlabel('Log Daily return')
    ax0.set_ylabel('Freq of log return')
    #plt.tight_layout()   
#    
def bs_call(S0,X,t,σ,r,q,td):
    σ,r,q,t = σ/100,r/100, q/100,t/td
  #https://unofficed.com/black-scholes-model-options-calculator-google-sheet/

    d1 = (np.log(S0/X)+(r-q+0.5*σ**2)*t)/(σ*np.sqrt(t))
  #stackoverflow.com/questions/34258537/python-typeerror-unsupported-operand-types-for-float-and-int

  #stackoverflow.com/questions/809362/how-to-calculate-cumulative-normal-distribution
    Nd1 = (np.exp((-d1**2)/2))/np.sqrt(2*np.pi)
    d2 = d1-σ*np.sqrt(t)
    Nd2 = norm.cdf(d2)
    call_theta =(-((S0*σ*np.exp(-q*t))/(2*np.sqrt(t))*
                (1/(np.sqrt(2*np.pi)))*np.exp(-(d1*d1)/2))-
                (r*X*np.exp(-r*t)*norm.cdf(d2))+(q*np.exp(-q*t)*S0*norm.cdf(d1)))/td
    call_premium =np.exp(-q*t)*S0*norm.cdf(d1)-X*np.exp(-r*t)*norm.cdf(d1-σ*np.sqrt(t))
    call_delta =np.exp(-q*t)*norm.cdf(d1)
    gamma =(np.exp(-r*t)/(S0*σ*np.sqrt(t)))*(1/(np.sqrt(2*np.pi)))*np.exp(-(d1*d1)/2)
    vega = ((1/100)*S0*np.exp(-r*t)*np.sqrt(t))*(1/(np.sqrt(2*np.pi))*np.exp(-(d1*d1)/2))
    call_rho =(1/100)*X*t*np.exp(-r*t)*norm.cdf(d2)
    print("Call Θ = ", call_theta)
    print("Call premium = " , call_premium)
    print("Call δ =",  call_delta)
    print("Call Gamma, γ =" , gamma)
    print("Call Vega, ν = ",  vega)
    print("Call ρ = ",  call_rho)
    return call_theta,call_premium,call_delta,gamma,vega,call_rho
#
def bs_put(S0,X,t,σ,r,q,td):
    σ,r,q,t = σ/100,r/100, q/100,t/td
  #https://unofficed.com/black-scholes-model-options-calculator-google-sheet/

    d1 = (np.log(S0/X)+(r-q+0.5*σ**2)*t)/(σ*np.sqrt(t))
  #stackoverflow.com/questions/34258537/python-typeerror-unsupported-operand-types-for-float-and-int

  #stackoverflow.com/questions/809362/how-to-calculate-cumulative-normal-distribution
    Nd1 = (np.exp((-d1**2)/2))/np.sqrt(2*np.pi)
    d2 = d1-σ*np.sqrt(t)
    Nd2 = norm.cdf(d2)
    
    put_theta =(-((S0*σ*np.exp(-q*t))/(2*np.sqrt(t))*
            (1/(np.sqrt(2*np.pi)))*np.exp(-(d1*d1)/2))
            +(r*X*np.exp(-r*t)*norm.cdf(-d2))-(q*np.exp(-q*t)*S0*norm.cdf(-d1)))/td
    put_premium =X*np.exp(-r*t)*norm.cdf(-d2)-np.exp(-q*t)*S0*norm.cdf(-d1)
    put_delta =np.exp(-q*t)*(norm.cdf(d1)-1)
    gamma =(np.exp(-r*t)/(S0*σ*np.sqrt(t)))*(1/(np.sqrt(2*np.pi)))*np.exp(-(d1*d1)/2)
    vega = ((1/100)*S0*np.exp(-r*t)*np.sqrt(t))*(1/(np.sqrt(2*np.pi))*np.exp(-(d1*d1)/2))
    put_rho =(-1/100)*X*t*np.exp(-r*t)*norm.cdf(-d2)
    print("Put Θ = ",  put_theta)
    print("Put premium = ", put_premium)
    print("Put δ =", put_delta)
    print("Put Gamma, γ =" , gamma)
    print("Put Vega, ν = ",  vega)
    print("Put ρ = ", put_rho)
    return put_theta,put_premium,put_delta,gamma,vega,put_rho
#
def bs_call_put(S0,X,t,σ,r,q,td):
    σ,r,q,t = σ/100,r/100, q/100,t/td
  #https://unofficed.com/black-scholes-model-options-calculator-google-sheet/

    d1 = (np.log(S0/X)+(r-q+0.5*σ**2)*t)/(σ*np.sqrt(t))
  #stackoverflow.com/questions/34258537/python-typeerror-unsupported-operand-types-for-float-and-int
  #stackoverflow.com/questions/809362/how-to-calculate-cumulative-normal-distribution
    Nd1 = (np.exp((-d1**2)/2))/np.sqrt(2*np.pi)
    d2 = d1-σ*np.sqrt(t)
    Nd2 = norm.cdf(d2)
    call_theta =(-((S0*σ*np.exp(-q*t))/(2*np.sqrt(t))*(1/(np.sqrt(2*np.pi)))*np.exp(-(d1*d1)/2))-(r*X*np.exp(-r*t)*norm.cdf(d2))+(q*np.exp(-q*t)*S0*norm.cdf(d1)))/td
    put_theta =(-((S0*σ*np.exp(-q*t))/(2*np.sqrt(t))*(1/(np.sqrt(2*np.pi)))*np.exp(-(d1*d1)/2))+(r*X*np.exp(-r*t)*norm.cdf(-d2))-(q*np.exp(-q*t)*S0*norm.cdf(-d1)))/td
    call_premium =np.exp(-q*t)*S0*norm.cdf(d1)-X*np.exp(-r*t)*norm.cdf(d1-σ*np.sqrt(t))
    put_premium =X*np.exp(-r*t)*norm.cdf(-d2)-np.exp(-q*t)*S0*norm.cdf(-d1)
    call_delta =np.exp(-q*t)*norm.cdf(d1)
    put_delta =np.exp(-q*t)*(norm.cdf(d1)-1)
    gamma =(np.exp(-r*t)/(S0*σ*np.sqrt(t)))*(1/(np.sqrt(2*np.pi)))*np.exp(-(d1*d1)/2)
    vega = ((1/100)*S0*np.exp(-r*t)*np.sqrt(t))*(1/(np.sqrt(2*np.pi))*np.exp(-(d1*d1)/2))
    call_rho =(1/100)*X*t*np.exp(-r*t)*norm.cdf(d2)
    put_rho =(-1/100)*X*t*np.exp(-r*t)*norm.cdf(-d2)
    
    d = [
     ["Premium (Rs)", call_premium, put_premium],
     ["Delta (δ)", call_delta, put_delta],
     ["Gamma (γ )", gamma, gamma] ,
     ["Theta (Θ)",call_theta,put_theta ], 
     ["Vega (ν)", vega, vega], 
     ["Rho (ρ)",call_rho, put_rho ]]  
    print(tabulate(d, headers=["OPTION/Greek", "CALL", "PUT"]))
    
    print("\nReport Based on Greeks:")
    
    if 0.8 <= call_delta <=1.0:
        print("Call Option: Deep ITM")     
    elif 0.6 <= call_delta <=1.0:
         print("Call Option: Slightly ITM")
            
    elif 0.45 <= call_delta <=0.55:        
         print("Call Option:  ATM")
            
    elif 0.45 <= call_delta <=0.3:    
         print("Call Option:  Slightly OTM")
            
    elif 0.3 <= call_delta >=0.0: 
        print("Call Option:  Deep OTM")
    #----------------------------
    
    if -0.8 <= put_delta <=-1.0:
        print("Put Option: Deep ITM")     
    elif -0.6 <= put_delta <=-1.0:
         print("Put Option: Slightly ITM")
            
    elif -0.45 <= put_delta <=-0.55:        
         print("Put Option:  ATM")
            
    elif -0.45 <= put_delta <=-0.3:    
         print("Put Option:  Slightly OTM")
            
    elif -0.3 <= put_delta >=0.0: 
        print("Put Option:  Deep OTM")
        
    print("\nBuying options in the take off stage (δ=0.25-0.35) tends to give high % return")
    
    ''' print("Call premium = " , call_premium)
    print("Put premium = ", put_premium)
    print("-------------------------------------")
    print("Call δ =",  call_delta)
    print("Put δ =", put_delta)
    print("-------------------------------------")
    print("Gamma, γ =" , gamma)
    print("-------------------------------------")
    print("Call Θ = ", call_theta)
    print("Put Θ = ",  put_theta)
    print("-------------------------------------")
    print("Vega, ν = ",  vega)
    print("Rho is the rate at which the price of a derivative changes relative to a change in the risk-free rate of interest")
    print("Call ρ = ",  call_rho)
    print("Put ρ = ", put_rho)
    return call_theta,put_theta,call_premium,put_premium,call_delta,put_delta,gamma,vega,call_rho,put_rho'''
    
    
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
    df1 = df[["STRIKE PRICE\n","Call OI\n", "Put OI\n"]]
    df1["St_Diff"]  = (df1['STRIKE PRICE\n'].shift(-1) - df1['STRIKE PRICE\n'].iloc[1]).fillna(0) 
    CT = np.empty((len(df),len(df)))
    CC = np.array((len(df)))
    for i in range(len(df)):
        for j in range(len(df)):
            CT[i,j] = df1['Call OI\n'].iloc[j]*df1["St_Diff"].iloc[i]
            CC =  left_diagonal_sum(CT)
    df1["Cumulative Call"] = pd.DataFrame(CC)
    #----------------------------------
    PT = np.empty((len(df),len(df)))
    PC1 = np.array((len(df)))
    PC = np.array((len(df)))

    for i in range(len(df1)):
        for j in range(len(df1)):
            PT[i,j] = df1["Put OI\n"].iloc[j]*df1["St_Diff"].iloc[i]
            PC1 = left_diagonal_sum(PT[::-1])[::-1]  # making right diagonal sum
            PC = PC1[0:len(df)]          
    df1["Cumulative Put"] = pd.DataFrame(PC[::-1])
    N1 = len(df)-1
    
    df1["Total"] = df1["Cumulative Call"] + df1["Cumulative Put"] 
    df1["Total"].iloc[0]  = df1["Cumulative Call"].iloc[N1] + df1["Cumulative Put"].iloc[0]
    
    fig, ax = plt.subplots(figsize=(7, 3),dpi=400)
    plt.xticks(fontsize=10)
    ax.bar(df1["STRIKE PRICE\n"], df1["Total"], color='tab:red', alpha=0.75, width=25)   
    
    CS = np.sum(df1["Call OI\n"])
    PS = np.sum(df1["Put OI\n"])
    PCR = PS/CS
    if PCR > 1:
        print("More Puts are bought than Calls: Market is extremly Bearish ")
    elif 0.1 <= PCR <= 0.5:
        print("More Calls are bought than Puts: Market is extremly Bullish ")
    else:
        print(" Market is Regular")
    N = df1['Total'].idxmin()
    St = df1['STRIKE PRICE\n'].iloc[N]
    print("Strike Price where option Writer meet the Minimum Pain (loss) = ", St, "Rs")
    print("Put Call Ratio (PCR) = ", round(PCR,4))
    
    chg_5p = (St*5)/100
    chg_2p5 = (St*2.5)/100
    chg_3p5 = (St*3.5)/100
    print("\nMinimum Pain Strike Price with 5% correction = ", chg_5p+St, "Rs")
    print("\nMinimum Pain Strike Price with 3.5% correction = ", chg_3p5+St, "Rs")
    print("\nMinimum Pain Strike Price with 2.5% correction = ", chg_2p5+St, "Rs")
    return df1

