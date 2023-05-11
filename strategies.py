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
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

#########################################################################################
def Bull_Call(MEP,LST,HST,Premium_Paid,Premium_Received):
#------------------------------------
    Net_debt = Premium_Paid - Premium_Received
#------------------------------------
    LSIV = np.where(MEP > LST, MEP - LST, 0)

# PREMIUM PAID
    PP = [-Premium_Paid]*len(MEP)
    PP1 = [Premium_Paid]*len(MEP)

# LOWER STRIKE - PAY OFF
    LSPO = LSIV-PP1 

# Higher STRIKE - Intrinsic Value
    HSIV = np.where(MEP > HST, MEP-HST, 0)
    PR = [Premium_Received ]*len(MEP)

# HS - Pay off
    HSPO = PR-HSIV 

# STRATEGY PAY OFF
    SGPO = LSPO + HSPO

# Break Even
    BE = LST+Net_debt
# Maximum Profit
    Max_Profit = HST-LST-Net_debt 
    
    d = dict(A=MEP, B=LSIV, C=PP, D=LSPO, E=HSIV, F=PR, G=HSPO, H = SGPO)
    df = pd.DataFrame(dict([ (pd.Series(k)) for k in d.items() ]))
    df.columns = ["Market Expiry","LS-IV", "PP", "LS Payoff", "HS-IV", "PR","HS Payoff", "Strategy Pay off"] 
    
    print("Break even (Rs) =", BE)
    print("\nMaximum Profit (Rs) =",Max_Profit )
    print("\nMaximum Loss (Rs) =", Net_debt)
    print("\n============================================================================")
    print("Strategy Report:")
    print("============================================================================")
    print("Market Expectation : Moderately Bullish")
    print("\n1) Strategy makes loss if price goes below ", LST,"Rs, Loss is restricted to", Net_debt,"Rs")
    print("\n2) Break Even (No profit or Loss) when price goes ", BE,"Rs")
    print("\n3) Makes money if price goes above ", BE,"Rs and maximum profit is", Max_Profit,"Rs" )
    print("============================================================================")

    fig, ax = plt.subplots(figsize=(7, 3),dpi=400)
    plt.title('Bull Call Spread Pay Off', fontsize=12)
    plt.xticks(fontsize=10)
    ax.plot(df["Market Expiry"], df["Strategy Pay off"], linestyle='-', color='tab:green', alpha=0.7, linewidth=1.5)
    plt.axhline(0, color='k', linestyle='-',linewidth = 0.5)
    plt.axvline(LST+Net_debt, color='tab:red', linestyle='-',linewidth = 1)
    plt.grid(b=None, which='major', axis='both')
    ax.set_xlabel("Market Expiry Price (Rs)")
    ax.set_ylabel("Profit/Loss (Rs)")
    xloc = MultipleLocator(50)
    yloc = MultipleLocator(20)
    ax.xaxis.set_minor_locator(xloc)
    ax.yaxis.set_minor_locator(yloc)
    return df

#########################################################################################
def Bull_Put(MEP,LST,HST,Premium_Paid,Premium_Received):
    Net_credit = Premium_Received - Premium_Paid
# DIFFERENCE MEP - SP
    LSIV = np.where(MEP < LST, LST-MEP, 0)
# PREMIUM PAID
    PP = [-Premium_Paid]*len(MEP)
    PP1 = [Premium_Paid]*len(MEP)
# LOWER STRIKE - PAY OFF
    LSPO = LSIV-PP1
# Higher STRIKE - Intrinsic Value
    HSIV = np.where(MEP < HST, HST-MEP, 0)
    PR = [Premium_Received ]*len(MEP)
# HS - Pay off
    HSPO = PR-HSIV
# STRATEGY PAY OFF
    SGPO = LSPO + HSPO
# Break Even
    BE = HST-Net_credit
#Maximum Loss/profit
    Max_Loss = HST-LST-Net_credit
    Max_profit = Net_credit
    print("Break even (Rs) =", BE)
    print("\nMaximum Loss (Rs) =", -Max_Loss)
    print("\nNet Credit & Maximum Profit (Rs) =", Max_profit)
    
    d = dict(A=MEP, B=LSIV, C=PP, D=LSPO, E=HSIV, F=PR, G=HSPO, H = SGPO)
    df = pd.DataFrame(dict([ (pd.Series(k)) for k in d.items() ]))
    df.columns = ["Market Expiry","LS-IV", "PP", "LS Payoff", "HS-IV", "PR","HS Payoff", "Strategy Pay off"] 
#    
    print("============================================================================")
    print("Strategy Report: Market Expectation : Moderately Bullish")
    print("============================================================================")
    print("Market Expectation : Moderately Bullish")
    print("\n1) Strategy makes loss if price goes below ", LST,"Rs, Loss is restricted to", Max_Loss,"Rs")
    print("\n2) Break Even (No profit or Loss) when price goes ", BE,"Rs")
    print("\n3) Makes money if price goes above ", BE,"Rs and maximum profit is", Net_credit,"Rs" )
    print("============================================================================")
#
    fig, ax = plt.subplots(figsize=(7, 3),dpi=400)
    plt.title('Bull Put Spread Pay Off', fontsize=12)
    plt.xticks(fontsize=10)
    ax.plot(df["Market Expiry"], df["Strategy Pay off"], linestyle='-', color='tab:green', alpha=0.7, linewidth=1.5)
    plt.axhline(0, color='k', linestyle='-',linewidth = 0.5)
    plt.axvline(HST-Net_credit, color='tab:red', linestyle='-',linewidth = 1)
    plt.grid(b=None, which='major', axis='both')
    ax.set_xlabel("Market Expiry Price (Rs)")
    ax.set_ylabel("Profit/Loss (Rs)")
    xloc = MultipleLocator(50)
    yloc = MultipleLocator(20)
    ax.xaxis.set_minor_locator(xloc)
    ax.yaxis.set_minor_locator(yloc)  
    return df

#########################################################################################
def Call_Ratio_Back(MEP,LST,HST,Premium_Paid1,Premium_Paid2, Premium_Received):
    Spread = HST - LST
#------------------------------------
    Premium_Paid =  Premium_Paid1 + Premium_Paid2
#------------------------------------
    Net_credit = Premium_Received - Premium_Paid
#------------------------------------
    LSIV = np.where(MEP > LST, MEP - LST, 0)
# PREMIUM PAID and Received list
    PP = [-Premium_Paid]*len(MEP)
    PP1 = [Premium_Paid]*len(MEP)
    PR = [Premium_Received ]*len(MEP)

# LOWER STRIKE - PAY OFF
    LSPO = PR - LSIV

# Higher STRIKE - Intrinsic Value
    HSIV = np.where(MEP > HST, MEP-HST, 0)

# Higher strike Pay off -> 2 Long Positions
    HSPO = (2*HSIV-PP1 )

# STRATEGY PAY OFF
    SGPO = LSPO + HSPO

# Maximum Loss
    Max_Loss = Spread - Net_credit
# Break Even
    Low_BE = LST + Net_credit
    High_BE = HST + Max_Loss

    print("Lower Break even (Rs) =", Low_BE)
    print("Higher  Break even (Rs) =", High_BE)
    print("\nMaximum Loss (Rs) =", Max_Loss )
    
    d = dict(A=MEP, B=LSIV, C=PR, D=LSPO, E=HSIV, F=PP, G=HSPO, H = SGPO)
    df = pd.DataFrame(dict([ (pd.Series(k)) for k in d.items() ]))
    df.columns = ["Market Expiry","LS-IV", "PR", "LS Payoff", "HS-IV", "PP","HS Payoff", "Strategy Pay off"] 
    
    print("============================================================================")
    print("Strategy Report:")
    print("============================================================================")
    print("Market Expectation : Rightly Bullish")
    print("\n1) Strategy makes loss if price goes below ", Low_BE ,"Rs, Loss is restricted to", Max_Loss,"Rs")
    print("\n2) Two Low and High Break Even (No profit or Loss) when price goes ", Low_BE,"Rs and", High_BE, "Rs" )
    print("\n3) Makes good money if price goes above ", High_BE)
    print("============================================================================")
    fig, ax = plt.subplots(figsize=(7, 3),dpi=400)
    plt.title('Call Ratio Back Spread Pay Off', fontsize=12)
    plt.xticks(fontsize=10)
    ax.plot(df["Market Expiry"], df["Strategy Pay off"], linestyle='-', color='tab:green', alpha=0.7, linewidth=1.5)
    plt.axhline(0, color='k', linestyle='-',linewidth = 0.5)
    plt.axvline(Low_BE, color='tab:red', linestyle='-',linewidth = 1)
    plt.axvline(High_BE, color='tab:red', linestyle='-',linewidth = 1)
    plt.grid(b=None, which='major', axis='both')
    ax.set_xlabel("Market Expiry Price (Rs)")
    ax.set_ylabel("Profit/Loss (Rs)")
#text(7570, 20, 'Break even' , fontsize='10', fontweight='bold')
    xloc = MultipleLocator(50)
    yloc = MultipleLocator(20)
    ax.xaxis.set_minor_locator(xloc)
    ax.yaxis.set_minor_locator(yloc)   
    return df

#########################################################################################
def Bear_Call_Ladder(MEP,LST,HST1,HST2,Premium_Paid1,Premium_Paid2,Premium_Received):
    Premium_Paid =  Premium_Paid1 + Premium_Paid2
#------------------------------------
    Net_credit = Premium_Received - Premium_Paid
#------------------------------------
    LSIV = np.where(MEP > LST, MEP - LST, 0)
# PREMIUM PAID and Received list
    PP1 = [-Premium_Paid1]*len(MEP)
    PP1_1 = [Premium_Paid1]*len(MEP)
    PP2 = [-Premium_Paid2]*len(MEP)
    PP2_2 = [Premium_Paid2]*len(MEP)
#
    PR = [Premium_Received ]*len(MEP)

# LOWER STRIKE - PAY OFF
    LSPO = PR - LSIV

# Higher STRIKE1 - Intrinsic Value
    HSIV1 = np.where(MEP > HST1, MEP-HST1, 0)
# Higher STRIKE1 - Intrinsic Value
    HSIV2 = np.where(MEP > HST2, MEP-HST2, 0)

# Higher strike Pay off -> 1 Long Position ATM
    HSPO1 = (HSIV1-PP1_1 )
# Higher strike Pay off -> 1 Long Position OTM
    HSPO2 = (HSIV2-PP2_2 )

# STRATEGY PAY OFF
    SGPO = LSPO + HSPO1+ HSPO2

# Spread =(ATM strike - ITM strike) 
    Spread = HST1 - LST
# Maximum Loss
    Max_Loss = Spread - Net_credit
# Break Even
    Low_BE = LST + Net_credit
    High_BE = HST1 + HST2 - LST - Net_credit

    print("Lower Break even (Rs) =", Low_BE)
    print("Higher  Break even (Rs) =", High_BE)
    print("\nMaximum Loss (Rs) =", Max_Loss )
    d = dict(A=MEP, B=LSIV, C=PR, D=LSPO, E=HSIV1, F=PP1, G=HSPO1, H = HSIV2, I=PP2, J=HSPO2, K=SGPO)
    df = pd.DataFrame(dict([ (pd.Series(k)) for k in d.items() ]))
    df.columns = ["Market Expiry","LS-IV", "PR", "Payoff", "HS-IV(ATM)", "PP","Payoff", "HS-IV(OTM)", "PP","Payoff", "Net Pay off"] 
    print("============================================================================")
    print("Strategy Report:")
    print("============================================================================")
    print("Market Expectation : Rightly Bullish")
    print("\n1) Strategy makes loss if price goes below ", Low_BE ,"Rs, Loss is restricted to", Max_Loss,"Rs")
    print("\n2) Two Low and High Break Even (No profit or Loss) when price goes ", Low_BE,"Rs and", High_BE, "Rs" )
    print("\n3) Makes good money if price goes above ", High_BE)
    print("============================================================================")
    fig, ax = plt.subplots(figsize=(7, 3),dpi=400)
    plt.title('Bear Call Ladder Pay Off', fontsize=12)
    plt.xticks(fontsize=10)
    ax.plot(df["Market Expiry"], df["Net Pay off"], linestyle='-', color='tab:green', alpha=0.7, linewidth=1.5)
    plt.axhline(0, color='k', linestyle='-',linewidth = 0.5)
    plt.axvline(Low_BE, color='tab:red', linestyle='-',linewidth = 1)
    plt.axvline(High_BE, color='tab:red', linestyle='-',linewidth = 1)
    plt.grid(b=None, which='major', axis='both')
    plt.yticks(np.arange(min(df["Net Pay off"]), max(df["Net Pay off"])+100, 100.0))
    ax.set_xlabel("Market Expiry Price (Rs)")
    ax.set_ylabel("Profit/Loss (Rs)")
#text(7570, 20, 'Break even' , fontsize='10', fontweight='bold')
    xloc = MultipleLocator(50)
    yloc = MultipleLocator(20)
    ax.xaxis.set_minor_locator(xloc)
    ax.yaxis.set_minor_locator(yloc)
    return df

#########################################################################################    
def Synthetic_Long_Arbitrage(MEP,LST,HST,Premium_Paid,Premium_Received):
    Net_debt = Premium_Paid - Premium_Received
#------------------------------------
    LSIV = np.where(MEP > LST, MEP - LST, 0)
# PREMIUM PAID
    PP = [-Premium_Paid]*len(MEP)
    PP1 = [Premium_Paid]*len(MEP)

# LOWER STRIKE - PAY OFF
    LSPO = LSIV-PP1 
# Higher STRIKE - Intrinsic Value
    HSIV = np.where(MEP < HST, HST-MEP, 0)

    PR = [Premium_Received ]*len(MEP)
# HS - Pay off
    HSPO = PR-HSIV

# STRATEGY NET PAY OFF
    SGPO = LSPO + HSPO

# Break Even
    BE = LST+Net_debt

    print("Break even (Rs) =", BE)
    print("\n============================================================================")
    print("Strategy Report:")
    print("============================================================================")
    print("This strategy provides similar to Long Futures Pay off")
    print("============================================================================")
    d = dict(A=MEP, B=LSIV, C=PP, D=LSPO, E=HSIV, F=PR, G=HSPO, H = SGPO)
    df = pd.DataFrame(dict([ (pd.Series(k)) for k in d.items() ]))
    df.columns = ["Market Expiry","LS-IV", "PP", "LS Payoff", "HS-IV", "PR","HS Payoff", "Net Pay off"] 
    
    fig, ax = plt.subplots(figsize=(7, 3),dpi=400)
    plt.title('Synthetic Long Arbitrage Pay Off', fontsize=12)
    plt.xticks(fontsize=10)
    ax.plot(df["Market Expiry"], df["Net Pay off"], linestyle='-', color='tab:green', alpha=0.7, linewidth=1.5)
    plt.axhline(0, color='k', linestyle='-',linewidth = 0.5)
    plt.axvline(LST+Net_debt, color='tab:red', linestyle='-',linewidth = 1)
    plt.grid(b=None, which='major', axis='both')
    ax.set_xlabel("Market Expiry Price (Rs)")
    ax.set_ylabel("Profit/Loss (Rs)")
#text(7570, 20, 'Break even' , fontsize='10', fontweight='bold')
    xloc = MultipleLocator(50)
    yloc = MultipleLocator(20)
    ax.xaxis.set_minor_locator(xloc)
    ax.yaxis.set_minor_locator(yloc)
    
    return df

#########################################################################################
def Bear_Put(MEP,LST,HST,Premium_Paid,Premium_Received):
    Net_debit = Premium_Received - Premium_Paid
#------------------------------------
    LSIV = np.where(MEP < LST, LST-MEP, 0)
# PREMIUM PAID
    PP = [-Premium_Paid]*len(MEP)
    PP1 = [Premium_Paid]*len(MEP)

# LOWER STRIKE - PAY OFF
    LSPO = LSIV-PP1

# Higher STRIKE - Intrinsic Value
    HSIV = np.where(MEP < HST, HST-MEP, 0)

    PR = [Premium_Received ]*len(MEP)

# HS - Pay off
    HSPO = PR-HSIV

# STRATEGY PAY OFF
    SGPO = LSPO + HSPO

# Break Even
    BE = LST+Net_debit

    Spread = LST-HST
#Maximum Loss/profit
    Max_Loss = Net_debit

    Max_Profit = Spread - (-Net_debit)

    print("Break even (Rs) =", BE)
    print("\nNet Debit & Maximum Loss (Rs) =", Net_debit)
    print("\nMaximum Profit (Rs) =", Max_Profit)
    
    d = dict(A=MEP, B=LSIV, C=PP, D=LSPO, E=HSIV, F=PR, G=HSPO, H = SGPO)
    df = pd.DataFrame(dict([ (pd.Series(k)) for k in d.items() ]))
    df.columns = ["Market Expiry","Long Put-IV", "PP", "LP Payoff", 
                  "Short Put-IV", "PR","SP Payoff", "Strategy Pay off"]
    
    fig, ax = plt.subplots(figsize=(7, 3),dpi=400)
    plt.title('Bear Put Spread Pay Off', fontsize=12)
    plt.xticks(fontsize=10)
    ax.plot(df["Market Expiry"], df["Strategy Pay off"], linestyle='-', 
            color='tab:green', alpha=0.7, linewidth=1.5)
    plt.axhline(0, color='k', linestyle='-',linewidth = 0.5)
    plt.axvline(LST+Net_debit, color='tab:red', linestyle='-',linewidth = 1)
    plt.grid(b=None, which='major', axis='both')
    ax.set_xlabel("Market Expiry Price (Rs)")
    ax.set_ylabel("Profit/Loss (Rs)")
#text(7570, 20, 'Break even' , fontsize='10', fontweight='bold')
    xloc = MultipleLocator(50)
    yloc = MultipleLocator(20)
    ax.xaxis.set_minor_locator(xloc)
    ax.yaxis.set_minor_locator(yloc)
    return df

#########################################################################################
def Bear_Call(MEP,LST,HST,Premium_Paid,Premium_Received):
    Net_credit = Premium_Received - Premium_Paid
#------------------------------------

    LSIV = np.where(MEP > LST, MEP - LST, 0)

# PREMIUM PAID
    PP = [-Premium_Paid]*len(MEP)
    PP1 = [Premium_Paid]*len(MEP)

# LOWER STRIKE - PAY OFF
    LSPO = LSIV-PP1 

# Higher STRIKE - Intrinsic Value
    HSIV = np.where(MEP > HST, MEP-HST, 0)
    PR = [Premium_Received ]*len(MEP)

# HS - Pay off
    HSPO = PR-HSIV 

# STRATEGY PAY OFF
    SGPO = LSPO + HSPO

    Spread = LST - HST
# Break Even
    BE = HST+Net_credit
# Maximum Profit
    Max_Profit = Net_credit 
    Max_Loss = Spread - Net_credit 

    print("Break even (Rs) =", BE)
    print("\nMaximum Profit (Rs) =",Max_Profit )
    print("\nMaximum Loss (Rs) =", Max_Loss)
    
    d = dict(A=MEP, B=LSIV, C=PP, D=LSPO, E=HSIV, F=PR, G=HSPO, H = SGPO)
    df = pd.DataFrame(dict([ (pd.Series(k)) for k in d.items() ]))
    df.columns = ["Market Expiry","Long Call-IV", "PP", "LC Payoff", 
                  "Short Call-IV", "PR","SC Payoff", "Strategy Pay off"] 
    
    fig, ax = plt.subplots(figsize=(7, 3),dpi=400)
    plt.title('Bear Call Spread Pay Off', fontsize=12)
    plt.xticks(fontsize=10)
    ax.plot(df["Market Expiry"], df["Strategy Pay off"], linestyle='-', color='tab:green', alpha=0.7, linewidth=1.5)
    plt.axhline(0, color='k', linestyle='-',linewidth = 0.5)
    plt.axvline(HST+Net_credit, color='tab:red', linestyle='-',linewidth = 1)
    plt.grid(b=None, which='major', axis='both')
    ax.set_xlabel("Market Expiry Price (Rs)")
    ax.set_ylabel("Profit/Loss (Rs)")
    xloc = MultipleLocator(50)
    yloc = MultipleLocator(20)
    ax.xaxis.set_minor_locator(xloc)
    ax.yaxis.set_minor_locator(yloc)
    return df

#########################################################################################
def Put_Ratio_Back(MEP,LST,HST,Premium_Paid1,Premium_Paid2,Premium_Received):
  
    Premium_Paid = Premium_Paid1 + Premium_Paid2
#------------------------------------
    Net_credit = Premium_Received - Premium_Paid
#------------------------------------

    LSIV = np.where(MEP < LST, LST-MEP, 0)

# PREMIUM PAID
    PP = [-Premium_Paid]*len(MEP)
    PP1 = [Premium_Paid]*len(MEP)
    PR = [Premium_Received ]*len(MEP)
# LOWER STRIKE - PAY OFF
    LSPO = PR - LSIV

# Higher STRIKE - Intrinsic Value
    HSIV = np.where(MEP < HST, 2*HST-2*MEP, 0)

# HS - Pay off
    HSPO = HSIV - PP1

# STRATEGY PAY OFF
    SGPO = LSPO + HSPO


    Spread = LST - HST
#Maximum Loss/profit
    Max_Loss = Spread - Net_credit

# Break Even
    Low_BE = HST  - Max_Loss
    High_BE = HST  + Max_Loss


    Max_Profit = Spread - (-Net_credit)
    print("Spread (Rs) =",LST-HST)
    print("\nLower Break even (Rs) =",Low_BE)
    print("\nHigher Break even (Rs) =",High_BE)

    d = dict(A=MEP, B=LSIV, C=PR, D=LSPO, E=HSIV, F=PP, G=HSPO, H = SGPO)
    df = pd.DataFrame(dict([ (pd.Series(k)) for k in d.items() ]))
    df.columns = ["Market Expiry","Short-IV", "PR", "Short Payoff", 
                  "2Long-IV", "PP","Long Payoff", "Strategy Pay off"] 
    
    fig, ax = plt.subplots(figsize=(7, 3),dpi=400)
    plt.title('Put Ratio Back Spread Pay Off', fontsize=12)
    plt.xticks(fontsize=10)
    ax.plot(df["Market Expiry"], df["Strategy Pay off"], 
          linestyle='-', color='tab:green', alpha=0.7, linewidth=1.5)
    plt.axhline(0, color='k', linestyle='-',linewidth = 0.5)
    plt.axvline(Low_BE, color='tab:red', linestyle='-',linewidth = 1)
    plt.axvline(High_BE, color='tab:red', linestyle='-',linewidth = 1)
    plt.grid(b=None, which='major', axis='both')
    ax.set_xlabel("Market Expiry Price (Rs)")
    ax.set_ylabel("Profit/Loss (Rs)")
#text(7570, 20, 'Break even' , fontsize='10', fontweight='bold')
    xloc = MultipleLocator(50)
    yloc = MultipleLocator(20)
    ax.xaxis.set_minor_locator(xloc)
    ax.yaxis.set_minor_locator(yloc)
    return df

#########################################################################################
def Long_Straddle(MEP,LST,HST,Premium_Paid1,Premium_Paid2):
    Net_debt = Premium_Paid1 + Premium_Paid2
#------------------------------------

    LSIV = np.where(MEP > LST, MEP - LST, 0)

# PREMIUM PAID
    PP1 = [-Premium_Paid1]*len(MEP)
    PP1_1 = [Premium_Paid1]*len(MEP)
    PP2 = [-Premium_Paid2]*len(MEP)
    PP2_2 = [Premium_Paid2]*len(MEP)

# LOWER STRIKE - PAY OFF
    LSPO = LSIV-PP1_1 

# Higher STRIKE - Intrinsic Value
    HSIV = np.where(MEP < HST, HST-MEP, 0)

# HS - Pay off
    HSPO = HSIV-PP2_2

# STRATEGY NET PAY OFF
    SGPO = LSPO + HSPO

# Maximum Loss
    Max_Loss = Net_debt
# Break Even
    Low_BE = LST +Net_debt
    High_BE = LST - Net_debt

    print("Lower Break even (Rs) =", Low_BE)
    print("Higher Break even (Rs) =", High_BE)
    print("\nMaximum Loss (Rs) =", Net_debt)
    
    d = dict(A=MEP, B=LSIV, C=PP1, D=LSPO, E=HSIV, F=PP2, G=HSPO, H = SGPO)
    df = pd.DataFrame(dict([ (pd.Series(k)) for k in d.items() ]))
    df.columns = ["Market Expiry","Long Call -IV", "PP", "LC Payoff", "Long Put-IV", "PR","LP Payoff", "Net Pay off"] 
    
    fig, ax = plt.subplots(figsize=(7, 3),dpi=400)
    plt.title('Long Straddle Pay Off', fontsize=12)
    plt.xticks(fontsize=10)
    ax.plot(df["Market Expiry"], df["Net Pay off"], linestyle='-', color='tab:green', alpha=0.7, linewidth=1.5)
    plt.axhline(0, color='k', linestyle='-',linewidth = 0.5)
    plt.axvline(Low_BE, color='tab:red', linestyle='-',linewidth = 1)
    plt.axvline(High_BE, color='tab:red', linestyle='-',linewidth = 1)
    plt.grid(b=None, which='major', axis='both')
    ax.set_xlabel("Market Expiry Price (Rs)")
    ax.set_ylabel("Profit/Loss (Rs)")
    xloc = MultipleLocator(50)
    yloc = MultipleLocator(20)
    ax.xaxis.set_minor_locator(xloc)
    ax.yaxis.set_minor_locator(yloc)
    return df

#########################################################################################

def Short_Straddle(MEP,LST,HST,Premium_Received1,Premium_Received2):
    Net_credit = Premium_Received1 + Premium_Received2
#------------------------------------

    LSIV = np.where(MEP > LST, MEP - LST, 0)

# PREMIUM PAID
    PR1 = [Premium_Received1]*len(MEP)
    PR2 = [Premium_Received2]*len(MEP)

# LOWER STRIKE - PAY OFF
    LSPO = PR1 - LSIV


# Higher STRIKE - Intrinsic Value
    HSIV = np.where(MEP < HST, HST-MEP, 0)

# HS - Pay off
    HSPO = PR2-HSIV

# STRATEGY NET PAY OFF
    SGPO = LSPO + HSPO

# Maximum Loss
    Max_Loss = Net_credit
# Break Even
    Low_BE = LST +Net_credit
    High_BE = LST - Net_credit

    print("\nMaximum Gain (Rs) =", Net_credit)
    
    d = dict(A=MEP, B=LSIV, C=PR1, D=LSPO, E=HSIV, F=PR2, G=HSPO, H = SGPO)
    df = pd.DataFrame(dict([ (pd.Series(k)) for k in d.items() ]))
    df.columns = ["Market Expiry","Short Call-IV", "PR", "SC Payoff", 
                  "Short Put-IV", "PR"," SP Payoff", "Net Pay off"] 
    
    fig, ax = plt.subplots(figsize=(7, 3),dpi=400)
    plt.title('Short Straddle Pay Off', fontsize=12)
    plt.xticks(fontsize=10)
    ax.plot(df["Market Expiry"], df["Net Pay off"], linestyle='-', color='tab:green', alpha=0.7, linewidth=1.5)
    plt.axhline(0, color='k', linestyle='-',linewidth = 0.5)
    plt.axvline(Low_BE, color='tab:red', linestyle='-',linewidth = 1)
    plt.axvline(High_BE, color='tab:red', linestyle='-',linewidth = 1)
    plt.grid(b=None, which='major', axis='both')
    ax.set_xlabel("Market Expiry Price (Rs)")
    ax.set_ylabel("Profit/Loss (Rs)")
#text(7570, 20, 'Break even' , fontsize='10', fontweight='bold')
    xloc = MultipleLocator(50)
    yloc = MultipleLocator(20)
    ylim(-250,250)
    ax.xaxis.set_minor_locator(xloc)
    ax.yaxis.set_minor_locator(yloc)
    
    return df

#########################################################################################

def Long_Strangle(MEP,LST,HST,Premium_Paid1,Premium_Paid2):
    Net_debt = Premium_Paid1 + Premium_Paid2
#------------------------------------

    LSIV = np.where(MEP > LST, MEP - LST, 0)

# PREMIUM PAID
    PP1 = [-Premium_Paid1]*len(MEP)
    PP1_1 = [Premium_Paid1]*len(MEP)
    PP2 = [-Premium_Paid2]*len(MEP)
    PP2_2 = [Premium_Paid2]*len(MEP)

# LOWER STRIKE - PAY OFF
    LSPO = LSIV-PP1_1 

# Higher STRIKE - Intrinsic Value
    HSIV = np.where(MEP < HST, HST-MEP, 0)

# HS - Pay off
    HSPO = HSIV-PP2_2

# STRATEGY NET PAY OFF
    SGPO = LSPO + HSPO

# Maximum Loss
    Max_Loss = Net_debt
# Break Even
    Low_BE = HST - Net_debt
    High_BE = LST + Net_debt

    print("Lower Break even (Rs) =", Low_BE)
    print("Higher Break even (Rs) =", High_BE)
    print("\nMaximum Loss (Rs) =", Net_debt)
    
    d = dict(A=MEP, B=LSIV, C=PP1, D=LSPO, E=HSIV, F=PP2, G=HSPO, H = SGPO)
    df = pd.DataFrame(dict([ (pd.Series(k)) for k in d.items() ]))
    df.columns = ["Market Expiry","Call-IV", "PP", "Call Payoff", "Put-IV", "PP","Put Payoff", "Net Pay off"] 

    fig, ax = plt.subplots(figsize=(7, 3),dpi=400)
    plt.title('Long Strangle Pay Off', fontsize=12)
    plt.xticks(fontsize=10)
    ax.plot(df["Market Expiry"], df["Net Pay off"], linestyle='-', color='tab:green', alpha=0.7, linewidth=1.5)
    plt.axhline(0, color='k', linestyle='-',linewidth = 0.5)
    plt.axvline(Low_BE, color='tab:red', linestyle='-',linewidth = 1)
    plt.axvline(High_BE, color='tab:red', linestyle='-',linewidth = 1)
    plt.grid(b=None, which='major', axis='both')
    ax.set_xlabel("Market Expiry Price (Rs)")
    ax.set_ylabel("Profit/Loss (Rs)")
#text(7570, 20, 'Break even' , fontsize='10', fontweight='bold')
    xloc = MultipleLocator(50)
    yloc = MultipleLocator(20)
    ax.xaxis.set_minor_locator(xloc)
    ax.yaxis.set_minor_locator(yloc)
    
    return df

#########################################################################################
def Short_Strangle(MEP,LST,HST,Premium_Received1,Premium_Received2):

    Net_credit = Premium_Received1 + Premium_Received2
#------------------------------------
    LSIV = np.where(MEP > LST, MEP - LST, 0)

# PREMIUM PAID
    PR1 = [Premium_Received1]*len(MEP)
    PR2 = [Premium_Received2]*len(MEP)

# LOWER STRIKE - PAY OFF
    LSPO = PR1 - LSIV
# Higher STRIKE - Intrinsic Value
    HSIV = np.where(MEP < HST, HST-MEP, 0)

# HS - Pay off
    HSPO = PR2 - HSIV
# STRATEGY NET PAY OFF
    SGPO = LSPO + HSPO
# Maximum Loss
    Max_Loss = Net_credit
# Break Even

    Low_BE = HST - Net_credit
    High_BE = LST + Net_credit
    print("\nMaximum Profit (Rs) =", Net_credit)
    
    d = dict(A=MEP, B=LSIV, C=PR1, D=LSPO, E=HSIV, F=PR2, G=HSPO, H = SGPO)
    df = pd.DataFrame(dict([ (pd.Series(k)) for k in d.items() ]))
    df.columns = ["Market Expiry","Call-IV", "PP", "Call Payoff", "Put-IV", "PR","Put Payoff", "Net Pay off"] 

    fig, ax = plt.subplots(figsize=(7, 3),dpi=400)
    plt.title('Short Strangle Pay Off', fontsize=12)
    plt.xticks(fontsize=10)
    ax.plot(df["Market Expiry"], df["Net Pay off"], linestyle='-', color='tab:green', alpha=0.7, linewidth=1.5)
    plt.axhline(0, color='k', linestyle='-',linewidth = 0.5)
    plt.axvline(Low_BE, color='tab:red', linestyle='-',linewidth = 1)
    plt.axvline(High_BE, color='tab:red', linestyle='-',linewidth = 1)
    plt.grid(b=None, which='major', axis='both')
    ax.set_xlabel("Market Expiry Price (Rs)")
    ax.set_ylabel("Profit/Loss (Rs)")
#text(7570, 20, 'Break even' , fontsize='10', fontweight='bold')
    xloc = MultipleLocator(50)
    yloc = MultipleLocator(20)
    ax.xaxis.set_minor_locator(xloc)
    ax.yaxis.set_minor_locator(yloc)
    return df


#########################################################################################
def Iron_Condor(MEP,LST1,LST2,HST1,HST2,Pre_Rec1,Pre_Rec2, Pre_Paid1, Pre_Paid2):
    Net_credit = Pre_Rec1 + Pre_Rec2 - Pre_Paid1 - Pre_Paid2
#------------------------------------
# LOWER STRIKE - Intrinsic Value
    LSIV1 = np.where(MEP > LST1, MEP - LST1, 0)

# PREMIUM PAID
    PR1 = [Pre_Rec1]*len(MEP)
    PR2 = [Pre_Rec2]*len(MEP)

# LOWER STRIKE - PAY OFF
    LSPO1 = PR1 - LSIV1

# Higher STRIKE - Intrinsic Value
    HSIV1 = np.where(MEP < HST1, HST1 -MEP, 0)

# HS - Pay off
    HSPO1 = PR2 - HSIV1
#--------------------------------------

    LSIV2 = np.where(MEP > LST2, MEP - LST2, 0)

# PREMIUM PAID
    PP1 = [-Pre_Paid1]*len(MEP)
    PP1_1 = [Pre_Paid1]*len(MEP)
    PP2 = [-Pre_Paid2]*len(MEP)
    PP2_2 = [Pre_Paid2]*len(MEP)

# LOWER STRIKE - PAY OFF
    LSPO2 = LSIV2 - PP1_1 

# Higher STRIKE - Intrinsic Value
    HSIV2 = np.where(MEP < HST2, HST2-MEP, 0)

# HS - Pay off
    HSPO2 = HSIV2 - PP2_2

# STRATEGY NET PAY OFF
    SGPO = LSPO1 + HSPO1 + LSPO2 + HSPO2

# Maximum Loss/Profit
    Max_Loss = HST1 - HST2 - Net_credit
    Max_Profit = Net_credit
# Break Even

    Low_BE = HST1 - Net_credit
    High_BE = LST1 + Net_credit
    
    print("\nMaximum Loss (Rs) =", Max_Loss)
    print("\nMaximum Profit (Rs) =", Max_Profit)
    print("\nHigher Break Even (Rs) =", High_BE)
    print("\nLower Break Even (Rs) =", Low_BE)
    
    d = dict(A=MEP, B=LSIV1, C=PR1, D=LSPO1, E=HSIV1, F=PR2, G=HSPO1, H=LSIV2, I=PP1, J=LSPO2, K=HSIV2, L=PP2, M=HSPO2, N=SGPO)
    df = pd.DataFrame(dict([(pd.Series(k)) for k in d.items() ]))
    df.columns = ["Market Expiry","Short_CE", "PR", "S_CE_Pay", "Short_PE", "PR","S_PE_Pay",
                  "Long_CE", "PP", "L_CE_Pay", "Long_PE", "PP","L_PE_Pay", "Net Pay off"] 
    
    fig, ax = plt.subplots(figsize=(7, 3),dpi=400)
    plt.title('Iron Condor Pay Off', fontsize=12)
    plt.xticks(fontsize=10)
    ax.plot(df["Market Expiry"], df["Net Pay off"], linestyle='-', color='tab:green', alpha=0.7, linewidth=1.5)
    plt.axhline(0, color='k', linestyle='-',linewidth = 0.5)
    plt.axvline(Low_BE, color='tab:red', linestyle='-',linewidth = 1)
    plt.axvline(High_BE, color='tab:red', linestyle='-',linewidth = 1)
    plt.grid(b=None, which='major', axis='both')
    ax.set_xlabel("Market Expiry Price (Rs)")
    ax.set_ylabel("Profit/Loss (Rs)")
#text(7570, 20, 'Break even' , fontsize='10', fontweight='bold')
    xloc = MultipleLocator(50)
    yloc = MultipleLocator(20)
    ax.xaxis.set_minor_locator(xloc)
    ax.yaxis.set_minor_locator(yloc)
    
    return df

#########################################################################################
def Butterfly_Spread(MEP,LST1,LST2,HST,Premium_Paid1, Premium_Paid2,Premium_Rec1,Premium_Rec2):
    Premium_Received = Premium_Rec1 + Premium_Rec2
    Premium_Paid = Premium_Paid1 + Premium_Paid2
#------------------------------------
    Net_debt = Premium_Paid - Premium_Received
#------------------------------------
# PREMIUM PAID
    PP1 = [-Premium_Paid1]*len(MEP)
    PP1_1 = [Premium_Paid1]*len(MEP)
    PP2 = [-Premium_Paid2]*len(MEP)
    PP2_2 = [Premium_Paid2]*len(MEP)

    PR = [Premium_Received]*len(MEP)

    LSIV1 = np.where(MEP > LST1, MEP - LST1, 0)
# LOWER STRIKE - PAY OFF
    LSPO1 = LSIV1 - PP1_1 

    LSIV2 = np.where(MEP > LST2, 2*MEP - 2*LST2, 0)

    LSPO2 = PR - LSIV2

# Higher STRIKE - Intrinsic Value
    HSIV = np.where(MEP > HST, MEP-HST, 0)
    HSPO = HSIV - PP2_2

# STRATEGY PAY OFF
    SGPO = LSPO1 + LSPO2 + HSPO

# Maximum Loss/Profit
    Max_Loss = Net_debt
    Max_Profit = LST2 - LST1 + Premium_Received - Premium_Paid

    Low_BE = LST1
    High_BE = HST
    print("Maximum Loss (Rs) = ", -Max_Loss )
    print("Maximum Profit (Rs) = ", Max_Profit )

    print("\nLower Break Even (Rs) = ", LST1 )
    print("Higher Break Even (Rs) = ", HST )
    
    d = dict(A=MEP, B=LSIV1, C=PP1, D=LSPO1, E=LSIV1, F=PR, G=LSPO2, H=HSIV, I=PP2, J=HSPO, N=SGPO)
    df = pd.DataFrame(dict([(pd.Series(k)) for k in d.items() ]))
    df.columns = ["Market Expiry","Long_CE-IV", "PP", "L_CE_Pay", "2_Short_CE-IV", "PR","2_S_CE_Pay",
                  "Long_CE-IV", "PP", "L_CE_Pay","Net Pay off"] 
     
    print("Strategy Report:")
    print("============================================================================")
    print("Maximum Profit at the price ", LST2, "(Short Position @ ATM)")
    print("============================================================================")

    fig, ax = plt.subplots(figsize=(7, 3),dpi=400)
    plt.title('Butterfly Pay Off', fontsize=12)
    plt.xticks(fontsize=10)
    ax.plot(df["Market Expiry"], df["Net Pay off"], linestyle='-', color='tab:green', alpha=0.7, linewidth=1.5)
    plt.axhline(0, color='k', linestyle='-',linewidth = 0.5)
    plt.axvline(Low_BE, color='tab:red', linestyle='-',linewidth = 1)
    #plt.axvline(LST2, color='tab:red', linestyle='-',linewidth = 1)
    plt.axvline(High_BE, color='tab:red', linestyle='-',linewidth = 1)
    plt.grid(b=None, which='major', axis='both')
    ax.set_xlabel("Market Expiry Price (Rs)")
    ax.set_ylabel("Profit/Loss (Rs)")
    xloc = MultipleLocator(50)
    yloc = MultipleLocator(20)
    ax.xaxis.set_minor_locator(xloc)
    ax.yaxis.set_minor_locator(yloc)
    
    return df
#########################################################################################

def Iron_Fly(MEP,LST1,LST2,HST1,HST2,Pre_Rec1,Pre_Rec2, Pre_Paid1, Pre_Paid2):
# BULL CALL
    Net_PR = Pre_Rec1+Pre_Rec2
    # LONG CALL - Intrinsic Value
    LSIV1 = np.where(MEP > LST1, MEP - LST1, 0)
    
# PREMIUM PAID
    PP1 = [-Pre_Paid1]*len(MEP)
    PP1_1 = [Pre_Paid1]*len(MEP)
    
# LONG CALL PAY OFF
    LSPO1 = LSIV1 - PP1_1

# SHORT CALL Intrinsic Value
    HSIV1 = np.where(MEP > HST1, MEP-HST1, 0)
# PREMIUM RECEIVED
    PR1 = [Pre_Rec1]*len(MEP)
    
# SHORT CALL Pay off
    HSPO1 = PR1 - HSIV1
    
# BEAR PUT    
    LSIV2 = np.where(MEP < LST2, LST2-MEP, 0)

    PP2 = [-Pre_Paid2]*len(MEP)
    PP2_2 = [Pre_Paid2]*len(MEP)
    
    PR2 = [Pre_Rec2]*len(MEP)

# LONG PUT PAY OFF
    LSPO2 = LSIV2 - PP2_2 

# Higher STRIKE - Intrinsic Value
    HSIV2 = np.where(MEP < HST2, HST2-MEP, 0)

# SHORT PUT Pay off
    HSPO2 = PR2 - HSIV2
    
#STRATEGY PAY OFF    
    SGPO = LSPO1 + HSPO1 + LSPO2 + HSPO2
    
    # High_BE = Strike Price of Short Call + Net Premium Received 
    High_BE = LST2 + Net_PR
    #Lower Breakeven Point = Strike Price of Short Put - Net Premium Received
    Low_BE = HST2 + Net_PR
    
    print("\nMaximum Loss (Rs) =", min(SGPO))
    print("\nMaximum Profit (Rs) =", max(SGPO) )
    #print("\n Higher Break Even (Rs) =", High_BE)
    #print("\n Lower Break Even (Rs) =", Low_BE)
    
    d = dict(A=MEP, B=LSIV1, C=PR1, D=LSPO1, E=HSIV1, F=PR2, G=HSPO1, 
              H=LSIV2, I=PP1, J=LSPO2, K=HSIV2, L=PP2, M=HSPO2, N=SGPO)
    df = pd.DataFrame(dict([(pd.Series(k)) for k in d.items() ]))
    df.columns = ["Market Expiry","Short_CE", "PR", "S_CE_Pay", "Short_PE", "PR","S_PE_Pay",
                  "Long_CE", "PP", "L_CE_Pay", "Long_PE", "PP","L_PE_Pay", "Net Pay off"] 
    
    fig, ax = plt.subplots(figsize=(7, 3),dpi=400)
    plt.title('Iron Fly Pay Off', fontsize=12)
    plt.xticks(fontsize=10)
    ax.plot(df["Market Expiry"], df["Net Pay off"], linestyle='-', color='tab:green', alpha=0.7, linewidth=1.5)
    plt.axhline(0, color='k', linestyle='-',linewidth = 0.5)
    #plt.axvline(Low_BE, color='tab:red', linestyle='-',linewidth = 1)
    #plt.axvline(High_BE, color='tab:red', linestyle='-',linewidth = 1)
    plt.grid(b=None, which='major', axis='both')
    ax.set_xlabel("Market Expiry Price (Rs)")
    ax.set_ylabel("Profit/Loss (Rs)")
#text(7570, 20, 'Break even' , fontsize='10', fontweight='bold')
    xloc = MultipleLocator(50)
    yloc = MultipleLocator(20)
    ax.xaxis.set_minor_locator(xloc)
    ax.yaxis.set_minor_locator(yloc)
    
    return df
#########################################################################################

#########################################################################################

