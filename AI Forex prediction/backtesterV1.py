#!/usr/bin/env python
# coding: utf-8

# In[ ]:

#explanation of these methods is given in report
# DATA ANALYSIS


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sb
import plotly.graph_objects as go
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


#regression

from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor

############################################################################


class DataHandler:
    
    def __init__(self, data, target, pos_Z_boundry, neg_Z_boundry, Zscore, Minmax, zout):
        self.data = data
        self.target = target
        self.df = data
        self.pos_Z_boundry = pos_Z_boundry
        self.neg_Z_boundry = neg_Z_boundry
        self.Zscore = Zscore
        self.Minmax = Minmax
        self.zout = zout
        self.all_features = data.columns[4:-1]
        
    def DataCleaner(self):
        self.df = self.df.fillna(0)     #fill nan=0
        return (self.df)
    
    def LabelHandler(self, clean):
        label = clean
        label[self.target] = np.where((clean[self.target]=="Buy"), 1, -1)
        return (label)
    
    def Z_outlier(self, data_df):
        Z = (data_df['Close']-data_df['Close'].mean())/data_df['Close'].std()
        Z = Z[(Z<=self.pos_Z_boundry) & (Z>=self.neg_Z_boundry)].index
        data_oc = self.data[['Close', 'Open']].loc[Z]
        data_df = data_df.loc[Z]
        return (data_df, data_oc)     
    
    def Normalizer(self, category):
        category = pd.DataFrame(category)
        y = np.array(category.iloc[:, -1:])
        if (self.Zscore == True):
          from sklearn.preprocessing import StandardScaler
          scaler = StandardScaler()
          scaled = scaler.fit_transform(category.iloc[:, 4:-1])                #except clf data
          df_ = pd.DataFrame(scaled, columns = self.all_features)
          df_['Signal'] = y
        elif (self.Minmax == True):
          from sklearn.preprocessing import MinMaxScaler
          scaler = MinMaxScaler()
          scaled = scaler.fit_transform(category.iloc[:, 4:-1])                #except clf data
          df_ = pd.DataFrame(scaled, columns = self.all_features)
          df_['Signal'] = y
        return (df_)
    
    def Pip_Analyser(self):
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_traces(go.Scatter(x = self.data.index, y = self.data['Close']-self.data['Open'], name = "pip range"))
        fig.add_traces(go.Scatter(x = [0, len(self.data)], y = [(self.data['Close']-self.data['Open']).mean(), 
                                                   (self.data['Close']-self.data['Open']).mean()], name = 'Mean-range'))
        fig.add_traces(go.Scatter(x = [0, len(self.data)], y = [(self.data['Close']-self.data['Open']).max(), (self.data['Close']-self.data['Open']).max()], name = 'Max-range'))
        fig.add_traces(go.Scatter(x = [0, len(self.data)], y = [(self.data['Close']-self.data['Open']).min(), 
                                                   (self.data['Close']-self.data['Open']).min()], name = 'Min-range'))
        fig.add_traces(go.Scatter(x = [0, len(self.data)], y = [(self.data['Close']-self.data['Open']).quantile(q = 0.25), 
                                                   (self.data['Close']-self.data['Open']).quantile(q = 0.25)], name = '25% quantile-range'))
        fig.add_traces(go.Scatter(x = [0, len(self.data)], y = [(self.data['Close']-self.data['Open']).quantile(q = 0.50), 
                                                   (self.data['Close']-self.data['Open']).quantile(q = 0.50)], name = '50% quantile-range'))
        fig.add_traces(go.Scatter(x = [0, len(self.data)], y = [(self.data['Close']-self.data['Open']).quantile(q = 0.75), 
                                                   (self.data['Close']-self.data['Open']).quantile(q = 0.75)], name = '75% quantile-range'))
        fig.update_layout(title = "Open-Close Pip Range", title_x = 0.5, showlegend = True, xaxis_title = "Total Pips" , yaxis_title = "Total Data" )
        
        
        fig_1 = go.Figure()
        fig_1.add_traces(go.Scatter(x = self.data.index, y = self.data['High']-self.data['Open'], name = "pip range"))
        fig_1.add_traces(go.Scatter(x = [0, len(self.data)], y = [(self.data['High']-self.data['Open']).mean(), 
                                                   (self.data['High']-self.data['Open']).mean()], name = 'Mean-range'))
        fig_1.add_traces(go.Scatter(x = [0, len(self.data)], y = [(self.data['High']-self.data['Open']).max(), (self.data['High']-self.data['Open']).max()], name = 'Max-range'))
        fig_1.add_traces(go.Scatter(x = [0, len(self.data)], y = [(self.data['High']-self.data['Open']).min(), 
                                                   (self.data['High']-self.data['Open']).min()], name = 'Min-range'))
        fig_1.add_traces(go.Scatter(x = [0, len(self.data)], y = [(self.data['High']-self.data['Open']).quantile(q = 0.25), 
                                                   (self.data['High']-self.data['Open']).quantile(q = 0.25)], name = '25% quantile-range'))
        fig_1.add_traces(go.Scatter(x = [0, len(self.data)], y = [(self.data['High']-self.data['Open']).quantile(q = 0.50), 
                                                   (self.data['High']-self.data['Open']).quantile(q = 0.50)], name = '50% quantile-range'))
        fig_1.add_traces(go.Scatter(x = [0, len(self.data)], y = [(self.data['High']-self.data['Open']).quantile(q = 0.75), 
                                                   (self.data['High']-self.data['Open']).quantile(q = 0.75)], name = '75% quantile-range'))
        fig_1.update_layout(title = "Open-High Pip Range", title_x = 0.5, showlegend = True,  xaxis_title = "Total Pips" , yaxis_title = "Total Data")

        
        fig_2 = go.Figure()
        fig_2.add_traces(go.Scatter(x = self.data.index, y = self.data['Open']-self.data['Low'], name = "pip range"))
        fig_2.add_traces(go.Scatter(x = [0, len(self.data)], y = [(self.data['Open']-self.data['Low']).mean(), 
                                                   (self.data['Open']-self.data['Low']).mean()], name = 'Mean-range'))
        fig_2.add_traces(go.Scatter(x = [0, len(self.data)], y = [(self.data['Open']-self.data['Low']).max(), (self.data['Open']-self.data['Low']).max()], name = 'Max-range'))
        fig_2.add_traces(go.Scatter(x = [0, len(self.data)], y = [(self.data['Open']-self.data['Low']).min(), 
                                                   (self.data['Open']-self.data['Low']).min()], name = 'Min-range'))
        fig_2.add_traces(go.Scatter(x = [0, len(self.data)], y = [(self.data['Open']-self.data['Low']).quantile(q = 0.25), 
                                                   (self.data['Open']-self.data['Low']).quantile(q = 0.25)], name = '25% quantile-range'))
        fig_2.add_traces(go.Scatter(x = [0, len(self.data)], y = [(self.data['Open']-self.data['Low']).quantile(q = 0.50), 
                                                   (self.data['Open']-self.data['Low']).quantile(q = 0.50)], name = '50% quantile-range'))
        fig_2.add_traces(go.Scatter(x = [0, len(self.data)], y = [(self.data['Open']-self.data['Low']).quantile(q = 0.75), 
                                                   (self.data['Open']-self.data['Low']).quantile(q = 0.75)], name = '75% quantile-range'))
        fig_2.update_layout(title = "Open-Low Pip Range", title_x = 0.5, showlegend = True,  xaxis_title = "Total Pips" , yaxis_title = "Total Data" )
        
        fig.show()
        fig_1.show()
        fig_2.show()
        return
        
    def DataAnalyzer(self):
        pips = self.Pip_Analyser()
        clean = self.DataCleaner()
        category = self.LabelHandler(clean)
        if (self.zout == True):
          z_data, data_oc = self.Z_outlier(category)
        else:
          z_data = category 
          data_oc = category[['Close', 'Open']]
        normalize = self.Normalizer(z_data)
        return (normalize, data_oc) 
    
    
class FeatureSelector:
    
    def __init__(self, X, y, train_size):
        self.X = X
        self.y = self.X['shift_1Close']
        self.train_size = train_size
        
    def corrleate(self):
        import matplotlib.pyplot as plt
        import seaborn as sb
        self.X['y'] = self.y
        plt.figure(figsize = (8,8))
        sb.heatmap(self.X.corr())
        plt.title("Correlation Matrix")
        plt.show()
        return 
        
    def regression_Selection(self):
        features = self.X.columns
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(self.X, self.y)
        coef_reg = regressor.coef_
        return (features, coef_reg)
        
    def forest_selection(self):
        features = self.X.columns
        from sklearn.ensemble import RandomForestRegressor
        rf_ = RandomForestRegressor(n_estimators = 50, criterion = 'mse')
        rf_.fit(self.X, self.y)
        rf_coef = rf_.feature_importances_
        return (features, rf_coef)

    def Pca_selector(self, df_x):
      from sklearn.decomposition import PCA
      pca = PCA(n_components = 1)
      pca_x = pca.fit_transform(df_x)
      return (pca_x)
    
    
class BackTester:

  def __init__(self, strategy, strategyType, initial_Amount, lot):

    self.data, self.predreg,self.actualReg, self.predclf, self.allpredclf, self.actualClf = strategy
    self.strategyType = strategyType
    self.initial_Amount = initial_Amount
    self.lot = lot

  def PNL_Calculation(self):

    if (self.strategyType == 'Long-Only'):
      AIbuy_pnl = []
      for price, open, signal in zip(self.data['Close'], self.data['Open'], self.data['AI_signal']):
        if (signal == 1):
          AIbuy_pnl.append((price-open)*10000)
        else:
          AIbuy_pnl.append(0)
      self.data['AI_P&L'] = AIbuy_pnl

    elif (self.strategyType == 'Short-Only'):
      AIsell_pnl = []
      for price, open, signal in zip(self.data['Close'], self.data['Open'], self.data['AI_signal']):
        if (signal == -1):
          AIsell_pnl.append((open-price)*10000)
        else:
          AIsell_pnl.append(0)
      self.data['AI_P&L'] = AIsell_pnl

    elif (self.strategyType == 'Long-Short'):
      AI_pnl = []
      for price, open, signal in zip(self.data['Close'], self.data['Open'], self.data['AI_signal']):
        if (signal == 1):
          AI_pnl.append((price-open)*10000)
        elif (signal == -1):
          AI_pnl.append((open-price)*10000)
        else:
            AI_pnl.append(0)    
      self.data['AI_P&L'] = AI_pnl

    return (self.data)

  def PNL_plots(self):

    PL = self.PNL_Calculation()
    #long-short
    AI_pnl = []
    initial_deposit = self.initial_Amount
    for profit in PL['AI_P&L']:
      initial_deposit = initial_deposit + profit*self.lot
      AI_pnl.append(initial_deposit)
    PnL_Both = pd.DataFrame(AI_pnl, columns = ["P&L"])
        
    #long only
    df_buyonly = pd.DataFrame(data = PL['AI_P&L'].loc[PL.loc[PL['AI_signal']==1].index]).reset_index(drop = True)
    pnl_buy = []
    initial_deposit = self.initial_Amount
    for profit in df_buyonly['AI_P&L']:
      initial_deposit = initial_deposit + profit*self.lot
      pnl_buy.append(initial_deposit)
    PnL_Buy = pd.DataFrame(pnl_buy, columns = ["P&L"])
    
    #short only
    df_sellonly = pd.DataFrame(data = PL['AI_P&L'].loc[PL.loc[PL['AI_signal']==-1].index]).reset_index(drop = True)
    pnl_sell = []
    initial_deposit = self.initial_Amount
    for profit in df_sellonly['AI_P&L']:
      initial_deposit = initial_deposit + profit*self.lot
      pnl_sell.append(initial_deposit)
    PnL_Sell = pd.DataFrame(pnl_sell, columns = ["P&L"])
    
    # report
    
    report_strt = {"Total Test Data":len(PL), "Initial Deposit":self.initial_Amount, 'Total Profit': PL.loc[PL['AI_P&L']>0].sum()[3], 
        'Total Loss':PL.loc[PL['AI_P&L']<0].sum()[3],  'Gross Profit':(PL['AI_P&L'].sum()+self.initial_Amount-self.initial_Amount), 
         'Net Profit($1 commission)':(PL['AI_P&L'].sum()+self.initial_Amount-self.initial_Amount)-1*(len(PL)), 
         'Largest Profit':PL['AI_P&L'].max(), 'Largest Loss':PL['AI_P&L'].min(), 
        'Total profit trades':len(PL.loc[PL['AI_P&L']>0]), 'Total Loss trades':len(PL.loc[PL['AI_P&L']<0])}
    strategy_report = pd.DataFrame(data = [report_strt])
    return(strategy_report, PnL_Both, PnL_Buy, PnL_Sell, self.actualReg.values, self.predreg.values, self.actualClf, self.predclf, self.allpredclf)

