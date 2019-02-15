#Load the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib
import seaborn as sns
import missingno as msno


from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


#load the dataset
data = pd.read_csv('cn_stocks.csv')

data.info()
data.dtypes

#visualize missing values
msno.matrix(data)
msno.bar(data)
data.isnull().sum()

#imputing with KNN
import fancyimpute as fi
stocks = fi.KNN(k=8).complete(data.iloc[:,1:20])
stocks = pd.DataFrame(stocks) #convert np.array to pd.df

#adding the date column back
stocks = pd.concat([data['Index'], stocks], axis=1)

#Rename columns
stocks.columns = ['Date', 'Drabona', 'Eladda', 'Nasin', 'Gilvar', 
                'Din', 'Sonja', 'Sarion', 'Zelda', 'Azurehoof',
                'Nietrem', 'Gath', 'Pypina', 'Volodymyr',
                'Ysydda', 'Ronak', 'Ulvila', 'Voorogg', 'Finalien']

#convert Date column to datetime
stocks['Date'] = pd.to_datetime(stocks['Date'], format='%Y-%m-%d')
stocks.set_index('Date', inplace=True) #set date column as index

#exploring with plots
stocks.describe()
stocks.plot()
stocks['Pypina'].plot()



stocks.reset_index().plot(x='Date', y=stocks.columns[stocks.columns != 'Pypina'])


hello = stocks.drop(['Pypina'], axis=1)
fig = plt.figure()
ax = plt.axes()
ax.plot(hello)
x1 = '2017-06-01'
x2 = '2017-12-29'
y1 = 0
y2 = 50
axins = zoomed_inset_axes(ax, 4, loc=1)
axins.plot(hello)
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
plt.xticks(visible=False)
plt.yticks(visible=False)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
plt.draw()

#fixing the last part
stocks = stocks.loc[stocks.index < pd.to_datetime('2017-10-08')]
stocks.plot()

#prices correlation heatmap
corrmat = stocks.corr()
cm = np.corrcoef(stocks.values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True,
                 fmt='.2f', annot_kws={'size':10},
                 yticklabels=stocks.columns, xticklabels=stocks.columns
                )
plt.show()

#daily returns
def compute_daily_returns(df):
    daily_returns = df.pct_change()
    daily_returns.iloc[0, :] = 0
    return daily_returns

returns = compute_daily_returns(stocks)

#more EDA


sns.set(style='darkgrid', font_scale=1.75)
sns.boxplot(x='variable', y='value', data=pd.melt(returns))
plt.xlabel('Stocks', fontsize=20, fontweight='bold', labelpad=15) 
plt.ylabel('Percent', fontsize=20, fontweight='bold', labelpad=15)
plt.title('% Change in Daily Returns', fontsize=30, fontweight='bold')

sns.violinplot(x='variable', y='value', data=pd.melt(returns))
plt.xlabel('Stocks', fontsize=20, fontweight='bold', labelpad=15) 
plt.ylabel('Percent', fontsize=20, fontweight='bold', labelpad=15)
plt.title('% Change in Daily Returns', fontsize=30, fontweight='bold')

#pandas histogram
returns.hist(bins=40)
#seaborn's histograms
sns.FacetGrid(data=pd.melt(returns), col='variable', col_wrap=3).map(sns.distplot, 'value')

#returns corrs heatmap
corrmat = returns.corr()
cm = np.corrcoef(returns.values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True,
                 fmt='.2f', annot_kws={'size':10},
                 yticklabels=returns.columns, xticklabels=returns.columns
                )
plt.show()
returns.plot()

#create portfolio
portfolio = returns.sum(axis=1)
portfolio.head()

#cumulative return of the portfolio
portfolio_cum = (returns + 1).cumprod() - 1
portfolio_cum = portfolio_cum.sum(axis=1)
portfolio_cum.plot()

# =============================================================================
# Trading strategy
# =============================================================================
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import mixture as mix

#define train/test ratios
n=10
t=0.7
split = int(t*len(portfolio))

#initiate mixture model
unsup = mix.GaussianMixture(n_components=2, 
                            covariance_type='spherical',
                            n_init=100,
                            random_state=0)

#fit the model to the data
scaler = StandardScaler()
unsup.fit(np.reshape(scaler.fit_transform(returns[:split]), (-1, returns.shape[1])))
regime = unsup.predict(np.reshape(scaler.fit_transform(returns[split:]), (-1, returns.shape[1])))

#create another column of the sum of daily returns for the entire portfolio
returns['Port_Return'] = returns.sum(axis=1) #sum row-wise across each column

#create the regimes dataframe and add othe required columns
Regimes = pd.DataFrame(regime, columns=['Regime'], index=returns[split:].index)\
            .join(returns[split:], how='inner')\
                        .reset_index(drop=False)\
                        .rename(columns={'index':'Date'})

#Cumulative return = (returns + 1).cumprod() - 1
returns11 = ((stocks[split:].pct_change().fillna(0) + 1).cumprod() - 1).sum(axis=1).reset_index()
returns11.drop('Date', axis=1, inplace=True)
Regimes['market_cu_return'] = returns11

#visualize the GMM model's results                     
order = [0, 1]
fig = sns.FacetGrid(data=Regimes, hue='Regime', hue_order=order,
                    aspect=2, height=4)
fig.map(plt.scatter, 'Date', 'market_cu_return', s=4).add_legend()
plt.show()

#print mean and covariance of each regime
for i in order:
    print('Mean for regime %i: '%i, unsup.means_[i][0])
    print('Covariance for regime %i: '%i, (unsup.covariances_[i]))
    
#scaling the Regimes df to train SVC and create signal column    
columns = Regimes.columns.drop(['Regime', 'Date'])
Regimes[columns] = scaler.fit_transform(Regimes[columns])
Regimes['Signal'] = 0
Regimes.loc[Regimes['Port_Return'] > 0, 'Signal'] = 1
Regimes.loc[Regimes['Port_Return'] < 0, 'Signal'] = -1

#initiate classifier
classifier = SVC(C=1.0, cache_size=100, class_weight=None, coef0=0.0,
                decision_function_shape=None, degree=1, gamma='auto',
                kernel='rbf', max_iter=-1, probability=False,
                random_state=None, shrinking=True, tol=0.001,
                verbose=False)

#split and fit the classifier
split2 = int(0.85 * len(Regimes))

X = Regimes.drop(['Signal', 'Port_Return', 'market_cu_return', 'Date'], axis=1)
y = Regimes['Signal']

classifier.fit(X[:split2], y[:split2])

#----------
p_data = len(X)-split2

returns['Pred_Signal'] = 0
returns.iloc[-p_data: , returns.columns.get_loc('Pred_Signal')] = classifier.predict(X[split2:])

print(returns['Pred_Signal'][-p_data:])
returns['str_ret'] = returns['Pred_Signal'] * returns['Port_Return'].shift(-1)
returns.fillna(0, inplace=True)

returns['strategy_cu_return'] = 0
returns['market_cu_return'] = 0
returns.iloc[-p_data: , returns.columns.get_loc('strategy_cu_return')]\
    = np.nancumsum(returns['str_ret'][-p_data:])
returns.iloc[-p_data: , returns.columns.get_loc('market_cu_return')]\
    = np.nancumsum(returns['Port_Return'][-p_data:])
Sharpe = (returns['strategy_cu_return'][-1] - returns['market_cu_return'][-1])\
    / np.nanstd(returns['strategy_cu_return'][-p_data:])
    
plt.plot(returns['strategy_cu_return'][-p_data:], color='g', label='Strategy Returns')
plt.plot(returns['market_cu_return'][-p_data:], color='r', label='Market Returns')
plt.figtext(0.14, 0.9, s='Sharpe ratio: %.2f'%Sharpe)
plt.legend(loc='best')
plt.show()

# =============================================================================
# Testing strategy on individual stocks
# =============================================================================
stock = pd.DataFrame(stocks['Volodymyr'])
stock['Return'] = stock.pct_change()
stock.fillna(0, inplace=True)

#define train/test ratios
n=10
t=0.8
split = int(t*len(stock))

#initiate mixture model
model = mix.GaussianMixture(n_components=2, 
                            covariance_type='spherical',
                            n_init=100,
                            random_state=0)

#fit the model to the data
scaler = StandardScaler()
model.fit(np.reshape(scaler.fit_transform(stock[:split]), (-1, stock.shape[1])))
regime = model.predict(np.reshape(scaler.fit_transform(stock[split:]), (-1, stock.shape[1])))

#create the regimes dataframe and add othe required columns
Regimes = pd.DataFrame(regime, columns=['Regime'], index=stock[split:].index)\
            .join(stock[split:], how='inner')\
                .assign(market_cu_return=stock[split:]\
                        .Return.cumsum())\
                        .reset_index(drop=False)\
                        .rename(columns={'index':'Date'})
   
#visualize the GMM model's results                     
order = [0, 1]
fig = sns.FacetGrid(data=Regimes, hue='Regime', hue_order=order,
                    aspect=2, height=4)
fig.map(plt.scatter, 'Date', 'market_cu_return', s=4).add_legend()
plt.show()

#print mean and covariance of each regime
for i in order:
    print('Mean for regime %i: '%i, model.means_[i][0])
    print('Covariance for regime %i: '%i, (model.covariances_[i]))
    
#scaling the Regimes df to train SVC and create signal column    
columns = Regimes.columns.drop(['Regime', 'Date'])
Regimes[columns] = scaler.fit_transform(Regimes[columns])
Regimes['Signal'] = 0
Regimes.loc[Regimes['Return'] > 0, 'Signal'] = 1
Regimes.loc[Regimes['Return'] < 0, 'Signal'] = -1

#initiate classifier
classifier = SVC(C=2.0, cache_size=100, class_weight=None, coef0=0.0,
                decision_function_shape=None, degree=2, gamma='auto',
                kernel='rbf', max_iter=-1, probability=False,
                random_state=None, shrinking=True, tol=0.001,
                verbose=False)

#split and fit the classifier
split2 = int(0.8 * len(Regimes))

X = Regimes.drop(['Signal', 'Return', 'market_cu_return', 'Date'], axis=1)
y = Regimes['Signal']

classifier.fit(X[:split2], y[:split2])

#----------
p_data = len(X)-split2

stock['Pred_Signal'] = 0
stock.iloc[-p_data: , stock.columns.get_loc('Pred_Signal')] = classifier.predict(X[split2:])

print(stock['Pred_Signal'][-p_data:])
stock['str_ret'] = stock['Pred_Signal'] * stock['Return'].shift(-1)
stock.fillna(0, inplace=True)

stock['strategy_cu_return'] = 0
stock['market_cu_return'] = 0
stock.iloc[-p_data: , stock.columns.get_loc('strategy_cu_return')]\
    = np.nancumsum(stock['str_ret'][-p_data:])
stock.iloc[-p_data: , stock.columns.get_loc('market_cu_return')]\
    = np.nancumsum(stock['Return'][-p_data:])
Sharpe = (stock['strategy_cu_return'][-1] - stock['market_cu_return'][-1])\
    / np.nanstd(stock['strategy_cu_return'][-p_data:])
    
plt.plot(stock['strategy_cu_return'][-p_data:], color='g', label='Strategy Returns')
plt.plot(stock['market_cu_return'][-p_data:], color='r', label='Market Returns')
plt.figtext(0.14, 0.9, s='Sharpe ratio: %.2f'%Sharpe)
plt.legend(loc='best')
plt.show()


# =============================================================================
# Alternative model
# =============================================================================
stock = pd.DataFrame(stocks['Volodymyr'])
stock['Return'] = stock.pct_change()
stock.fillna(0, inplace=True)

#initiate mixture model
model = mix.GaussianMixture(n_components=2, 
                            covariance_type='spherical',
                            n_init=100,
                            random_state=0)

#fit the model to the data
scaler = StandardScaler()
model.fit(np.reshape(scaler.fit_transform(stock), (-1, stock.shape[1])))
regime = model.predict(np.reshape(scaler.fit_transform(stock), (-1, stock.shape[1])))


#create the regimes dataframe and add othe required columns
Regimes = pd.DataFrame(regime, columns=['Regime'], index=stock.index)\
            .join(stock, how='inner')\
                        .reset_index(drop=False)\
                        .rename(columns={'index':'Date'})
                        
#Cumulative return
Regimes['market_cu_return'] = (Regimes['Return'] + 1).cumprod() - 1
   
#visualize the GMM model's results                     
order = [0, 1]
fig = sns.FacetGrid(data=Regimes, hue='Regime', hue_order=order,
                    aspect=2, height=4)
fig.map(plt.scatter, 'Date', 'market_cu_return', s=4).add_legend()
plt.show()

#print mean and covariance of each regime
for i in order:
    print('Mean for regime %i: '%i, model.means_[i][0])
    print('Covariance for regime %i: '%i, (model.covariances_[i]))
    
#scaling the Regimes df to train SVC and create signal column    
columns = Regimes.columns.drop(['Regime', 'Date'])
Regimes[columns] = scaler.fit_transform(Regimes[columns])
Regimes['Signal'] = 0
Regimes.loc[Regimes['Regime'] == 0, 'Signal'] = -1
Regimes.loc[Regimes['Regime'] == 1, 'Signal'] = 1

#Testing-------------------------------------------------------------
Regimes['str_ret'] = Regimes['Signal'] * Regimes['Return'].shift(-1)
Regimes.fillna(0, inplace=True)

Regimes['strategy_cu_return'] = (Regimes['str_ret'].cumsum())

    
plt.plot(Regimes['strategy_cu_return'], color='g', label='Strategy Returns')
plt.plot(Regimes['market_cu_return'], color='r', label='Market Returns')
plt.legend(loc='best')
plt.show()


#--------------------------------------------------------------------
