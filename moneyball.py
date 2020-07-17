import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
import sklearn
from sklearn import linear_model
df = pd.read_csv('baseball.csv')
df.head()

flatui = ["#6cdae7", "#fd3a4a", "#ffaa1d", "#ff23e5", "#34495e", "#2ecc71"]
sns.set_palette(flatui)
#sns.palplot(sns.color_palette())

sns.lmplot(x = "W", y = "RS", fit_reg = False, hue = "Playoffs", data=df,height=7, aspect=1.25)
plt.xlabel("Wins", fontsize = 20)
plt.ylabel("Runs Scored", fontsize = 20)
plt.axvline(99, 0, 1, color = "Black", ls = '--')
plt.show()

x = np.array(df.RD)
y = np.array(df.W)
slope, intercept = np.polyfit(x, y, 1)
abline_values = [slope * i + intercept for i in x]
plt.figure(figsize=(10,8))
plt.scatter(x, y)
plt.plot(x, abline_values, 'r')
plt.title("Slope = %s" % (slope), fontsize = 12)
plt.xlabel("Run Difference", fontsize =20)
plt.ylabel("Wins", fontsize = 20)
plt.axhline(99, 0, 1, color = "k", ls = '--')
plt.show()

corrcheck = df[['RD', 'W', 'Playoffs']].copy()
g = sns.pairplot(corrcheck, hue = 'Playoffs',vars=["RD", "W"])
g.fig.set_size_inches(14,10)

corrcheck.corr(method='pearson')
podesta = df[['OBP','SLG','BA','RS']]
podesta.corr(method='pearson')
moneyball = df.dropna()

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X = moneyball.iloc[:,6:9]  #independent columns
y = moneyball.iloc[:,-1]    #target column
from sklearn.ensemble import ExtraTreesClassifier
 
model = ExtraTreesClassifier()
model.fit(X,y)
# print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(3).plot(kind='barh', figsize = (12,8))
plt.xlabel("Importance", fontsize = 20)
plt.ylabel("Statistic", fontsize = 20)
plt.show()

x = df[['OBP','SLG']].values
y = df[['RS']].values 
Runs = linear_model.LinearRegression() 
Runs.fit(x,y)
# print(Runs.intercept_) 
# print(Runs.coef_)

x = moneyball[['OOBP','OSLG']].values
y = moneyball[['RA']].values
RunsAllowed = linear_model.LinearRegression()
RunsAllowed.fit(x,y)
 
# print(RunsAllowed.intercept_)
# print(RunsAllowed.coef_)

x = moneyball[['RD']].values
y = moneyball[['W']].values
Wins = linear_model.LinearRegression()
Wins.fit(x,y)
 
# print(Wins.intercept_)
# print(Wins.coef_)

print(Runs.predict([[0.339,0.430]]))
print(RunsAllowed.predict([[0.307,0.373]]))
print(Wins.predict([[177]]))
