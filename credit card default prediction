import pandas as pd
default = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/Credit%20Default.csv')
default.head()
default.info()
default.describe()
default['Default'].value_counts()
default.columns
y = default['Default']
X = default.drop(['Default'],axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, random_state=2529)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
model.intercept_
model.coef_
y_pred = model.predict(X_test)
y_pred
