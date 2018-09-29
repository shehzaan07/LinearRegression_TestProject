##Default imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics 


df1 = pd.read_csv('train_class.csv')
df2 = pd.read_csv('test_class_1.csv')

#missing data observing in columns
total = df1.isnull().sum(axis=0).sort_values(ascending=False)
percent = ((df1.isnull().sum(axis=0)/df1.isnull().count(axis=0))*100).sort_values(ascending=False)


#count the number of null values in the column and their percentage of the total data
missing_data_columns = pd.concat([total,percent],axis=1, keys=['Total', 'Percent'])
missing_data_columns


#delete empty column
newdf1 = df1.drop(['Unnamed: 0','PoolQC', 'MiscFeature', 'Alley', 'Fence','FireplaceQu'], axis=1)


# Doing EDA for numerical variables
data_type = newdf1.dtypes
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numericaldf = newdf1.select_dtypes(include=numerics)
num_col_high_corr_to_y = numericaldf.loc[:,numericaldf.corr()['SalePrice']>0.25]
num_col_high_corr_to_y = num_col_high_corr_to_y.fillna(num_col_high_corr_to_y.mean())



# Doing EDA for categorical variables
category = ['object']
categorydf = newdf1.select_dtypes(include=category)
category_var = categorydf.fillna('Missing') #Instead of imputing mode we created a new category as 'Missing'

#Imputed dummies in the categorical data
cat_col_imp = pd.get_dummies(category_var)

#Concatinate numerical variables and categorical variables
main_df = pd.concat([cat_col_imp, num_col_high_corr_to_y], axis=1)
main_corr_matrix_df = main_df.loc[:,main_df.corr()['SalePrice']>0.4]


X = main_corr_matrix_df.drop(['SalePrice'], axis=1)
Y = main_corr_matrix_df['SalePrice']

#Splitting the dataset into train(80%) and test(20%)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

#Creating an object
regressor = LinearRegression()

#Implementing Linear Regression
regressor.fit(X_train,Y_train)
Y_pred = regressor.predict(X_test)
plt.scatter(Y_test,Y_pred)
print("MAE:",metrics.mean_absolute_error(Y_test,Y_pred))
print("MSE:",metrics.mean_squared_error(Y_test,Y_pred))
print("RMSE:",np.sqrt(metrics.mean_squared_error(Y_test,Y_pred)))