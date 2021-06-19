STEP 1 - Importing the dataset
In this step, we will import the dataset through the link with the help of pandas library and then we will observe the data

# Importing all the required libraries
​
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
%matplotlib inline
import seaborn as sns 
​
# To ignore the warnings 
import warnings as wg
wg.filterwarnings("ignore")
# Reading data from remote link
​
url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
df = pd.read_csv(url)
# now let's observe the dataset 
df.head()
df.tail()
# To find the number of columns and rows 
df.shape
# To find more information about our dataset
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 25 entries, 0 to 24
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   Hours   25 non-null     float64
 1   Scores  25 non-null     int64  
dtypes: float64(1), int64(1)
memory usage: 528.0 bytes
df.describe()
# now we will check if our dataset contains null or missings values  
df.isnull().sum()
Step 2 - Visualizing the dataset
# Plotting the dataset
plt.rcParams["figure.figsize"] = [16,9]
df.plot(x='Hours', y='Scores', style='.', color='Red', markersize=10)
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.grid()
plt.show()

# we can also use .corr to determine the corelation between the variables 
df.corr()
Step 3 - Data preparation¶
df.head()
# using iloc function we will divide the data 
X = df.iloc[:, :1].values  
y = df.iloc[:, 1:].values
​
X
y
# Splitting data into training and testing data
​
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0)
Step 4 - Training the Algorithm
​
from sklearn.linear_model import LinearRegression  
​
model = LinearRegression()  
model.fit(X_train, y_train)
Step 5 - Visualizing the model
​
line = model.coef_*X + model.intercept_
​
# Plotting for the training data
plt.rcParams["figure.figsize"] = [16,9]
plt.scatter(X_train, y_train, color='blue')
plt.plot(X, line, color='green');
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score') 
plt.grid()
plt.show()

​
# Plotting for the testing data
plt.rcParams["figure.figsize"] = [16,9]
plt.scatter(X_test, y_test, color='blue')
plt.plot(X, line, color='green');
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score') 
plt.grid()
plt.show()

Step 6 - Making Predictions
print(X_test) # Testing data - In Hours
y_pred = model.predict(X_test) # Predicting the scores
[[1.5]
 [3.2]
 [7.4]
 [2.5]
 [5.9]]
# Comparing Actual vs Predicted
​
y_test
y_pred
# Comparing Actual vs Predicted
comp = pd.DataFrame({ 'Actual':[y_test],'Predicted':[y_pred] })
comp
​
# Testing with your own data
​
hours = 9.25
own_pred = model.predict([[hours]])
print("The predicted score if a person studies for",hours,"hours is",own_pred[0])
The predicted score if a person studies for 9.25 hours is [93.69173249]
Step 7 - Evaluating the model
from sklearn import metrics  
​
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
​
Mean Absolute Error: 4.183859899002975
