```python
# import libraries

import pandas as pd
import numpy as np

from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm
from scipy import stats
                                                                                        

import seaborn as sns
import matplotlib.pyplot as plt

```


```python
#Load data

df_car = pd.read_csv('D:/Mahdieh_CourseUniversity/University_courses/ALY6020/Module_2/Project/car.csv')

```

Exploratory data analysis


```python
# check shape

df_car.shape
```




    (398, 8)




```python
#descriptive analysis 

df_car.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MPG</th>
      <th>Cylinders</th>
      <th>Displacement</th>
      <th>Weight</th>
      <th>Acceleration</th>
      <th>Model Year</th>
      <th>US Made</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>398.000000</td>
      <td>398.000000</td>
      <td>398.000000</td>
      <td>398.000000</td>
      <td>398.000000</td>
      <td>398.000000</td>
      <td>398.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>23.514573</td>
      <td>5.454774</td>
      <td>193.425879</td>
      <td>2970.424623</td>
      <td>15.568090</td>
      <td>76.010050</td>
      <td>0.625628</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.815984</td>
      <td>1.701004</td>
      <td>104.269838</td>
      <td>846.841774</td>
      <td>2.757689</td>
      <td>3.697627</td>
      <td>0.484569</td>
    </tr>
    <tr>
      <th>min</th>
      <td>9.000000</td>
      <td>3.000000</td>
      <td>68.000000</td>
      <td>1613.000000</td>
      <td>8.000000</td>
      <td>70.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>17.500000</td>
      <td>4.000000</td>
      <td>104.250000</td>
      <td>2223.750000</td>
      <td>13.825000</td>
      <td>73.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>23.000000</td>
      <td>4.000000</td>
      <td>148.500000</td>
      <td>2803.500000</td>
      <td>15.500000</td>
      <td>76.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>29.000000</td>
      <td>8.000000</td>
      <td>262.000000</td>
      <td>3608.000000</td>
      <td>17.175000</td>
      <td>79.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>46.600000</td>
      <td>8.000000</td>
      <td>455.000000</td>
      <td>5140.000000</td>
      <td>24.800000</td>
      <td>82.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Check type of variables and find null values                                                                                                                                               

df_car.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 398 entries, 0 to 397
    Data columns (total 8 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   MPG           398 non-null    float64
     1   Cylinders     398 non-null    int64  
     2   Displacement  398 non-null    float64
     3   Horsepower    398 non-null    object 
     4   Weight        398 non-null    int64  
     5   Acceleration  398 non-null    float64
     6   Model Year    398 non-null    int64  
     7   US Made       398 non-null    int64  
    dtypes: float64(3), int64(4), object(1)
    memory usage: 25.0+ KB
    


```python
#check missing values

df_car.isnull().sum()
```




    MPG             0
    Cylinders       0
    Displacement    0
    Horsepower      0
    Weight          0
    Acceleration    0
    Model Year      0
    US Made         0
    dtype: int64




```python
#check missing values

df_car.isnull().any()
```




    MPG             False
    Cylinders       False
    Displacement    False
    Horsepower      False
    Weight          False
    Acceleration    False
    Model Year      False
    US Made         False
    dtype: bool




```python
#check missing values

plt.figure(figsize=(6,4))
sns.heatmap(df_car.isnull())
```




    <AxesSubplot:>




    
![png](output_8_1.png)
    



```python
# check unique variables for abnormal values

df_car.Horsepower.unique() # In th third line of our output, we see that we have '?' as a value. it seem there are abnormal values.
#We can either try to impute by using the median value of "horsepower", or delete the observations where "horsepower" has '?' as a value. 

df_car[df_car['Horsepower'] =='?']

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MPG</th>
      <th>Cylinders</th>
      <th>Displacement</th>
      <th>Horsepower</th>
      <th>Weight</th>
      <th>Acceleration</th>
      <th>Model Year</th>
      <th>US Made</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32</th>
      <td>25.0</td>
      <td>4</td>
      <td>98.0</td>
      <td>?</td>
      <td>2046</td>
      <td>19.0</td>
      <td>71</td>
      <td>1</td>
    </tr>
    <tr>
      <th>126</th>
      <td>21.0</td>
      <td>6</td>
      <td>200.0</td>
      <td>?</td>
      <td>2875</td>
      <td>17.0</td>
      <td>74</td>
      <td>1</td>
    </tr>
    <tr>
      <th>330</th>
      <td>40.9</td>
      <td>4</td>
      <td>85.0</td>
      <td>?</td>
      <td>1835</td>
      <td>17.3</td>
      <td>80</td>
      <td>0</td>
    </tr>
    <tr>
      <th>336</th>
      <td>23.6</td>
      <td>4</td>
      <td>140.0</td>
      <td>?</td>
      <td>2905</td>
      <td>14.3</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>354</th>
      <td>34.5</td>
      <td>4</td>
      <td>100.0</td>
      <td>?</td>
      <td>2320</td>
      <td>15.8</td>
      <td>81</td>
      <td>0</td>
    </tr>
    <tr>
      <th>374</th>
      <td>23.0</td>
      <td>4</td>
      <td>151.0</td>
      <td>?</td>
      <td>3035</td>
      <td>20.5</td>
      <td>82</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# check unique variables for abnormal values
#Since there aren't many, I chose to delete them

df_car = df_car[df_car.Horsepower != '?']

```


```python
# the variable "horsepower" has a type "object" hence change the type of variable horsepower 

df_car.Horsepower = df_car.Horsepower.astype('float')

```


```python
#change the variables name - Model Year and Us made 

df_car = df_car.rename(columns ={'Model Year': 'Model_Year','US Made': 'US_Made'})

df_car
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MPG</th>
      <th>Cylinders</th>
      <th>Displacement</th>
      <th>Horsepower</th>
      <th>Weight</th>
      <th>Acceleration</th>
      <th>Model_Year</th>
      <th>US_Made</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130.0</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165.0</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150.0</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>8</td>
      <td>304.0</td>
      <td>150.0</td>
      <td>3433</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>8</td>
      <td>302.0</td>
      <td>140.0</td>
      <td>3449</td>
      <td>10.5</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>393</th>
      <td>27.0</td>
      <td>4</td>
      <td>140.0</td>
      <td>86.0</td>
      <td>2790</td>
      <td>15.6</td>
      <td>82</td>
      <td>1</td>
    </tr>
    <tr>
      <th>394</th>
      <td>44.0</td>
      <td>4</td>
      <td>97.0</td>
      <td>52.0</td>
      <td>2130</td>
      <td>24.6</td>
      <td>82</td>
      <td>0</td>
    </tr>
    <tr>
      <th>395</th>
      <td>32.0</td>
      <td>4</td>
      <td>135.0</td>
      <td>84.0</td>
      <td>2295</td>
      <td>11.6</td>
      <td>82</td>
      <td>1</td>
    </tr>
    <tr>
      <th>396</th>
      <td>28.0</td>
      <td>4</td>
      <td>120.0</td>
      <td>79.0</td>
      <td>2625</td>
      <td>18.6</td>
      <td>82</td>
      <td>1</td>
    </tr>
    <tr>
      <th>397</th>
      <td>31.0</td>
      <td>4</td>
      <td>119.0</td>
      <td>82.0</td>
      <td>2720</td>
      <td>19.4</td>
      <td>82</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>392 rows × 8 columns</p>
</div>




```python
# check distribution

df_car.hist()
fig=plt.gcf()
fig.set_size_inches(13,13)
plt.show()
```


    
![png](output_13_0.png)
    



```python
# check outliers
plt.figure(figsize=(12,6))
df_car.boxplot()
```




    <AxesSubplot:>




    
![png](output_14_1.png)
    



```python
# Pair scatter plot

sns.pairplot(df_car ,  height=1.70 , diag_kind='kde').add_legend()
```




    <seaborn.axisgrid.PairGrid at 0x225607b0a30>




    
![png](output_15_1.png)
    



```python
    # check for variables correlation

corr = df_car.corr()
corr
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MPG</th>
      <th>Cylinders</th>
      <th>Displacement</th>
      <th>Horsepower</th>
      <th>Weight</th>
      <th>Acceleration</th>
      <th>Model_Year</th>
      <th>US_Made</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MPG</th>
      <td>1.000000</td>
      <td>-0.777618</td>
      <td>-0.805127</td>
      <td>-0.778427</td>
      <td>-0.832244</td>
      <td>0.423329</td>
      <td>0.580541</td>
      <td>-0.565161</td>
    </tr>
    <tr>
      <th>Cylinders</th>
      <td>-0.777618</td>
      <td>1.000000</td>
      <td>0.950823</td>
      <td>0.842983</td>
      <td>0.897527</td>
      <td>-0.504683</td>
      <td>-0.345647</td>
      <td>0.610494</td>
    </tr>
    <tr>
      <th>Displacement</th>
      <td>-0.805127</td>
      <td>0.950823</td>
      <td>1.000000</td>
      <td>0.897257</td>
      <td>0.932994</td>
      <td>-0.543800</td>
      <td>-0.369855</td>
      <td>0.655936</td>
    </tr>
    <tr>
      <th>Horsepower</th>
      <td>-0.778427</td>
      <td>0.842983</td>
      <td>0.897257</td>
      <td>1.000000</td>
      <td>0.864538</td>
      <td>-0.689196</td>
      <td>-0.416361</td>
      <td>0.489625</td>
    </tr>
    <tr>
      <th>Weight</th>
      <td>-0.832244</td>
      <td>0.897527</td>
      <td>0.932994</td>
      <td>0.864538</td>
      <td>1.000000</td>
      <td>-0.416839</td>
      <td>-0.309120</td>
      <td>0.600978</td>
    </tr>
    <tr>
      <th>Acceleration</th>
      <td>0.423329</td>
      <td>-0.504683</td>
      <td>-0.543800</td>
      <td>-0.689196</td>
      <td>-0.416839</td>
      <td>1.000000</td>
      <td>0.290316</td>
      <td>-0.258224</td>
    </tr>
    <tr>
      <th>Model_Year</th>
      <td>0.580541</td>
      <td>-0.345647</td>
      <td>-0.369855</td>
      <td>-0.416361</td>
      <td>-0.309120</td>
      <td>0.290316</td>
      <td>1.000000</td>
      <td>-0.136065</td>
    </tr>
    <tr>
      <th>US_Made</th>
      <td>-0.565161</td>
      <td>0.610494</td>
      <td>0.655936</td>
      <td>0.489625</td>
      <td>0.600978</td>
      <td>-0.258224</td>
      <td>-0.136065</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# visualize correlation

plt.figure(figsize=(8,6))
sns.heatmap(df_car.corr() , annot= True , cmap='Blues')
plt.show()
```


    
![png](output_17_0.png)
    



There are four assumptions associated with a linear regression model:

1- Linearity: The relationship between X and the mean of Y is linear.

2- Homoscedasticity: The variance of residual is the same for any value of X.

3- Independence: Observations are independent of each other.

4- Normality: For any fixed value of X, Y is normally distributed.


We can see that there is a problem of multicollinearity in our data since some of the variables.we can also see clearly that the displacement,horsepower,weight,and cylinders have a strong positive correlations between themselves and they are the cause of the multicollinearity as shown in the correlation heatmap:



Perform Linear Regression



```python
# change US Made variable to dummy

df_car = pd.get_dummies(df_car , columns= ['US_Made'])

```


```python
# indicate predictor and predicted variables

y = df_car['MPG']

x = df_car.iloc[: , 1:9]
```


```python
# split our data into training and testing data

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
```


```python
#form linear regression with all variables 

x2 = sm.add_constant(x_train)
model = sm.OLS(y_train,x2)
model_2 = model.fit()
print(model_2.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                    MPG   R-squared:                       0.818
    Model:                            OLS   Adj. R-squared:                  0.813
    Method:                 Least Squares   F-statistic:                     195.4
    Date:                Tue, 24 Jan 2023   Prob (F-statistic):          1.05e-108
    Time:                        21:51:42   Log-Likelihood:                -822.07
    No. Observations:                 313   AIC:                             1660.
    Df Residuals:                     305   BIC:                             1690.
    Df Model:                           7                                         
    Covariance Type:            nonrobust                                         
    ================================================================================
                       coef    std err          t      P>|t|      [0.025      0.975]
    --------------------------------------------------------------------------------
    const          -12.3333      3.479     -3.545      0.000     -19.180      -5.487
    Cylinders       -0.2944      0.383     -0.768      0.443      -1.048       0.459
    Displacement     0.0174      0.009      2.004      0.046       0.000       0.034
    Horsepower      -0.0164      0.015     -1.073      0.284      -0.046       0.014
    Weight          -0.0062      0.001     -8.750      0.000      -0.008      -0.005
    Acceleration     0.0297      0.111      0.267      0.790      -0.189       0.249
    Model_Year       0.7969      0.058     13.776      0.000       0.683       0.911
    US_Made_0       -4.5894      1.732     -2.649      0.008      -7.998      -1.181
    US_Made_1       -7.7439      1.792     -4.322      0.000     -11.270      -4.218
    ==============================================================================
    Omnibus:                       21.113   Durbin-Watson:                   2.074
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               29.851
    Skew:                           0.488   Prob(JB):                     3.30e-07
    Kurtosis:                       4.157   Cond. No.                     5.12e+18
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is 1.19e-28. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    

Optimize the model using selection techniques:

1- drop variables that have high multicollinearity

2- use the stepwise forward selection technique to find the most important features


```python
# perform model with dropping variables with high multicollinearity

x = df_car[['Weight','Model_Year', 'US_Made_1','US_Made_1']]
y = df_car['MPG']

# split our data into training and testing data

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

#form linear regression with all variables 

x2 = sm.add_constant(x_train)
model = sm.OLS(y_train,x2)
model_2 = model.fit()
print(model_2.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                    MPG   R-squared:                       0.810
    Model:                            OLS   Adj. R-squared:                  0.807
    Method:                 Least Squares   F-statistic:                     382.7
    Date:                Tue, 24 Jan 2023   Prob (F-statistic):           6.72e-97
    Time:                        21:51:03   Log-Likelihood:                -722.69
    No. Observations:                 274   AIC:                             1453.
    Df Residuals:                     270   BIC:                             1468.
    Df Model:                           3                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const        -18.0413      4.911     -3.674      0.000     -27.709      -8.373
    Weight        -0.0058      0.000    -17.986      0.000      -0.006      -0.005
    Model_Year     0.7949      0.061     13.132      0.000       0.676       0.914
    US_Made_1     -1.2419      0.265     -4.679      0.000      -1.764      -0.719
    US_Made_1     -1.2419      0.265     -4.679      0.000      -1.764      -0.719
    ==============================================================================
    Omnibus:                       24.877   Durbin-Watson:                   2.098
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               46.723
    Skew:                           0.502   Prob(JB):                     7.15e-11
    Kurtosis:                       4.757   Cond. No.                     2.46e+19
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is 4.27e-30. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    


```python
# Perform stepwise regression to optimize the model

lreg = LinearRegression()
sfs1 = sfs(lreg, k_features= 4, forward=True, verbose=2, scoring='neg_mean_squared_error')

sfs1 = sfs1.fit(x, y)

feat_names = list(sfs1.k_feature_names_)
print(feat_names)
```

    ['Weight', 'Model_Year', 'US_Made_1', 'Displacement']
    

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   4 out of   4 | elapsed:    0.0s finished
    
    [2023-01-24 21:50:11] Features: 1/4 -- score: -25.02282189317482[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   3 out of   3 | elapsed:    0.0s finished
    
    [2023-01-24 21:50:11] Features: 2/4 -- score: -15.211394485226005[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   2 out of   2 | elapsed:    0.0s finished
    
    [2023-01-24 21:50:11] Features: 3/4 -- score: -14.869679673926004[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s finished
    
    [2023-01-24 21:50:11] Features: 4/4 -- score: -14.651921235002348


```python
# perform model with selected features

x = df_car[['Weight','Model_Year', 'US_Made_1','Displacement']]
y = df_car['MPG']

# split our data into training and testing data

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

#form linear regression with all variables 

x2 = sm.add_constant(x_train)
model = sm.OLS(y_train,x2)
model_2 = model.fit()
print(model_2.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                    MPG   R-squared:                       0.834
    Model:                            OLS   Adj. R-squared:                  0.831
    Method:                 Least Squares   F-statistic:                     337.1
    Date:                Tue, 24 Jan 2023   Prob (F-statistic):          1.83e-103
    Time:                        21:39:37   Log-Likelihood:                -700.23
    No. Observations:                 274   AIC:                             1410.
    Df Residuals:                     269   BIC:                             1429.
    Df Model:                           4                                         
    Covariance Type:            nonrobust                                         
    ================================================================================
                       coef    std err          t      P>|t|      [0.025      0.975]
    --------------------------------------------------------------------------------
    const          -15.4035      4.416     -3.488      0.001     -24.098      -6.709
    Weight          -0.0072      0.001    -11.059      0.000      -0.009      -0.006
    Model_Year       0.7826      0.057     13.803      0.000       0.671       0.894
    US_Made_1       -2.5536      0.546     -4.674      0.000      -3.629      -1.478
    Displacement     0.0131      0.006      2.212      0.028       0.001       0.025
    ==============================================================================
    Omnibus:                       20.266   Durbin-Watson:                   1.817
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               33.089
    Skew:                           0.458   Prob(JB):                     6.53e-08
    Kurtosis:                       4.435   Cond. No.                     7.29e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 7.29e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.
    


```python

```
