# Car_dataset_linear_Regression_In_Python
Implementing linear Regression technique in Python

I used a car data set and perform the linear Regression technique to predict the Mileage per gallon of a car.
I want to perform the linear Regression model in Car data set to predict the Mileage per gallon of a car based on some other features and optimize the model using a stepwise 
feature selection technique. 

In this study, I performed exploratory data analysis to get a better understanding of the data, and then I examined the Linear Regression to predict the Mileage per gallon of new cars. I 
discovered that there is a relationship between mpg and other variables. Also, I found that There is a strong positive correlation between displacement, horsepower, weight, and cylinders, which 
violates the non-multicollinearity assumption of linear regression. I used a feature selection method to eliminate the effect of multicollinearity in the data set. Next, I performed a linear Regression 
model using all variables with 81% accuracy but the p-values of several independent variables were much higher than 5%. Hence, I performed a forward stepwise method to discover the most important 
variables. The result showed Displacement, Weight, Model_Year, and US_Made_1 were variables that had the most significant impact on mpg. Finally, I performed the model using the most impact 
features. the result showed the accuracy of the model was 83% and increased by 2% and the AIC prediction error decreased as well.
