# Linear Regression Problem in Python
## Instructions:
1. **Identify dependent and independent variables**: The code identifies `X` and `y` as the independent and dependent variables, respectively.

2. **Perform exploratory data analysis on the data**:
   - **Draw a scatter plot**: The code uses `matplotlib` to create a scatter plot of the independent and dependent variables.
   - **Data cleaning**: The code includes a placeholder for data cleaning, which you would need to fill in based on your specific dataset.
   - **Split the dataset**: The code uses `train_test_split()` from `sklearn.model_selection` to split the dataset into training and testing sets.

3. **Import `LinearRegression` from `sklearn` library and train the model**: The code imports the `LinearRegression` model from `sklearn.linear_model`, fits the model using the training data, and makes predictions on the test data.

4. **Show the following performance metrics of your model**:
   - **Coefficients**: The code prints the coefficients of the model.
   - **Mean Squared Error (MSE)**: The code calculates and prints the MSE using `mean_squared_error` from `sklearn.metrics`.
   - **r_2 score**: The code calculates and prints the R-squared score using `r2_score` from `sklearn.metrics`.
