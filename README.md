# DVD Rental Duration Prediction

This project aims to predict the number of days a customer will rent a DVD for a DVD rental company based on various features. The company requires a regression model with a Mean Squared Error (MSE) of 3 or less on a test set to help them with efficient inventory planning.

## Dataset

The dataset used for this project is `rental_info.csv`, which contains the following features:

- `rental_date`: The date (and time) the customer rents the DVD.
- `return_date`: The date (and time) the customer returns the DVD.
- `amount`: The amount paid by the customer for renting the DVD.
- `amount_2`: The square of `amount`.
- `rental_rate`: The rate at which the DVD is rented.
- `rental_rate_2`: The square of `rental_rate`.
- `release_year`: The year the movie being rented was released.
- `length`: Length of the movie being rented, in minutes.
- `length_2`: The square of `length`.
- `replacement_cost`: The amount it will cost the company to replace the DVD.
- `special_features`: Any special features, such as trailers or deleted scenes, that the DVD has.
- `NC-17`, `PG`, `PG-13`, `R`: Dummy variables indicating the rating of the movie. The reference dummy has already been dropped.

## Requirements

- Python 3.x
- Pandas
- NumPy
- Scikit-learn

## Project Structure

─ rental_info.csv # The dataset
─ rental_duration_prediction.ipynb # Jupyter notebook with the code
─ README.md # This file
─ requirements.txt # Required Python packages

## Data Preprocessing
1. Convert the `rental_date` and `return_date` columns to datetime format.
2. Calculate the rental length in days.
3. Create dummy variables for special features:
   - `deleted_scenes`: 1 if the special feature includes "Deleted Scenes", otherwise 0.
   - `behind_the_scenes`: 1 if the special feature includes "Behind the Scenes", otherwise 0.
4. Drop unnecessary columns (`rental_date`, `return_date`, `special_features`).
## Model Training and Evaluation
1. Separate the features (`X`) and target variable (`y`).
2. Standardize the features using `StandardScaler`.
3. Split the data into training and testing sets.
4. Define a set of regression models:
   - Linear Regression
   - Ridge Regression
   - Lasso Regression
   - Decision Tree Regressor
   - Random Forest Regressor
   - Gradient Boosting Regressor
5. Perform hyperparameter tuning using `GridSearchCV` to find the best parameters for each model.
6. Evaluate each model on the test set.
7. Identify and save the best model based on the lowest MSE.
  
## Results

The results of the model evaluations on the test set are as follows:

- Ridge Regression MSE: 2.941870616734153
- Linear Regression MSE: 2.9417238646975976
- Lasso Regression MSE: 2.9417116642920518
- Decision Tree Regressor MSE: 2.165347833153885
- Random Forest Regressor MSE: 2.031285398706584
- Gradient Boosting Regressor MSE: 2.06977259149959

The best model for predicting the rental duration is `RandomForestRegressor` with an MSE of `2.031285398706584`.

## Conclusion

The `RandomForestRegressor` model was found to be the best model for predicting the rental duration with an MSE of 2.031285398706584, which is well below the required MSE of 3.

## Usage

To use the model, run the Jupyter notebook `rental_duration_prediction.ipynb` which contains all the necessary code for data preprocessing, model training, and evaluation.

## Author

Denis Evmenenko

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
