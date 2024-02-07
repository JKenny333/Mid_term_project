#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Defining functions

def data_frame_overview(data_frame):
    print(f'Column names: \n {data_frame.columns}\n')
    print(f'Dimensions: {data_frame.shape}\n')
    print(data_frame.info())
    return data_frame.head(10)

def format_column_names(data_frame, column_name_mapping = {}):
    '''
    Formats column names in a DataFrame based on a provided mapping.

    Parameters:
    data_frame: The DataFrame to format.
    column_mapping (dict): A dictionary containing old column names as keys and new names as values.

    Returns:
    None (modifies the DataFrame in place).
    '''
    # Perform additional formatting (lowercase, strip and replace spaces with underscores)
    data_frame.columns = [name.strip().replace(" ", "_").lower() for name in data_frame.columns]

    # Iterate through the provided column_name_mapping dictionary
    for old_name, new_name in column_name_mapping.items():
        # Check if the old column name exists in the DataFrame
        if old_name in data_frame.columns:
            # Rename the column with the new name
            data_frame.rename(columns={old_name: new_name}, inplace=True)
    print(f'New column names: \n {data_frame.columns}')

            
def null_check(data_frame):
    print(f'Total null values per row: \n{data_frame.isnull().sum(axis=1)}\n')
    print(f'Total null values per column: \n{data_frame.isnull().sum()}\n')
    
    
def dropna_rows_cols(data_frame, row_thresh, col_thresh):
    '''
    removes rows and columns with null values
    
    parameters:
    data_frame: data_frame from which to remove rows and columns
    row_thresh: minimum threshold for the number of non-null values that a row must have in order to be kept
    col_thresh: minimum threshold for the number of non-null values that a column must have in order to be kept
    
    returns:
    data frame in which the rows and columns containing null values have been removed
    '''
    rows_before = len(data_frame)
    cols_before = len(data_frame.columns)
    data_frame.dropna(thresh = row_thresh, inplace = True)
    data_frame.dropna(axis=1, thresh = col_thresh, inplace = True)
    rows_after = len(data_frame)
    cols_after = len(data_frame.columns)
    rows_deleted = rows_before - rows_after
    cols_deleted = cols_before - cols_after
    print(f'Deleted {rows_deleted} rows')
    print(f'Deleted {cols_deleted} columns')


def dup_check(data_frame):
    print(f'Duplicates found: {data_frame.duplicated().any()}\n')
    print(f'Number of duplicates: {data_frame.duplicated().sum()}\n')
    
def drop_dup_reset(data_frame):
    rows_before = len(data_frame)
    data_frame.drop_duplicates(inplace=True)
    data_frame.reset_index(drop=True, inplace=True)
    rows_after = len(data_frame)
    num_dups_deleted = rows_before - rows_after
    print(f'Deleted {num_dups_deleted} duplicates')

def clean_sex_column(data_frame):
    # Store the original 'sex' column for comparison
    original_sex_column = data_frame['sex'].copy()

    # Clean the 'sex' column
    data_frame['sex'] = data_frame['sex'].str.strip().str.lower()
    data_frame['sex'].replace({'male': 'm', 'female': 'f', 'femal': 'f'}, inplace=True)

    # Define a set of valid values
    valid_values = {'m', 'f'}

    # Replace invalid entries with NaN
    data_frame['sex'] = data_frame['sex'].apply(lambda x: x if x in valid_values else pd.NA)

    # Calculate the number of changed values
    changes = (original_sex_column.str.strip().str.lower() != data_frame['sex']).sum()
    print(f"Number of values changed in the 'sex' column: {changes}")
    print(data_frame['sex'].unique())

    return data_frame

def replace_strings_from_dict(df, column_name, replace_dict):
    """
    Replace entire strings in a DataFrame column based on multiple keywords, and print the number of changes for each.

    Parameters:
    df (pandas.DataFrame): The DataFrame to operate on.
    column_name (str): The name of the column to clean.
    replace_dict (dict): A dictionary where keys are keywords to search for, and values are the new values to replace the entire     string with.

    Returns:
    pandas.DataFrame: The DataFrame with the modified column.
    """

    # Check if column exists in DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    for keyword, new_value in replace_dict.items():
        # Use str.contains() to find rows where the column contains the keyword
        mask = df[column_name].str.contains(keyword, case=False, na=False)

        # Count the number of values that will be changed
        num_changes = mask.sum()
        if num_changes > 0:
            print(f"Number of values changed for '{keyword}': {num_changes}")

            # Replace the entire string in these rows with the new value
            df.loc[mask, column_name] = new_value

    return df

    # Performs linear regression delivering R2, adjusted R2 and coefficients 

def linear_regression(X_train, X_test, y_train, y_test):
    # Linear regression
    lm = LinearRegression()
    model = lm.fit(X_train, y_train)
    print(f'model coefficients:\n {model.coef_}\n')
    print(f'model intercept:\n {model.intercept_}\n')
    
    # Applying model to X test
    y_pred = model.predict(X_test)
    
    # Ensure y_test is in the correct format (pandas Series or 1D numpy array)
    if isinstance(y_test, pd.Series):
        y_test_reset = y_test.reset_index(drop=True)
    else:
        y_test_reset = y_test  # Assuming y_test is already a numpy array
    
    # Creating combined table with y_test and y_pred
    # Check if y_test_reset is a pandas Series and convert y_pred to a similar type
    if isinstance(y_test_reset, pd.Series):
        y_pred_series = pd.Series(y_pred, index=y_test_reset.index, name='y_pred')
        residuals_df = pd.concat([y_test_reset, y_pred_series], axis=1)
    else:
        # If inputs are numpy arrays, stack them horizontally
        residuals_df = np.column_stack((y_test_reset, y_pred))
        # Convert to DataFrame for easier manipulation later on
        residuals_df = pd.DataFrame(residuals_df, columns=["y_test", "y_pred"])
    
    # Calculating residuals
    residuals_df["residual"] = residuals_df["y_test"] - residuals_df["y_pred"]
    print(f'Residuals:\n {residuals_df}\n')

    # Root mean squared error
    rmse = mse(y_test_reset, residuals_df["y_pred"], squared=False)
    print(f'Root mean squared error: {rmse} \n')

    # R^2
    r2 = r2_score(y_test_reset, residuals_df["y_pred"])
    print(f'R2: {r2} \n')

    # Calculating adjusted R^2
    n = X_train.shape[0]  # Number of observations in the training set
    p = X_train.shape[1]  # Number of features used for training
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    print(f'Adjusted R2: {adjusted_r2} \n')

    return model.coef_

# Handles outliers 

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.25)
    quartile3 = dataframe[variable].quantile(0.75)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit,5)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit,5)
    


