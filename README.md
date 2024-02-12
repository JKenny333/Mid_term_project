# Mid_term_project

February 2024, IronHack Berlin

**Project participants:** 
- James Kenny
- Matthew Batchelor

**Team name**
Vino Metrics 

**Goal of the project:** 

We will build a model that predicts wine quality based on the properties of the wine. 
We will examine the following input variables across 6000+ instances: 

1. fixed_acidity	
2. volatile_acidity	
3. citric_acid	
4. residual_sugar	
5. chlorides	
6. free_sulfur_dioxide	
7. total_sulfur_dioxide	
8. density	
9. pH	
10. sulphates

Our output/explained variable is a score between 0 and 10. 

The source of our data is here: https://archive.ics.uci.edu/dataset/186/wine+quality

**File structure**

  **Data** 
  - includes CSVs used to conduct analysis

  **Notebooks**
  - includes .ipynb Notebooks used to conduct analysis and modeling

    - 1 Wine Data Cleaning and Initial EDA 
    - 2 Wine Detailed EDA 
    - 3 Wine Data Outlier Identification 
    - 4 Wine Preprocessing - Scaler testing 
    - 5 Wine Preprocessing and Linear regression - red_df (_without improvement (scalers etc.) to generate a              baseline_)
    - 6 Wine Preprocessing and Linear regression 
    - 7 Wine Data Hypothesis testing (_this sheets generates F-statistic and prob F-statistic_) 
    - 8 Wine Data Export Quality and Residuals (_export for visualisation on Tableau_) 
    - 9 White Wine preprocessing and linear regression 
    - 10 Wine quality predictor - (_tool for winemakers: generates quality prediction based on input_)

  **Project Notes and Presentational Material**
  - Vino Metrics Project Notes - offers overview of decision-making and logic applied through the week
  - Vino Metrics Presentation Skeleton - first draft of presentation

  **Visualisations** 
  - Visualisations from Tableau

  **Images**
  - Vino Metrics logo 

**Week plan**

  *Pre-work*

  **Dataset selection** Choose a compelling dataset 

  **Dataset Search & Evaluation:** Verify the dataset size, quality, and relevance. Ensure it's suitable for     linear regression analysis.

  *Monday* 

  **Overview Presentation:** Present project idea, covering the chosen topic, dataset, and what we intend to explore or predict with our linear regression model. Gather feedback to refine our approach if needed.

  **Dataset Search & Evaluation:** Verify the dataset size, quality, and relevance. Ensure it's suitable for linear regression analysis.

  **Initial Data Exploration:** Conduct a quick exploratory data analysis (EDA) to understand the dataset's structure, variables, and any apparent trends or issues like missing values.

  **Detailed EDA:** Dive deeper into the data to uncover patterns, correlations, and distributions. Identify potential features for our linear regression model.

  **Data Cleaning:** Address missing values, outliers, and incorrect data types. Ensure the dataset is clean and ready for further analysis.

  *Tuesday* 

  **Data Preprocessing:**  Based on our EDA, prepare the data for modeling. This may include feature engineering, normalization or standardization of variables, and encoding categorical features.

  **Model Development:** Begin with simple linear regression models to establish a baseline. Experiment with different feature combinations and model parameters.

  *Wednesday*

  **Model Refinement:** Evaluate model performance using appropriate metrics (e.g., R², RMSE). Adjust our model by adding, removing, or transforming features as necessary.

  **Validation:** Perform cross-validation to ensure your model's reliability and generalizability to unseen data.

  *Thursday* 

  **Final Analysis:** Interpret the model results to draw meaningful conclusions. Identify key predictors and their impact on the target variable.

  **Presentation Drafting:** Start compiling our results, insights, and methodology into a coherent presentation. Include visualizations that clearly communicate our findings.

  **Finalization Phase** Finalize our presentation and rehearse for the final presentation.

  *Friday*

  **Deliver our Presentation:** Confidently present our project, articulating the problem, the approach, key findings, and the implications of our results. Engage with our audience during the Q&A session to address their questions and feedback.
