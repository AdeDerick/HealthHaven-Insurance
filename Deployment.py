import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
import joblib
import base64
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error, r2_score

# Define the predictor columns used for the model
predictor_columns = [
    'Age', 'Base Premium', 'No. of Hospital Visits',
    'No. of Lab Visits', 'No. of GP Visit', 'No. of SP Visit', 'Smokes',
    'Pre-existing Condition', 'Hypertention', 'Diabetes', 'Dyslipidaemia/ Hyperlipidaemia',
    'Refractive Error', 'Spondylosis', 'Stomach Ulcer', 'Gender_Male', 'Marital Status_Single',
    'Work Industry_Construction', 'Work Industry_Consulting', 'Work Industry_Education',
    'Work Industry_Embassy', 'Work Industry_Healthcare', 'Work Industry_Manufacturing',
    'Work Industry_Media', 'Work Industry_NGO', 'Work Industry_Oil & Gas',
    'Work Industry_Procurement', 'Work Industry_Public Service', 'Work Industry_Telecommunication',
    'Work Industry_Trading', 'Policy Duration', 'Total_Healthcare_Expenditure'
]


# Function to load models
def load_model(model_path):
    return joblib.load(model_path)

# Load the models
regression_model = load_model('gradient_boosting_model.pkl')
classification_model = load_model('random_forest_model.pkl')


# Define your industry options globally
industry_options = [
    'Construction', 'Consulting', 'Education', 'Embassy', 'Healthcare',
    'Manufacturing', 'Media', 'NGO', 'Oil & Gas', 'Procurement',
    'Public Service', 'Telecommunication', 'Trading'
]


# Load the dataset
model_data = pd.read_csv('model_data.csv')

def plot_pairplot(data, columns):
    # Explicitly creating a figure with subplots
    plt.figure(figsize=(10, 8))  # Adjust size as needed
    g = sns.pairplot(data[columns], diag_kind='kde')  # 'kde' for density plots on the diagonal
    return g

def plot_boxplot(data, columns):
    # Creating a new figure and axes for boxplot
    fig, ax = plt.subplots(figsize=(12, 6))
    # Generating a box plot
    sns.boxplot(data=data[columns], ax=ax)
    plt.title('Box Plot for Detecting Outliers')
    plt.xticks(rotation=45)  # Rotate labels for better readability if necessary
    plt.tight_layout()  # Adjust layout to fit all elements
    return fig

def to_excel(df):
    """Convert a DataFrame to an Excel file."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(df):
    """Generate a download link allowing the output data to be downloaded as an Excel file."""
    val = to_excel(df)
    b64 = base64.b64encode(val)  # b64 encode the bytes
    href = f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="prediction_results.xlsx">Download Excel file</a>'
    return href


# Define all pages
def home_page():
    st.title('Health Insurance Utilization and Cost Prediction')

    # Create two columns for images
    col1, col2 = st.columns(2)

    with col1:
        st.image('hi_image4.jpg', caption='Health Insurance Company', width=300)  # Adjust width as needed

    with col2:
        st.image('consulting.jpeg', caption='Consulting Services', width=300)  # Adjust width as needed

    # Problem Statement
    st.subheader("Problem Statement")
    st.write("""
    Determination of premiums to charge health insurance subscribers is a difficult one as insurers would like to know the cost to be incurred on the subscriber.
    Health insurance organizations normally rely on subscriber habits, age, pre-existing health conditions, the work of the subscriber, and compare it with data on similar past subscribers.
    This approach, although useful, has negative impacts on the organization, including but not limited to:
    - Customer retention
    - Incurred expense exceeding the premium charged
    """)

    # Objectives
    st.subheader("Objectives")
    st.write("""
    - **Predict Premiums**: Develop a model to predict the total amount (utilization) the health insurance scheme is likely to spend on a subscriber.
    - **Identify Low-Cost Subscribers**: Use classification techniques to identify subscribers who are likely to incur minimal costs.
    """)

    # Members
    st.write("---")
    st.subheader("Project Members")
    st.write("""
    - Derick Ade Ofosu: 11365310
    - Samuel Kwafo Offe: 11367142
    - Amanda Emefa Mensah: 11365194
    - Ebenezer Hammond: 22009139
    - Loverage Paa Ekow Amorin: 11367159
    """)

# New "Dataset & Preprocessing Steps" page
def dataset_preprocessing_page():
    st.subheader("Dataset & Preprocessing Steps")

    # Original Data
    st.subheader("Original Dataset")
    original_data = pd.read_csv('original_data.csv')
    st.dataframe(original_data)

    # Preprocessing Steps
    st.subheader("Preprocessing Steps")
    st.markdown("""
    **Step 1: Data Cleaning**
    - **Handling Missing Data**: We replaced missing values in Utilization and corresponding null columns with zero.
    - **Handling Missing Values**: Missing values were filled with appropriate strategies like K-Nearest Neighbors for the number of visits.

    **Step 2: Convert Dates and Categorical Variables**
    - **Date Conversion**: Converted 'Policy Start Date' and 'Policy End Date' from object to datetime format.
    - **Categorical Variables**: Converted binary categories to numeric (0 and 1) and applied one-hot encoding to other categorical variables.

    **Step 3: Convert Monetary Values from Object to Float**
    - **Monetary Conversion**: Consultation and drug costs were cleaned for non-numeric characters and converted to float.
    - **Imputation**: Missing values in monetary columns were imputed with the mean of each column.
    """)

    # Feature Engineering
    st.subheader("Feature Engineering")
    st.markdown("""
        **Step 4: Feature Creation**
        - **Healthcare Services Indicator**: Created a binary indicator for whether any healthcare services were used based on utilization.
        - **Policy Duration**: Calculated the duration of the insurance policy in days.
        - **Total Healthcare Expenditure**: Created a composite variable for total expenditure from various individual cost components.
        """)


    # Displaying Cleaned Data
    st.subheader("Cleaned Dataset")
    cleaned_data = pd.read_csv('cleaned_data.csv')
    st.dataframe(cleaned_data)

    # List of features used in both models
    st.subheader("Selected Features for Modeling")
    st.write("""
        The following features were selected based on their importance and relevance to the target variable (utilization):
        - Age
        - Base Premium
        - No. of Hospital Visits
        - No. of Lab Visits
        - No. of GP Visit
        - No. of SP Visit
        - Smokes
        - Pre-existing Condition
        - Hypertention
        - Diabetes
        - Dyslipidaemia/ Hyperlipidaemia
        - Refractive Error
        - Spondylosis
        - Stomach Ulcer
        - Gender_Male
        - Marital Status_Single
        - Policy Duration
        - Total Healthcare Expenditure
        """)


    # Display Model Data
    st.subheader("Model Dataset")
    model_data = pd.read_csv('model_data.csv')
    st.dataframe(model_data)


def eda_page():
    st.subheader("Exploratory Data Analysis")
    st.write("Here, we visualize various aspects of the healthcare data to uncover patterns and insights.")
    # Assume some visualizations are here
    # For example:

    # Step 1: Statistical Summary
    st.write("### 1. Statistical Summary")
    st.write(model_data.describe())

    # Step 2: Distribution Plots for All Selected Features
    st.write("### 2. Distribution of Selected Numeric Features")

    # List of features for distribution plots
    numeric_cols = ['Age', 'Base Premium', 'No. of Hospital Visits',
                    'Total_Healthcare_Expenditure', 'No. of Lab Visits',
                    'No. of GP Visit', 'No. of SP Visit', 'Policy Duration', 'Utilization']

    # Create subplots with 3 charts per row
    num_cols_per_row = 3
    num_rows = (len(numeric_cols) + num_cols_per_row - 1) // num_cols_per_row  # Calculate required rows
    fig, ax = plt.subplots(num_rows, num_cols_per_row, figsize=(18, 12))

    # Flatten axes for easier indexing
    ax = ax.flatten()

    # Plot each feature in the subplots
    for i, col in enumerate(numeric_cols):
        sns.histplot(model_data[col], kde=True, ax=ax[i])
        ax[i].set_title(f'Distribution of {col}')

    # Remove any empty subplots
    for i in range(len(numeric_cols), len(ax)):
        fig.delaxes(ax[i])

    # Adjust the layout to prevent overlap
    plt.tight_layout()
    st.pyplot(fig)

    # Step 3: Correlation Matrix Heatmap
    st.write("### Correlation Heatmap")
    correlation_matrix = model_data.corr()
    # Increase the plot size
    plt.figure(figsize=(20, 15))
    # Mask for the upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    # Drawing the heatmap with the mask
    sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt=".2f", linewidths=.5, cmap='coolwarm',
                cbar_kws={"shrink": .5}, annot_kws={"size": 8})
    st.pyplot(plt)
    plt.clf()  # Clear plot to avoid overlap with future plots

    # Step 4: High Correlations with Target and Among Predictors
    st.write("### 4. High Correlations with Target and Among Predictors")

    target_variable = 'Utilization'

    # Filter for high correlations with the target variable
    correlation_df = model_data.corr()
    target_corr = correlation_df[target_variable][abs(correlation_df[target_variable]) > 0.5]
    target_corr = target_corr[target_corr.index != target_variable].reset_index()
    target_corr.columns = ['Feature', 'Correlation with Target']

    # Filter for high correlations among predictors
    high_corr = correlation_df[abs(correlation_df) > 0.5].stack().reset_index()
    high_corr.columns = ['Feature 1', 'Feature 2', 'Correlation']
    high_corr = high_corr[high_corr['Feature 1'] != high_corr['Feature 2']]  # Remove self-correlations
    predictor_corr = high_corr[
        ~high_corr['Feature 1'].eq(target_variable) & ~high_corr['Feature 2'].eq(target_variable)]

    # Display correlations with the target
    st.write("#### High Correlations with Target Variable:")
    st.dataframe(target_corr.sort_values(by='Correlation with Target', ascending=False))

    # Display high correlations among predictors
    st.write("#### High Correlations Among Predictors:")
    st.dataframe(predictor_corr.sort_values(by='Correlation', ascending=False))

    # Step 5: Pairplot for selected variables
    st.write("### 5. Pairplot of Selected Features")
    selected_cols = ['Age', 'Base Premium', 'No. of Hospital Visits', 'Utilization', 'Total_Healthcare_Expenditure']
    # Call the plotting function and pass the relevant DataFrame and columns
    pairplot_graph = plot_pairplot(model_data, selected_cols)
    st.pyplot(pairplot_graph)

    # Anomaly Detection Section
    st.write("### Anomaly Detection with Box Plots")
    columns_to_check = ['Age', 'Base Premium', 'No. of Hospital Visits', 'Total_Healthcare_Expenditure', 'Utilization']
    fig = plot_boxplot(model_data, columns_to_check)
    st.pyplot(fig)


def model_evaluation_page():
    st.title("Model and Evaluation Metrics")

    # Introduction to Model Approaches
    st.header("Model Approaches")
    st.write("""
    We utilized **Gradient Boosting Regressor** for the regression problem to predict healthcare utilization costs, 
    and **Random Forest Classifier** for the classification problem to identify low-cost subscribers.
    """)

    # Evaluation Metrics Used
    st.header("Evaluation Metrics")
    st.write("""
    The following metrics were considered for model evaluation:
    - **Regression**: Root Mean Squared Error (RMSE), R-Squared
    - **Classification**: Accuracy, Precision, Recall, F1-Score
    """)

    # Model Selection
    st.header("Model Selection")
    st.write("""
    - **Regression Model Choice**: Gradient Boosting was chosen for its ability to handle various types of data and its effectiveness in reducing overfitting through boosting techniques.
    - **Classification Model Choice**: Random Forest was selected due to its robustness against overfitting and its excellent performance on imbalanced datasets.
    """)

    # Final Model Evaluation
    st.header("Final Model Evaluation")
    st.subheader("Gradient Boosting Regressor")
    st.write("Final RMSE: 1400.56")
    st.write("Final R-Squared: 0.865")

    st.subheader("Random Forest Classifier")
    st.write("Final Accuracy: 94.6%")
    st.write("Final F1-Score: 0.946")


def prediction_page():
    st.title("Subscriber Prediction")
    st.header("Predict Healthcare Utilization and Cost Classification")
    st.write("Please enter the details of the subscriber to predict healthcare utilization and cost classification.")

    with st.form(key='prediction_form'):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.subheader("Demographic Details")
            age = st.number_input('Age', min_value=18, max_value=100, step=1)
            gender_male = st.selectbox('Gender (Male)', [0, 1])
            marital_status_single = st.selectbox('Marital Status (Single)', [0, 1])
            base_premium = st.number_input('Base Premium', min_value=100.0, max_value=10000.0, step=0.1)

        with col2:
            st.subheader("Health Status")
            smokes = st.selectbox('Smokes', [0, 1])
            pre_existing_condition = st.selectbox('Pre-existing Condition', [0, 1])
            hypertention = st.selectbox('Hypertension', [0, 1])
            diabetes = st.selectbox('Diabetes', [0, 1])
            dyslipidemia = st.selectbox('Dyslipidaemia/Hyperlipidaemia', [0, 1])
            refractive_error = st.selectbox('Refractive Error', [0, 1])
            spondylosis = st.selectbox('Spondylosis', [0, 1])
            stomach_ulcer = st.selectbox('Stomach Ulcer', [0, 1])

        with col3:
            st.subheader("Healthcare Utilization")
            no_of_hospital_visits = st.number_input('Number of Hospital Visits', min_value=0, max_value=100, step=1)
            no_of_lab_visits = st.number_input('Number of Lab Visits', min_value=0, max_value=100, step=1)
            no_of_gp_visit = st.number_input('Number of GP Visits', min_value=0, max_value=100, step=1)
            no_of_sp_visit = st.number_input('Number of Specialist Visits', min_value=0, max_value=100, step=1)
            total_healthcare_expenditure = st.number_input('Total_Healthcare_Expenditure', min_value=0.0,
                                                           max_value=100000.0, step=0.1)

        with col4:
            st.subheader("Employment Details")
            industry = st.selectbox('Work Industry', industry_options)
            policy_duration = st.number_input('Policy Duration (in Days)', min_value=0, max_value=3650, step=1)

        submit_button = st.form_submit_button(label='Predict')

    if submit_button:
        # Create dictionary to store industry flags
        industry_dummies = {f"Work Industry_{opt}": 1 if opt == industry else 0 for opt in industry_options}

        # Prepare the input data as a dictionary
        input_data_dict = {
            'Age': [age], 'Base Premium': [base_premium], 'No. of Hospital Visits': [no_of_hospital_visits],
            'No. of Lab Visits': [no_of_lab_visits], 'No. of GP Visit': [no_of_gp_visit],
            'No. of SP Visit': [no_of_sp_visit],
            'Smokes': [smokes], 'Pre-existing Condition': [pre_existing_condition], 'Hypertention': [hypertention],
            'Diabetes': [diabetes], 'Dyslipidaemia/ Hyperlipidaemia': [dyslipidemia],
            'Refractive Error': [refractive_error],
            'Spondylosis': [spondylosis], 'Stomach Ulcer': [stomach_ulcer], 'Gender_Male': [gender_male],
            'Marital Status_Single': [marital_status_single], 'Policy Duration': [policy_duration],
            'Total_Healthcare_Expenditure': [total_healthcare_expenditure]
        }

        # Update input dictionary with industry dummies
        input_data_dict.update(industry_dummies)

        # Create DataFrame from the input data dictionary
        input_data = pd.DataFrame(input_data_dict)

        # Reorder DataFrame columns to match the training data
        ordered_columns = [
            'Age', 'Base Premium', 'No. of Hospital Visits', 'No. of Lab Visits', 'No. of GP Visit', 'No. of SP Visit',
            'Smokes',
            'Pre-existing Condition', 'Hypertention', 'Diabetes', 'Dyslipidaemia/ Hyperlipidaemia', 'Refractive Error',
            'Spondylosis', 'Stomach Ulcer', 'Gender_Male', 'Marital Status_Single', 'Work Industry_Construction',
            'Work Industry_Consulting', 'Work Industry_Education', 'Work Industry_Embassy', 'Work Industry_Healthcare',
            'Work Industry_Manufacturing', 'Work Industry_Media', 'Work Industry_NGO', 'Work Industry_Oil & Gas',
            'Work Industry_Procurement', 'Work Industry_Public Service', 'Work Industry_Telecommunication',
            'Work Industry_Trading', 'Policy Duration', 'Total_Healthcare_Expenditure'
        ]
        # Ensure the input data follows the exact order
        input_data = input_data[ordered_columns]

        # Predict Utilization and Cost Classification
        utilization_pred = regression_model.predict(input_data)[0]
        cost_classification_pred = classification_model.predict(input_data)[0]

        # Display the predictions
        st.success(f"Predicted Utilization: ${utilization_pred:.2f}")
        st.success(f"Cost Classification: {'Low Cost' if cost_classification_pred == 1 else 'High Cost'}")


def upload_prediction_page():
    st.title("Upload Prediction")
    st.header("Batch Prediction for Healthcare Utilization and Cost Classification")
    uploaded_file = st.file_uploader("Choose an Excel file (.xlsx) containing the test data", type="xlsx")

    if uploaded_file is not None:
        # Load the data
        data = pd.read_excel(uploaded_file)

        # List of feature columns expected by the model
        feature_columns = [
            'Age', 'Base Premium', 'No. of Hospital Visits', 'No. of Lab Visits', 'No. of GP Visit',
            'No. of SP Visit', 'Smokes', 'Pre-existing Condition', 'Hypertention', 'Diabetes',
            'Dyslipidaemia/ Hyperlipidaemia', 'Refractive Error', 'Spondylosis', 'Stomach Ulcer',
            'Gender_Male', 'Marital Status_Single', 'Work Industry_Construction', 'Work Industry_Consulting',
            'Work Industry_Education', 'Work Industry_Embassy', 'Work Industry_Healthcare',
            'Work Industry_Manufacturing', 'Work Industry_Media', 'Work Industry_NGO', 'Work Industry_Oil & Gas',
            'Work Industry_Procurement', 'Work Industry_Public Service', 'Work Industry_Telecommunication',
            'Work Industry_Trading', 'Policy Duration', 'Total_Healthcare_Expenditure'
        ]

        # Make sure to select only the columns that were used during model training
        data_for_prediction = data[feature_columns]

        # Show the data
        st.write("Data Uploaded Successfully:")
        st.dataframe(data.head())

        if st.button("Predict"):
            # Predict Utilization and Cost Classification
            data['Predicted Utilization'] = regression_model.predict(data_for_prediction)
            predicted_classes = classification_model.predict(data_for_prediction)
            data['Cost Classification'] = pd.Series(predicted_classes).apply(
                lambda x: 'Low Cost' if x == 1 else 'High Cost')

            # Show the updated data
            st.write("Prediction Results:")
            st.dataframe(data)

            # Summary statistics
            st.write("Summary Statistics for Predicted Utilization:")
            st.write(data['Predicted Utilization'].describe())

            # Distribution of Predicted Utilization
            st.write("Distribution of Predicted Utilization:")
            fig, ax = plt.subplots()
            sns.histplot(data['Predicted Utilization'], kde=True, ax=ax)
            st.pyplot(fig)

            # Pie chart for Cost Classification
            st.write("Cost Classification Distribution:")
            fig1, ax1 = plt.subplots()
            data['Cost Classification'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax1)
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(fig1)

            # Allow user to download the results as Excel
            output = data.to_csv(index=False)
            st.download_button(
                label="Download data as CSV",
                data=output,
                file_name='predicted_results.csv',
                mime='text/csv',
            )


# Add to sidebar navigation
option = st.sidebar.selectbox('Choose a page:', ['Home', 'Dataset & Preprocessing', 'EDA', 'Model and Evaluation Metrics', 'Prediction', 'Upload Prediction'])

# Display the correct page based on user selection
if option == 'Model and Evaluation Metrics':
    model_evaluation_page()
elif option == 'Home':
    home_page()
elif option == 'Dataset & Preprocessing':
    dataset_preprocessing_page()
elif option == 'EDA':
    eda_page()
elif option == 'Prediction':
    prediction_page()
elif option == 'Upload Prediction':
    upload_prediction_page()