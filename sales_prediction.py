import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import io

# Set page config
st.set_page_config(page_title="Sales Prediction Model", page_icon="📈", layout="wide")

# Title and description
st.title("📈 Sales Prediction Model")
st.markdown("This app uses linear regression to predict sales based on monthly data.")

# Sidebar for user inputs
st.sidebar.header("Configuration")

# Option to upload CSV or use sample data
data_option = st.sidebar.radio("Data Source:", ["Use Sample Data", "Upload CSV File"])

@st.cache_data
def create_sample_data():
    """Create sample sales data for demonstration"""
    np.random.seed(42)
    months = np.arange(1, 25)  # 24 months
    # Create a trend with some noise
    sales = 1000 + months * 50 + np.random.normal(0, 100, len(months))
    df = pd.DataFrame({
        'Month': months,
        'Sales': sales
    })
    return df

@st.cache_data
def load_data(uploaded_file):
    """Load data from uploaded file"""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# Load data based on user choice
if data_option == "Use Sample Data":
    df = create_sample_data()
    st.success("Sample data loaded successfully!")
else:
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.success("File uploaded successfully!")
    else:
        st.info("Please upload a CSV file to continue.")
        st.stop()

# Display data info
if df is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Data Overview")
        st.write(f"Dataset shape: {df.shape}")
        st.write("First few rows:")
        st.dataframe(df.head())
    
    with col2:
        st.subheader("📈 Data Statistics")
        st.write(df.describe())

    # Check if required columns exist
    required_columns = ['Month', 'Sales']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        st.info("Your CSV file should have 'Month' and 'Sales' columns.")
        st.stop()

    # Model parameters
    st.sidebar.subheader("Model Parameters")
    test_size = st.sidebar.slider("Test Size (fraction)", 0.1, 0.5, 0.2, 0.05)
    random_state = st.sidebar.number_input("Random State", 0, 100, 42)

    # Prepare data
    X = df[['Month']]
    y = df['Sales']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Model training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_all = model.predict(X)

    # Model evaluation
    r2 = r2_score(y_test, y_pred)

    # Display results
    st.subheader("🎯 Model Performance")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
    
    with col2:
        st.metric("Training Samples", len(X_train))
    
    with col3:
        st.metric("Test Samples", len(X_test))

    # Visualization
    st.subheader("📊 Sales Prediction Visualization")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot actual data points
    ax.scatter(X, y, color='blue', alpha=0.6, label='Actual Sales')
    
    # Plot regression line
    ax.plot(X, y_pred_all, color='red', linewidth=2, label='Prediction Line')
    
    # Plot test predictions
    ax.scatter(X_test, y_pred, color='orange', alpha=0.8, s=60, 
               label='Test Predictions', marker='x')
    
    ax.set_title('Sales Prediction Model', fontsize=16, fontweight='bold')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Sales', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)

    # Model coefficients
    st.subheader("🔍 Model Details")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Model Coefficients:**")
        st.write(f"Slope: {model.coef_[0]:.2f}")
        st.write(f"Intercept: {model.intercept_:.2f}")
    
    with col2:
        st.write("**Model Equation:**")
        st.write(f"Sales = {model.intercept_:.2f} + {model.coef_[0]:.2f} × Month")

    # Prediction for new values
    st.subheader("🔮 Make Predictions")
    new_month = st.number_input("Enter month for prediction:", 
                               min_value=1, max_value=100, value=25)
    
    if st.button("Predict Sales"):
        prediction = model.predict([[new_month]])[0]
        st.success(f"Predicted sales for month {new_month}: ${prediction:,.2f}")

    # Download results
    st.subheader("💾 Download Results")
    
    # Create results dataframe
    results_df = df.copy()
    results_df['Predicted_Sales'] = y_pred_all
    results_df['Residuals'] = y - y_pred_all
    
    csv_buffer = io.StringIO()
    results_df.to_csv(csv_buffer, index=False)
    
    st.download_button(
        label="Download Predictions as CSV",
        data=csv_buffer.getvalue(),
        file_name="sales_predictions.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("Built with Streamlit 🎈 | Sales Prediction Model")
