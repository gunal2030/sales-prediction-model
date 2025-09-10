import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Advanced Sales Predictor", page_icon="📊", layout="wide")

# Title with styling
st.markdown("""
<style>
.main-title {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🚀 Advanced Sales Prediction Platform</h1>', unsafe_allow_html=True)

@st.cache_data
def generate_sample_data(months=36):
    """Generate realistic sample sales data"""
    np.random.seed(42)
    dates = pd.date_range('2021-01-01', periods=months, freq='M')
    
    # Base trend
    base_sales = 10000 + np.arange(months) * 200
    
    # Seasonal pattern
    seasonal = 2000 * np.sin(2 * np.pi * np.arange(months) / 12)
    
    # External factors
    marketing = np.random.uniform(1000, 5000, months)
    price = np.random.uniform(45, 55, months)
    competitor = np.random.randint(3, 8, months)
    temperature = 20 + 15 * np.sin(2 * np.pi * np.arange(months) / 12) + np.random.normal(0, 3, months)
    
    # Calculate sales with external influences
    sales = (base_sales + seasonal + marketing * 0.5 - (price - 50) * 200 
             - competitor * 150 + (temperature - 20) * 50)
    sales += np.random.normal(0, 500, months)
    sales = np.maximum(sales, 1000)  # Minimum sales
    
    return pd.DataFrame({
        'Date': dates,
        'Month': np.arange(1, months+1),
        'Sales': sales,
        'Marketing_Spend': marketing,
        'Price': price,
        'Competitors': competitor,
        'Temperature': temperature
    })

def create_features(df):
    """Add engineered features"""
    df = df.copy()
    
    # Time-based features
    if 'Date' in df.columns:
        df['Year'] = df['Date'].dt.year
        df['Quarter'] = df['Date'].dt.quarter
        df['Month_Name'] = df['Date'].dt.month_name()
    
    # Lag features
    df['Sales_Lag1'] = df['Sales'].shift(1)
    df['Sales_Lag3'] = df['Sales'].shift(3)
    
    # Rolling statistics
    df['Sales_MA3'] = df['Sales'].rolling(window=3).mean()
    df['Sales_MA6'] = df['Sales'].rolling(window=6).mean()
    df['Sales_Std3'] = df['Sales'].rolling(window=3).std()
    
    # Growth rates
    df['Sales_Growth'] = df['Sales'].pct_change()
    df['Sales_Growth_3M'] = df['Sales'].pct_change(periods=3)
    
    return df

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration Panel")

# Data source selection
data_source = st.sidebar.radio("📊 Data Source", ["Generate Sample Data", "Upload CSV File"])

if data_source == "Generate Sample Data":
    st.sidebar.subheader("Sample Data Parameters")
    months = st.sidebar.slider("Number of Months", 12, 60, 36)
    df = generate_sample_data(months)
    st.success(f"✅ Generated {months} months of sample data!")
else:
    uploaded_file = st.sidebar.file_uploader("Choose CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
    else:
        st.info("📁 Please upload a CSV file to continue.")
        st.stop()

# Feature engineering
df = create_features(df)

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🔍 Analysis", "🤖 Model Training", "🔮 Forecasting"])

with tab1:
    st.header("📊 Data Overview & Statistics")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Total Records", len(df))
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Avg Sales", f"${df['Sales'].mean():,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Max Sales", f"${df['Sales'].max():,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        growth = df['Sales_Growth'].mean() * 100 if 'Sales_Growth' in df.columns else 0
        st.metric("Avg Growth", f"{growth:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Sales trend chart
    st.subheader("📈 Sales Trend Over Time")
    fig, ax = plt.subplots(figsize=(12, 6))
    x_axis = df['Date'] if 'Date' in df.columns else df['Month']
    ax.plot(x_axis, df['Sales'], linewidth=2, color='#1f77b4')
    ax.set_title('Sales Over Time', fontsize=16, fontweight='bold')
    ax.set_xlabel('Period')
    ax.set_ylabel('Sales ($)')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Data preview
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head(10))
    
    # Statistical summary
    st.subheader("📊 Statistical Summary")
    st.dataframe(df.describe())

with tab2:
    st.header("🔍 Exploratory Data Analysis")
    
    # Correlation analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        st.subheader("🔗 Feature Correlation Matrix")
        corr_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                    square=True, ax=ax, cbar_kws={'shrink': 0.8})
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
    
    # Feature relationships
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Marketing_Spend' in df.columns:
            st.subheader("📊 Sales vs Marketing Spend")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(df['Marketing_Spend'], df['Sales'], alpha=0.6, color='#1f77b4')
            z = np.polyfit(df['Marketing_Spend'], df['Sales'], 1)
            p = np.poly1d(z)
            ax.plot(df['Marketing_Spend'], p(df['Marketing_Spend']), "r--", alpha=0.8)
            ax.set_xlabel('Marketing Spend ($)')
            ax.set_ylabel('Sales ($)')
            ax.set_title('Sales vs Marketing Spend')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
    
    with col2:
        if 'Quarter' in df.columns:
            st.subheader("📈 Sales Distribution by Quarter")
            fig, ax = plt.subplots(figsize=(8, 6))
            df.boxplot(column='Sales', by='Quarter', ax=ax)
            ax.set_title('Sales Distribution by Quarter')
            ax.set_xlabel('Quarter')
            ax.set_ylabel('Sales ($)')
            plt.suptitle('')  # Remove default title
            plt.tight_layout()
            st.pyplot(fig)

with tab3:
    st.header("🤖 Advanced Model Training & Comparison")
    
    # Feature selection
    st.subheader("🎯 Feature Selection for Modeling")
    
    available_features = [col for col in df.columns 
                         if col not in ['Sales', 'Date', 'Month_Name'] and df[col].dtype in ['int64', 'float64']]
    
    default_features = ['Month', 'Marketing_Spend', 'Price', 'Competitors']
    default_features = [f for f in default_features if f in available_features]
    
    selected_features = st.multiselect(
        "Select features for modeling:",
        available_features,
        default=default_features[:4] if default_features else available_features[:4]
    )
    
    if not selected_features:
        st.warning("⚠️ Please select at least one feature for modeling.")
        st.stop()
    
    # Model parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, 0.05)
    with col2:
        random_state = st.number_input("Random State", 0, 100, 42)
    with col3:
        cv_enabled = st.checkbox("Enable Cross Validation", True)
    
    # Prepare data
    X = df[selected_features].fillna(df[selected_features].mean())
    y = df['Sales']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Model definitions
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=random_state),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=random_state),
        'Support Vector Regression': SVR(kernel='rbf', C=1.0)
    }
    
    if st.button("🚀 Train All Models", type="primary"):
        results = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (name, model) in enumerate(models.items()):
            status_text.text(f"Training {name}...")
            
            try:
                # Handle scaling for SVR
                if name == 'Support Vector Regression':
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                
                results[name] = {
                    'R²': r2,
                    'RMSE': rmse,
                    'MAE': mae,
                    'model': model,
                    'predictions': y_pred
                }
                
            except Exception as e:
                st.warning(f"Model {name} failed: {str(e)}")
                results[name] = {'R²': 0, 'RMSE': float('inf'), 'MAE': float('inf')}
            
            progress_bar.progress((i + 1) / len(models))
        
        status_text.text("Training completed!")
        
        # Display results
        st.subheader("📊 Model Performance Comparison")
        
        # Create results dataframe
        results_data = []
        for name, metrics in results.items():
            if isinstance(metrics.get('R²'), (int, float)):
                results_data.append({
                    'Model': name,
                    'R² Score': f"{metrics['R²']:.4f}",
                    'RMSE': f"{metrics['RMSE']:.2f}",
                    'MAE': f"{metrics['MAE']:.2f}"
                })
        
        results_df = pd.DataFrame(results_data)
        results_df = results_df.sort_values('R² Score', ascending=False)
        st.dataframe(results_df, use_container_width=True)
        
        # Best model highlight
        if results_data:
            best_model_name = results_df.iloc[0]['Model']
            best_r2 = results_df.iloc[0]['R² Score']
            st.success(f"🏆 Best Performing Model: **{best_model_name}** (R² = {best_r2})")
        
        # Performance visualization
        if len(results_data) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # R² Score comparison
            models_list = [r['Model'] for r in results_data]
            r2_scores = [float(r['R² Score']) for r in results_data]
            
            ax1.bar(models_list, r2_scores, color='skyblue', alpha=0.7)
            ax1.set_title('Model Comparison - R² Scores', fontweight='bold')
            ax1.set_ylabel('R² Score')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # RMSE comparison
            rmse_scores = [float(r['RMSE']) for r in results_data]
            ax2.bar(models_list, rmse_scores, color='lightcoral', alpha=0.7)
            ax2.set_title('Model Comparison - RMSE', fontweight='bold')
            ax2.set_ylabel('RMSE')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Store results in session state
        st.session_state['model_results'] = results
        st.session_state['selected_features'] = selected_features
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test

with tab4:
    st.header("🔮 Sales Forecasting & Prediction")
    
    if 'model_results' in st.session_state:
        results = st.session_state['model_results']
        features = st.session_state['selected_features']
        
        # Model selection
        available_models = [name for name, res in results.items() if 'model' in res]
        if available_models:
            selected_model = st.selectbox("🎯 Select Model for Forecasting", available_models)
            
            # Forecasting parameters
            col1, col2 = st.columns(2)
            with col1:
                forecast_periods = st.slider("📅 Forecast Periods", 1, 24, 6)
            with col2:
                confidence_level = st.selectbox("📊 Confidence Level", [90, 95, 99], index=1)
            
            # Scenario planning inputs
            st.subheader("🎛️ Scenario Planning - Set Future Values")
            scenario_values = {}
            
            cols = st.columns(min(3, len(features)))
            for i, feature in enumerate(features):
                col_idx = i % len(cols)
                with cols[col_idx]:
                    if feature == 'Month':
                        current_val = df[feature].max()
                        scenario_values[feature] = st.number_input(
                            f"Starting {feature}", 
                            value=int(current_val + 1),
                            key=f"scenario_{feature}"
                        )
                    else:
                        current_val = df[feature].iloc[-1] if len(df) > 0 else df[feature].mean()
                        scenario_values[feature] = st.number_input(
                            f"Future {feature}", 
                            value=float(current_val),
                            key=f"scenario_{feature}"
                        )
            
            if st.button("📈 Generate Forecast", type="primary"):
                model = results[selected_model]['model']
                
                # Create future data points
                future_data = []
                for i in range(forecast_periods):
                    future_point = {}
                    for feature in features:
                        if feature == 'Month':
                            future_point[feature] = scenario_values[feature] + i
                        else:
                            future_point[feature] = scenario_values[feature]
                    future_data.append(future_point)
                
                future_df = pd.DataFrame(future_data)
                
                # Generate predictions
                try:
                    if selected_model == 'Support Vector Regression':
                        # Handle scaling for SVR
                        scaler = StandardScaler()
                        X_current = df[features].fillna(df[features].mean())
                        scaler.fit(X_current)
                        future_scaled = scaler.transform(future_df)
                        predictions = model.predict(future_scaled)
                    else:
                        predictions = model.predict(future_df)
                    
                    # Calculate confidence intervals
                    if 'predictions' in results[selected_model]:
                        residuals = st.session_state['y_test'] - results[selected_model]['predictions']
                        std_residual = np.std(residuals)
                        
                        # Z-scores for confidence intervals
                        z_scores = {90: 1.645, 95: 1.96, 99: 2.576}
                        z_score = z_scores[confidence_level]
                        
                        margin_error = z_score * std_residual
                        upper_bound = predictions + margin_error
                        lower_bound = predictions - margin_error
                    else:
                        # Fallback confidence intervals
                        margin_error = np.std(df['Sales']) * 0.1
                        upper_bound = predictions + margin_error
                        lower_bound = predictions - margin_error
                    
                    # Create forecast visualization
                    st.subheader("📊 Forecast Visualization")
                    
                    fig, ax = plt.subplots(figsize=(14, 8))
                    
                    # Plot historical data
                    x_hist = df['Date'] if 'Date' in df.columns else df['Month']
                    ax.plot(x_hist, df['Sales'], label='Historical Sales', 
                           color='blue', linewidth=2, alpha=0.8)
                    
                    # Plot forecast
                    if 'Date' in df.columns:
                        last_date = df['Date'].max()
                        future_dates = pd.date_range(
                            start=last_date + pd.DateOffset(months=1), 
                            periods=forecast_periods, 
                            freq='M'
                        )
                        x_future = future_dates
                    else:
                        x_future = range(df['Month'].max() + 1, 
                                       df['Month'].max() + forecast_periods + 1)
                    
                    ax.plot(x_future, predictions, label='Forecast', 
                           color='red', linewidth=2, linestyle='--', alpha=0.8)
                    
                    # Plot confidence intervals
                    ax.fill_between(x_future, lower_bound, upper_bound, 
                                   color='red', alpha=0.2, 
                                   label=f'{confidence_level}% Confidence Interval')
                    
                    ax.set_title(f'Sales Forecast - {selected_model}', 
                               fontsize=16, fontweight='bold')
                    ax.set_xlabel('Period')
                    ax.set_ylabel('Sales ($)')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    if 'Date' in df.columns:
                        plt.xticks(rotation=45)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Forecast summary table
                    st.subheader("📋 Forecast Summary")
                    
                    forecast_summary = pd.DataFrame({
                        'Period': range(1, forecast_periods + 1),
                        'Predicted_Sales': [f"${pred:,.2f}" for pred in predictions],
                        'Lower_Bound': [f"${lb:,.2f}" for lb in lower_bound],
                        'Upper_Bound': [f"${ub:,.2f}" for ub in upper_bound]
                    })
                    
                    st.dataframe(forecast_summary, use_container_width=True)
                    
                    # Key insights
                    st.subheader("💡 Key Insights")
                    avg_prediction = np.mean(predictions)
                    total_forecast = np.sum(predictions)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Predicted Sales", f"${avg_prediction:,.2f}")
                    with col2:
                        st.metric("Total Forecast Period", f"${total_forecast:,.2f}")
                    with col3:
                        current_avg = df['Sales'].tail(3).mean()
                        change_pct = ((avg_prediction - current_avg) / current_avg) * 100
                        st.metric("Expected Change", f"{change_pct:+.1f}%")
                    
                    # Download forecast data
                    forecast_download = pd.DataFrame({
                        'Period': range(1, forecast_periods + 1),
                        'Predicted_Sales': predictions,
                        'Lower_Bound': lower_bound,
                        'Upper_Bound': upper_bound
                    })
                    
                    csv_data = forecast_download.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Forecast Data",
                        data=csv_data,
                        file_name="sales_forecast.csv",
                        mime="text/csv",
                        type="secondary"
                    )
                    
                except Exception as e:
                    st.error(f"Error generating forecast: {str(e)}")
        else:
            st.warning("⚠️ No trained models available for forecasting.")
    else:
        st.info("🎯 Please train models first in the 'Model Training' tab to enable forecasting.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; color: #666;'>
    <p><strong>🚀 Advanced Sales Prediction Platform</strong></p>
    <p>Built with Streamlit | Powered by Machine Learning & Advanced Analytics</p>
</div>
""", unsafe_allow_html=True)


