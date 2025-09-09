import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Advanced Sales Predictor", page_icon="📊", layout="wide")
st.title("🚀 Advanced Sales Prediction Platform")

@st.cache_data
def generate_sample_data(months=36):
    np.random.seed(42)
    dates = pd.date_range('2021-01-01', periods=months, freq='M')
    base_sales = 10000 + np.arange(months) * 200
    seasonal = 2000 * np.sin(2 * np.pi * np.arange(months) / 12)
    marketing = np.random.uniform(1000, 5000, months)
    price = np.random.uniform(45, 55, months)
    competitor = np.random.randint(3, 8, months)
    
    sales = base_sales + seasonal + marketing * 0.5 - (price - 50) * 200 - competitor * 150
    sales += np.random.normal(0, 500, months)
    
    return pd.DataFrame({
        'Date': dates, 'Month': np.arange(1, months+1), 'Sales': sales,
        'Marketing_Spend': marketing, 'Price': price, 'Competitors': competitor
    })

# Sidebar
st.sidebar.header("⚙️ Configuration")
data_source = st.sidebar.radio("Data Source", ["Sample Data", "Upload CSV"])

if data_source == "Sample Data":
    months = st.sidebar.slider("Months", 12, 60, 36)
    df = generate_sample_data(months)
else:
    file = st.sidebar.file_uploader("Upload CSV", type="csv")
    if file:
        df = pd.read_csv(file)
    else:
        st.info("Please upload a CSV file")
        st.stop()

# Feature engineering
df['Sales_MA3'] = df['Sales'].rolling(3).mean()
df['Sales_Growth'] = df['Sales'].pct_change()
df['Quarter'] = ((df['Month'] - 1) // 3) + 1

# Main interface
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Analysis", "🤖 Models", "🔮 Forecast"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records", len(df))
    col2.metric("Avg Sales", f"${df['Sales'].mean():,.0f}")
    col3.metric("Max Sales", f"${df['Sales'].max():,.0f}")
    col4.metric("Growth Rate", f"{df['Sales_Growth'].mean()*100:.1f}%")
    
    st.subheader("Sales Trend")
    fig = px.line(df, x='Date' if 'Date' in df.columns else 'Month', y='Sales')
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(df.head())

with tab2:
    st.subheader("Correlation Analysis")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    fig = px.imshow(corr, color_continuous_scale='RdBu', aspect='auto')
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(df, x='Marketing_Spend', y='Sales', trendline='ols')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(df, x='Quarter', y='Sales')
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Model Training & Comparison")
    
    features = st.multiselect("Select Features", 
                             ['Month', 'Marketing_Spend', 'Price', 'Competitors'],
                             default=['Month', 'Marketing_Spend', 'Price'])
    
    if features:
        X = df[features].fillna(df[features].mean())
        y = df['Sales']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=42),
            'SVR': SVR(kernel='rbf')
        }
        
        if st.button("🚀 Train Models"):
            results = {}
            progress = st.progress(0)
            
            for i, (name, model) in enumerate(models.items()):
                try:
                    if name == 'SVR':
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    results[name] = {'R²': r2, 'RMSE': rmse, 'model': model}
                except:
                    results[name] = {'R²': 0, 'RMSE': float('inf'), 'model': None}
                
                progress.progress((i + 1) / len(models))
            
            # Results table
            results_df = pd.DataFrame({k: {'R²': v['R²'], 'RMSE': v['RMSE']} 
                                     for k, v in results.items()}).T
            st.dataframe(results_df.sort_values('R²', ascending=False))
            
            # Best model visualization
            best_model = max(results.keys(), key=lambda x: results[x]['R²'])
            st.success(f"🏆 Best Model: {best_model} (R² = {results[best_model]['R²']:.4f})")
            
            # Performance chart
            fig = px.bar(x=list(results.keys()), y=[v['R²'] for v in results.values()],
                        title="Model Performance (R² Score)")
            st.plotly_chart(fig, use_container_width=True)
            
            st.session_state['results'] = results
            st.session_state['features'] = features

with tab4:
    st.subheader("Sales Forecasting")
    
    if 'results' in st.session_state:
        results = st.session_state['results']
        features = st.session_state['features']
        
        model_choice = st.selectbox("Select Model", list(results.keys()))
        forecast_months = st.slider("Forecast Periods", 1, 12, 6)
        
        # Scenario inputs
        st.write("**Scenario Planning:**")
        col1, col2 = st.columns(2)
        scenario = {}
        for i, feature in enumerate(features):
            col = col1 if i % 2 == 0 else col2
            current_val = df[feature].iloc[-1] if feature != 'Month' else df['Month'].max()
            scenario[feature] = col.number_input(f"Future {feature}", value=float(current_val))
        
        if st.button("📈 Generate Forecast"):
            model = results[model_choice]['model']
            if model:
                # Create future data
                future_data = []
                for i in range(forecast_months):
                    future_point = scenario.copy()
                    if 'Month' in future_point:
                        future_point['Month'] = df['Month'].max() + i + 1
                    future_data.append(future_point)
                
                future_df = pd.DataFrame(future_data)
                
                # Handle SVR scaling
                if model_choice == 'SVR':
                    X_current = df[features].fillna(df[features].mean())
                    scaler = StandardScaler()
                    scaler.fit(X_current)
                    future_scaled = scaler.transform(future_df)
                    predictions = model.predict(future_scaled)
                else:
                    predictions = model.predict(future_df)
                
                # Visualization
                fig = go.Figure()
                
                # Historical data
                x_axis = df['Date'] if 'Date' in df.columns else df['Month']
                fig.add_trace(go.Scatter(x=x_axis, y=df['Sales'], name='Historical', line=dict(color='blue')))
                
                # Forecast
                if 'Date' in df.columns:
                    future_dates = pd.date_range(df['Date'].max(), periods=forecast_months+1, freq='M')[1:]
                    fig.add_trace(go.Scatter(x=future_dates, y=predictions, name='Forecast', 
                                           line=dict(color='red', dash='dash')))
                else:
                    future_months_x = range(df['Month'].max()+1, df['Month'].max()+forecast_months+1)
                    fig.add_trace(go.Scatter(x=future_months_x, y=predictions, name='Forecast', 
                                           line=dict(color='red', dash='dash')))
                
                fig.update_layout(title=f"Sales Forecast - {model_choice}", xaxis_title="Period", yaxis_title="Sales")
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast table
                forecast_df = pd.DataFrame({
                    'Period': range(1, forecast_months+1),
                    'Predicted_Sales': predictions
                })
                st.dataframe(forecast_df)
                
                # Download
                csv = forecast_df.to_csv(index=False)
                st.download_button("📥 Download Forecast", csv, "forecast.csv", "text/csv")
    else:
        st.info("Please train models first in the Models tab.")

st.markdown("---")
st.markdown("🚀 **Advanced Sales Prediction Platform** | Built with Streamlit & Machine Learning")

