"""
Vehicle Analysis Multi-Agent System
Version: 2.2.0
Date: May 31, 2025
Author: Multi-Agent CSV Data Analysis System

Changelog:
v2.2.0 - Added proper time-series forecasting with seasonal analysis using listing_date
v2.1.0 - Fixed electric vehicle filtering bug and persistent status messages
v2.0.0 - Multi-agent architecture with DataAgent, PlotAgent, ForecastAgent, OrchestratorAgent
v1.0.0 - Initial release

Features:
- Natural language query processing for vehicle data
- Multi-agent system architecture
- Interactive data visualization with Plotly
- Proper time-series forecasting with seasonal analysis
- Comprehensive vehicle filtering (make, model, year, fuel type, state, body style)
- Real-time data analysis and charts
- Monthly, weekly, and quarterly seasonal pattern analysis
- Confidence intervals for forecasts

Installation Requirements:
pip install streamlit pandas numpy plotly scikit-learn

Usage:
streamlit run vehicle_analysis_v2_2_0.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import re
import time

# Application Configuration
APP_VERSION = "2.2.0"
APP_NAME = "Vehicle Analysis Multi-Agent System"
RELEASE_DATE = "May 31, 2025"

class DataAgent:
    """DataAgent handles all data loading, cleaning, and preprocessing operations."""
    
    def __init__(self):
        self.data = None
        
    def load_csv(self, file_path_or_buffer):
        """Load CSV data from file path or buffer"""
        try:
            self.data = pd.read_csv(file_path_or_buffer)
            return True, f"Data loaded successfully: {self.data.shape[0]} rows, {self.data.shape[1]} columns"
        except Exception as e:
            return False, f"Error loading data: {str(e)}"
    
    def clean_data(self):
        """Clean data by handling missing values"""
        try:
            if self.data is None:
                return False, "No data loaded"
                
            # Handle numeric columns - fill with median
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            categorical_cols = self.data.select_dtypes(include=['object']).columns
            
            for col in numeric_cols:
                self.data[col].fillna(self.data[col].median(), inplace=True)
            
            # Handle categorical columns - fill with mode or 'Unknown'
            for col in categorical_cols:
                mode_val = self.data[col].mode()
                fill_val = mode_val[0] if not mode_val.empty else 'Unknown'
                self.data[col].fillna(fill_val, inplace=True)
            
            return True, "Data cleaned successfully"
        except Exception as e:
            return False, f"Error cleaning data: {str(e)}"
    
    def get_data_summary(self):
        """Get basic summary of the loaded data"""
        if self.data is None:
            return None
            
        return {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'sample': self.data.head().to_dict()
        }

class PlotAgent:
    """PlotAgent handles all visualization and charting operations."""
    
    def create_bar_chart(self, data, x_col, y_col, title="Bar Chart"):
        """Create a bar chart using Plotly"""
        try:
            fig = px.bar(data, x=x_col, y=y_col, title=title)
            fig.update_layout(
                xaxis_tickangle=-45,
                height=500,
                showlegend=False
            )
            return fig
        except Exception as e:
            st.error(f"Error creating bar chart: {str(e)}")
            return None
    
    def create_pie_chart(self, data, names_col, values_col, title="Pie Chart"):
        """Create a pie chart using Plotly"""
        try:
            fig = px.pie(data, names=names_col, values=values_col, title=title)
            fig.update_layout(height=500)
            return fig
        except Exception as e:
            st.error(f"Error creating pie chart: {str(e)}")
            return None

class ForecastAgent:
    """ForecastAgent handles predictive analytics and time-series forecasting operations."""
    
    def time_series_forecast(self, data, target_col, periods=30, date_col='listing_date'):
        """Generate time-series forecast using actual dates"""
        try:
            # Validate required columns
            if date_col not in data.columns:
                return None, f"Date column '{date_col}' not found in data"
            if target_col not in data.columns:
                return None, f"Target column '{target_col}' not found in data"
            
            # Clean and prepare data
            data_clean = data[[date_col, target_col]].dropna()
            if len(data_clean) < 10:
                return None, "Insufficient data for time-series forecasting (need at least 10 data points)"
            
            # Convert date column to datetime
            data_clean = data_clean.copy()
            data_clean[date_col] = pd.to_datetime(data_clean[date_col])
            data_clean = data_clean.sort_values(date_col)
            
            # Aggregate by date (average if multiple entries per date)
            daily_data = data_clean.groupby(date_col)[target_col].mean().reset_index()
            
            # Create complete date range to fill gaps
            date_range = pd.date_range(start=daily_data[date_col].min(), 
                                     end=daily_data[date_col].max(), 
                                     freq='D')
            complete_df = pd.DataFrame({date_col: date_range})
            daily_data = complete_df.merge(daily_data, on=date_col, how='left')
            
            # Forward fill missing values (updated syntax for newer pandas)
            daily_data[target_col] = daily_data[target_col].ffill().bfill()
            
            # Extract time features for modeling
            daily_data['days_since_start'] = (daily_data[date_col] - daily_data[date_col].min()).dt.days
            daily_data['day_of_week'] = daily_data[date_col].dt.dayofweek
            daily_data['month'] = daily_data[date_col].dt.month
            daily_data['quarter'] = daily_data[date_col].dt.quarter
            daily_data['year'] = daily_data[date_col].dt.year
            
            # Create cyclical features for seasonality
            daily_data['day_of_week_sin'] = np.sin(2 * np.pi * daily_data['day_of_week'] / 7)
            daily_data['day_of_week_cos'] = np.cos(2 * np.pi * daily_data['day_of_week'] / 7)
            daily_data['month_sin'] = np.sin(2 * np.pi * daily_data['month'] / 12)
            daily_data['month_cos'] = np.cos(2 * np.pi * daily_data['month'] / 12)
            
            # Prepare features for modeling
            feature_cols = ['days_since_start', 'day_of_week_sin', 'day_of_week_cos', 
                           'month_sin', 'month_cos']
            X = daily_data[feature_cols].values
            y = daily_data[target_col].values
            
            # Fit linear regression model with time and seasonal features
            model = LinearRegression()
            model.fit(X, y)
            
            # Generate future dates
            last_date = daily_data[date_col].max()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                       periods=periods, freq='D')
            
            # Create future features
            future_df = pd.DataFrame({date_col: future_dates})
            future_df['days_since_start'] = (future_df[date_col] - daily_data[date_col].min()).dt.days
            future_df['day_of_week'] = future_df[date_col].dt.dayofweek
            future_df['month'] = future_df[date_col].dt.month
            future_df['day_of_week_sin'] = np.sin(2 * np.pi * future_df['day_of_week'] / 7)
            future_df['day_of_week_cos'] = np.cos(2 * np.pi * future_df['day_of_week'] / 7)
            future_df['month_sin'] = np.sin(2 * np.pi * future_df['month'] / 12)
            future_df['month_cos'] = np.cos(2 * np.pi * future_df['month'] / 12)
            
            future_X = future_df[feature_cols].values
            future_predictions = model.predict(future_X)
            
            # Calculate model performance
            r2 = r2_score(y, model.predict(X))
            
            # Create result dataframes
            historical_df = daily_data[[date_col, target_col]].copy()
            historical_df.columns = ['date', 'actual']
            historical_df['type'] = 'historical'
            
            forecast_df = pd.DataFrame({
                'date': future_dates,
                'forecast': future_predictions,
                'type': 'forecast'
            })
            
            return {
                'historical': historical_df,
                'forecast': forecast_df,
                'r2_score': r2,
                'seasonal_analysis': self._analyze_seasonality(daily_data, target_col, date_col),
                'model_features': feature_cols,
                'date_range': f"{daily_data[date_col].min().strftime('%Y-%m-%d')} to {daily_data[date_col].max().strftime('%Y-%m-%d')}"
            }, f"Time-series forecast completed successfully (R² = {r2:.4f})"
            
        except Exception as e:
            return None, f"Error in time-series forecasting: {str(e)}"
    
    def _analyze_seasonality(self, data, target_col, date_col):
        """Analyze seasonal patterns in the data"""
        try:
            seasonal_analysis = {}
            
            # Monthly averages
            monthly_avg = data.groupby('month')[target_col].mean()
            seasonal_analysis['monthly_patterns'] = {
                'data': monthly_avg.to_dict(),
                'peak_month': monthly_avg.idxmax(),
                'low_month': monthly_avg.idxmin(),
                'variation': (monthly_avg.max() - monthly_avg.min()) / monthly_avg.mean() * 100
            }
            
            # Day of week patterns
            dow_avg = data.groupby('day_of_week')[target_col].mean()
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            seasonal_analysis['weekly_patterns'] = {
                'data': {day_names[i]: dow_avg.get(i, 0) for i in range(7)},
                'peak_day': day_names[dow_avg.idxmax()],
                'low_day': day_names[dow_avg.idxmin()]
            }
            
            # Quarterly patterns
            quarterly_avg = data.groupby('quarter')[target_col].mean()
            quarter_names = ['Q1', 'Q2', 'Q3', 'Q4']
            seasonal_analysis['quarterly_patterns'] = {
                'data': {quarter_names[i-1]: quarterly_avg.get(i, 0) for i in range(1, 5)},
                'peak_quarter': quarter_names[quarterly_avg.idxmax() - 1],
                'low_quarter': quarter_names[quarterly_avg.idxmin() - 1]
            }
            
            # Overall trend
            first_half = data[data[date_col] <= data[date_col].quantile(0.5)][target_col].mean()
            second_half = data[data[date_col] > data[date_col].quantile(0.5)][target_col].mean()
            trend_direction = "increasing" if second_half > first_half else "decreasing"
            trend_magnitude = abs(second_half - first_half) / first_half * 100
            
            seasonal_analysis['overall_trend'] = {
                'direction': trend_direction,
                'magnitude_percent': trend_magnitude
            }
            
            return seasonal_analysis
            
        except Exception as e:
            return {'error': f"Error analyzing seasonality: {str(e)}"}
    
    def advanced_forecast(self, data, target_col, periods=30, date_col='listing_date', include_confidence=True):
        """Advanced time-series forecast with confidence intervals"""
        try:
            # Get basic forecast first
            basic_result, message = self.time_series_forecast(data, target_col, periods, date_col)
            if basic_result is None:
                return None, message
            
            # Add confidence intervals using historical volatility
            historical_data = basic_result['historical']
            
            # Calculate rolling standard deviation
            window_size = min(30, len(historical_data) // 4)
            historical_data['rolling_std'] = historical_data['actual'].rolling(window=window_size, min_periods=1).std()
            
            # Use recent volatility for confidence intervals
            recent_std = historical_data['rolling_std'].tail(window_size).mean()
            
            # Add confidence intervals to forecast
            forecast_data = basic_result['forecast'].copy()
            forecast_data['upper_bound'] = forecast_data['forecast'] + (1.96 * recent_std)  # 95% confidence
            forecast_data['lower_bound'] = forecast_data['forecast'] - (1.96 * recent_std)
            
            basic_result['forecast'] = forecast_data
            basic_result['confidence_interval'] = f"95% confidence interval using σ = {recent_std:.2f}"
            
            return basic_result, message
            
        except Exception as e:
            return None, f"Error in advanced forecasting: {str(e)}"

class OrchestratorAgent:
    """OrchestratorAgent coordinates all other agents and handles query processing."""
    
    def __init__(self):
        self.data_agent = DataAgent()
        self.plot_agent = PlotAgent()
        self.forecast_agent = ForecastAgent()
    
    def extract_vehicle_filters(self, query, data):
        """Extract filters from natural language query"""
        filters = {}
        query_lower = query.lower()
        
        # Extract year using regex
        years = re.findall(r'\b(19\d{2}|20\d{2})\b', query)
        if years:
            filters['year'] = int(years[0])
        
        # Extract make using actual data - case insensitive exact match
        if 'make' in data.columns:
            actual_makes = data['make'].unique()
            for make in actual_makes:
                if pd.notna(make) and make.lower() in query_lower:
                    filters['make'] = make
                    break
        
        # FIXED: Extract fuel type using actual data - improved exact matching
        if 'fuel_type' in data.columns:
            actual_fuel_types = data['fuel_type'].unique()
            
            # Create a mapping of query terms to actual fuel types
            fuel_mappings = {}
            for fuel_type in actual_fuel_types:
                if pd.notna(fuel_type):
                    fuel_lower = fuel_type.lower()
                    fuel_mappings[fuel_lower] = fuel_type
                    
                    # Add common variations
                    if 'electric' in fuel_lower:
                        fuel_mappings['electric'] = fuel_type
                        fuel_mappings['ev'] = fuel_type
                    elif 'hybrid' in fuel_lower:
                        fuel_mappings['hybrid'] = fuel_type
                    elif 'gasoline' in fuel_lower or 'gas' in fuel_lower:
                        fuel_mappings['gasoline'] = fuel_type
                        fuel_mappings['gas'] = fuel_type
                    elif 'diesel' in fuel_lower:
                        fuel_mappings['diesel'] = fuel_type
            
            # Check for fuel type matches in query using word boundaries
            for query_term, actual_fuel in fuel_mappings.items():
                # Use word boundaries to ensure exact matches
                if re.search(r'\b' + re.escape(query_term) + r'\b', query_lower):
                    filters['fuel_type'] = actual_fuel
                    break
        
        # Extract state using actual data
        if 'state' in data.columns:
            actual_states = data['state'].unique()
            for state in actual_states:
                if pd.notna(state) and state.lower() in query_lower:
                    filters['state'] = state
                    break
        
        # Extract body style using actual data
        if 'body_style' in data.columns:
            actual_body_styles = data['body_style'].unique()
            for style in actual_body_styles:
                if pd.notna(style) and style.lower() in query_lower:
                    filters['body_style'] = style
                    break
        
        # Extract price range
        price_matches = re.findall(r'under\s+\$?(\d+(?:,\d{3})*)', query_lower)
        if price_matches:
            max_price = int(price_matches[0].replace(',', ''))
            filters['max_price'] = max_price
        
        price_matches = re.findall(r'over\s+\$?(\d+(?:,\d{3})*)', query_lower)
        if price_matches:
            min_price = int(price_matches[0].replace(',', ''))
            filters['min_price'] = min_price
        
        return filters
    
    def apply_filters(self, data, filters):
        """Apply extracted filters to the data"""
        filtered_data = data.copy()
        
        for column, value in filters.items():
            if column == 'max_price' and 'price' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['price'] <= value]
            elif column == 'min_price' and 'price' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['price'] >= value]
            elif column in filtered_data.columns:
                if isinstance(value, str):
                    # Exact string match, case insensitive
                    filtered_data = filtered_data[filtered_data[column].str.lower() == value.lower()]
                else:
                    # Exact numeric match
                    filtered_data = filtered_data[filtered_data[column] == value]
        
        return filtered_data
    
    def process_query(self, query, data):
        """Process natural language query and return results"""
        try:
            # Extract vehicle filters from query
            filters = self.extract_vehicle_filters(query, data)
            
            # Apply filters to data
            filtered_data = self.apply_filters(data, filters)
            
            if filtered_data.empty:
                filter_desc = ', '.join([f"{k}: {v}" for k, v in filters.items()]) if filters else "criteria"
                return None, f"No vehicles found matching {filter_desc}"
            
            # Determine action based on query keywords
            query_lower = query.lower()
            
            if any(word in query_lower for word in ['cheapest', 'lowest', 'least expensive']):
                numbers = re.findall(r'\d+', query)
                n = int(numbers[0]) if numbers else 10
                result_data = filtered_data.nsmallest(n, 'price').copy()
                action_desc = "cheapest"
                
            elif any(word in query_lower for word in ['top', 'expensive', 'highest', 'most expensive']):
                numbers = re.findall(r'\d+', query)
                n = int(numbers[0]) if numbers else 10
                result_data = filtered_data.nlargest(n, 'price').copy()
                action_desc = "most expensive"
                
            elif 'average' in query_lower or 'mean' in query_lower:
                # Calculate averages
                if 'price' in filtered_data.columns:
                    avg_price = filtered_data['price'].mean()
                    result_data = filtered_data.head(20).copy()
                    action_desc = f"average price: ${avg_price:,.2f}"
                else:
                    result_data = filtered_data.head(20).copy()
                    action_desc = "summary"
            else:
                # Default summary - show filtered results
                result_data = filtered_data.head(20).copy()
                action_desc = "summary"
            
            # Reset index to create proper row numbers starting from 1
            result_data.reset_index(drop=True, inplace=True)
            result_data.index = result_data.index + 1
            result_data.index.name = 'Row'
            
            # Create visualization
            viz_data = result_data.copy()
            fig = None
            
            if len(result_data) > 0:
                if 'make' in viz_data.columns and 'model' in viz_data.columns and 'price' in viz_data.columns:
                    # Create vehicle labels for bar chart
                    viz_data['vehicle_label'] = (
                        viz_data['year'].astype(str) + ' ' + 
                        viz_data['make'] + ' ' + 
                        viz_data['model']
                    )
                    fig = self.plot_agent.create_bar_chart(
                        viz_data.head(10), 'vehicle_label', 'price', 
                        f"{action_desc.title()} Vehicles"
                    )
                elif 'make' in filtered_data.columns:
                    # Pie chart by make distribution
                    make_counts = filtered_data['make'].value_counts().head(10)
                    fig = self.plot_agent.create_pie_chart(
                        pd.DataFrame({'make': make_counts.index, 'count': make_counts.values}),
                        'make', 'count', "Vehicle Distribution by Make"
                    )
            
            # Create description
            filter_desc = ', '.join([f"{k}: {v}" for k, v in filters.items()]) if filters else "all vehicles"
            description = f"{action_desc} {filter_desc}" if filters else f"{action_desc} vehicles"
            
            return {
                'data': result_data,
                'chart': fig,
                'description': description,
                'filters_applied': filters,
                'total_matches': len(filtered_data)
            }, "Query processed successfully"
            
        except Exception as e:
            return None, f"Error processing query: {str(e)}"

def format_dataframe_display(df):
    """Format dataframe for proper display with correct number formatting"""
    if df is None or df.empty:
        return df
        
    display_df = df.copy()
    
    # Format year column - no commas
    if 'year' in display_df.columns:
        display_df['year'] = display_df['year'].astype(int).astype(str)
    
    # Format price column - with dollar sign and commas
    if 'price' in display_df.columns:
        display_df['price'] = display_df['price'].apply(lambda x: f"${x:,.0f}")
    
    # Format other numeric columns that shouldn't have commas
    for col in display_df.columns:
        if col in ['days_on_market', 'inventory'] and col in display_df.columns:
            display_df[col] = display_df[col].astype(int).astype(str)
    
    return display_df

def display_app_info():
    """Display application information and credits"""
    with st.expander("ℹ️ About This Application"):
        st.markdown(f"""
        **{APP_NAME}** v{APP_VERSION}
        
        This application uses a multi-agent architecture to analyze vehicle data:
        
        - 🤖 **DataAgent**: Handles data loading, cleaning, and preprocessing
        - 📊 **PlotAgent**: Creates interactive visualizations and charts  
        - 🔮 **ForecastAgent**: Generates predictive analytics and forecasts
        - 🧠 **OrchestratorAgent**: Coordinates all agents and processes queries
        
        **Features:**
        - Natural language query processing
        - Real-time data filtering and analysis
        - Interactive Plotly visualizations
        - Time-series forecasting with seasonal analysis
        - Export capabilities for results
        
        **Supported Query Types:**
        - "top 10 electric vehicles"
        - "cheapest Honda cars from 2020"
        - "hybrid vehicles under $30,000"
        - "forecast vehicle prices"
        
        **Version History:**
        - v2.2.0: Added proper time-series forecasting with seasonal analysis
        - v2.1.0: Fixed electric vehicle filtering and persistent status messages
        - v2.0.0: Multi-agent architecture implementation
        - v1.0.0: Initial release
        
        Released: {RELEASE_DATE}
        """)

def main():
    """Main application function"""
    st.set_page_config(
        page_title=f"{APP_NAME} v{APP_VERSION}", 
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🚗"
    )
    
    # Header with version info
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("## 🚗 Vehicle Analysis Multi-Agent System")
    with col2:
        st.markdown(f"**Version {APP_VERSION}**")
        st.caption(f"Released: {RELEASE_DATE}")
    
    # Initialize session state variables
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = OrchestratorAgent()
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'csv_filename' not in st.session_state:
        st.session_state.csv_filename = None
    if 'query_results' not in st.session_state:
        st.session_state.query_results = None
    if 'status_message' not in st.session_state:
        st.session_state.status_message = None
    if 'status_type' not in st.session_state:
        st.session_state.status_type = None
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### 📁 Data Management")
        
        # Data upload section (moved up)
        if st.session_state.uploaded_data is not None:
            st.success(f"✅ {st.session_state.csv_filename}")
            st.caption(f"{st.session_state.uploaded_data.shape[0]} rows × {st.session_state.uploaded_data.shape[1]} cols")
            
            # Data summary
            if st.button("📊 Show Data Summary"):
                summary = st.session_state.orchestrator.data_agent.get_data_summary()
                if summary:
                    st.json(summary)
        
        uploaded_file = st.file_uploader("Upload CSV File", type="csv", help="Upload your vehicle data CSV file")
        
        # Handle file upload
        if uploaded_file is not None and st.session_state.csv_filename != uploaded_file.name:
            with st.spinner("Loading and cleaning data..."):
                success, message = st.session_state.orchestrator.data_agent.load_csv(uploaded_file)
                if success:
                    st.session_state.uploaded_data = st.session_state.orchestrator.data_agent.data.copy()
                    st.session_state.csv_filename = uploaded_file.name
                    
                    # Clean the data
                    clean_success, clean_message = st.session_state.orchestrator.data_agent.clean_data()
                    if clean_success:
                        st.session_state.uploaded_data = st.session_state.orchestrator.data_agent.data.copy()
                        st.success(f"✅ {message}")
                    else:
                        st.warning(f"⚠️ {clean_message}")
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
        
        # Update orchestrator data if available
        if st.session_state.uploaded_data is not None:
            st.session_state.orchestrator.data_agent.data = st.session_state.uploaded_data.copy()
        
        st.markdown("---")
        st.markdown("### 📱 App Info")
        st.markdown(f"**{APP_NAME}**")
        st.markdown(f"Version: `{APP_VERSION}`")
        st.markdown(f"Released: {RELEASE_DATE}")
        
        # Download source code button disabled
        # st.download_button(
        #     label="📥 Download Source Code",
        #     data=source_code,
        #     file_name=f"vehicle_analysis_v{APP_VERSION.replace('.', '_')}.py",
        #     mime="text/plain",
        #     help="Download the complete Python source code"
        # )
        
        # App information
        st.markdown("---")
        display_app_info()
    
    # Main content area
    if st.session_state.uploaded_data is None:
        # Welcome screen
        st.info("👆 Upload a CSV file to begin vehicle analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔍 Example Queries")
            st.code("cheapest Honda cars")
            st.code("top 10 expensive vehicles")
            st.code("Toyota from 2020")
            st.code("top 10 electric vehicles")
            st.code("BMW cars from 2021")
            st.code("hybrid vehicles under $30000")
        
        with col2:
            st.markdown("### 🚀 Getting Started")
            st.markdown("""
            1. **Upload** your vehicle CSV file
            2. **Query** your data using natural language
            3. **Explore** results in Data and Charts tabs
            4. **Forecast** trends in the Forecast tab
            
            **Required CSV Columns:**
            - `price` (numeric)
            - `make` (text)
            - `model` (text)
            - `year` (numeric)
            - `listing_date` (date, for forecasting)
            - `fuel_type` (text, optional)
            - `state` (text, optional)
            """)
    
    else:
        # Main application tabs
        data = st.session_state.uploaded_data
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Query", "📋 Data", "📊 Charts", "🔮 Forecast", "📈 Analytics"])
        
        with tab1:
            st.markdown("### 🔍 Query Your Data")
            
            # Simple approach - back to text input
            query = st.text_input(
                "Ask about your vehicles:", 
                placeholder="e.g., top 10 electric vehicles",
                help="Use natural language to query your vehicle data"
            )
            
            # Example queries section
            st.markdown("**Example Queries:**")
            col1, col2, col3 = st.columns(3)
            
            # Use st.code to show clickable examples
            with col1:
                st.code("cheapest Honda cars")
                st.code("top 10 expensive vehicles")
            with col2:
                st.code("Toyota from 2020") 
                st.code("top 10 electric vehicles")
            with col3:
                st.code("BMW cars from 2021")
                st.code("hybrid vehicles under $30000")
            
            st.info("💡 **How to use:** Copy any example query above and paste it into the text box, then click Process Query.")
            
            # Process and Clear buttons
            col1, col2 = st.columns([3, 1])
            with col1:
                process_clicked = st.button("🚀 Process Query", type="primary", use_container_width=True)
            with col2:
                if st.button("🗑️ Clear Results", use_container_width=True):
                    st.session_state.query_results = None
                    st.session_state.status_message = None
                    st.session_state.status_type = None
            
            # Process query when button is clicked - fix for first click
            if process_clicked:
                if query and query.strip():
                    query_text = query.strip()
                    # Clear previous results
                    st.session_state.query_results = None
                    st.session_state.status_message = None
                    st.session_state.status_type = None
                    
                    # Show progress
                    with st.spinner("Processing query..."):
                        result, message = st.session_state.orchestrator.process_query(query_text, data)
                    
                    if result:
                        st.session_state.query_results = result
                        st.session_state.status_message = "✅ Query processed successfully! Check Data and Charts tabs for results."
                        st.session_state.status_type = "success"
                        
                        # Show filters applied
                        if result.get('filters_applied'):
                            filters_text = ', '.join([f"{k}: {v}" for k, v in result['filters_applied'].items()])
                            st.info(f"🔍 **Filters applied:** {filters_text}")
                        
                        # Show stats
                        if result.get('total_matches'):
                            st.info(f"📊 **Found {result['total_matches']} total matches, showing top {len(result['data'])} results**")
                    else:
                        st.session_state.status_message = f"❌ {message}"
                        st.session_state.status_type = "error"
                else:
                    st.warning("⚠️ Please enter a query")
            
            # Show status message
            if st.session_state.status_message:
                if st.session_state.status_type == "success":
                    st.success(st.session_state.status_message)
                elif st.session_state.status_type == "error":
                    st.error(st.session_state.status_message)
        
        with tab2:
            st.markdown("### 📋 Data Results")
            
            if st.session_state.query_results:
                result = st.session_state.query_results
                
                # Results header
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Query Analysis:** {result['description']}")
                with col2:
                    if result.get('data') is not None and len(result['data']) > 0:
                        # Export button
                        csv_data = result['data'].to_csv(index=True)
                        st.download_button(
                            label="📥 Export CSV",
                            data=csv_data,
                            file_name=f"vehicle_query_results_{int(time.time())}.csv",
                            mime="text/csv"
                        )
                
                if result.get('data') is not None and len(result['data']) > 0:
                    # Format the dataframe for display
                    display_data = format_dataframe_display(result['data'])
                    
                    # Reorder columns for better vehicle display
                    if 'make' in display_data.columns and 'model' in display_data.columns:
                        preferred_cols = ['year', 'make', 'model', 'price', 'state', 'fuel_type', 'body_style']
                        available_cols = [col for col in preferred_cols if col in display_data.columns]
                        other_cols = [col for col in display_data.columns if col not in available_cols]
                        column_order = available_cols + other_cols
                        display_data = display_data[column_order]
                    
                    st.dataframe(display_data, use_container_width=True, height=400)
                    
                    # Quick statistics
                    st.markdown("#### 📊 Quick Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Results Found", len(result['data']))
                    with col2:
                        if 'price' in result['data'].columns:
                            avg_price = result['data']['price'].mean()
                            st.metric("Average Price", f"${avg_price:,.0f}")
                    with col3:
                        if 'year' in result['data'].columns:
                            year_range = f"{result['data']['year'].min()}-{result['data']['year'].max()}"
                            st.metric("Year Range", year_range)
                    with col4:
                        if 'make' in result['data'].columns:
                            unique_makes = result['data']['make'].nunique()
                            st.metric("Unique Makes", unique_makes)
                else:
                    st.warning("No data available from query results")
            else:
                st.info("Process a query in the Query tab to see data results here")
        
        with tab3:
            st.markdown("### 📊 Charts & Visualizations")
            
            if st.session_state.query_results:
                result = st.session_state.query_results
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Analysis:** {result['description']}")
                with col2:
                    chart_type = st.selectbox("Chart Type", ["Auto", "Bar Chart", "Pie Chart"], key="chart_type")
                
                # Display main chart
                if result.get('chart'):
                    st.plotly_chart(result['chart'], use_container_width=True)
                
                # Additional charts based on data
                if result.get('data') is not None and len(result['data']) > 0:
                    data_for_charts = result['data']
                    
                    if 'fuel_type' in data_for_charts.columns:
                        st.markdown("#### 🔋 Fuel Type Distribution")
                        fuel_counts = data_for_charts['fuel_type'].value_counts()
                        fig_fuel = px.bar(
                            x=fuel_counts.index, 
                            y=fuel_counts.values,
                            title="Vehicle Count by Fuel Type",
                            labels={'x': 'Fuel Type', 'y': 'Count'}
                        )
                        st.plotly_chart(fig_fuel, use_container_width=True)
            else:
                st.info("Process a query in the Query tab to see charts here")
        
        with tab4:
            st.markdown("### 🔮 Time-Series Forecasting & Seasonal Analysis")
            
            # Check if listing_date column exists
            if 'listing_date' not in data.columns:
                st.error("❌ 'listing_date' column not found in the dataset. Time-series forecasting requires a date column.")
                st.info("💡 Please ensure your CSV has a 'listing_date' column with date values.")
            else:
                # Get numeric columns for forecasting
                numeric_cols = [col for col in data.select_dtypes(include=[np.number]).columns 
                               if col.lower() not in ['year']]
                
                if not numeric_cols:
                    st.warning("No suitable numeric columns found for forecasting.")
                else:
                    # Forecasting configuration
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        target_column = st.selectbox("Target Column:", numeric_cols, key="ts_forecast_col")
                    
                    with col2:
                        forecast_periods = st.number_input("Forecast Days:", min_value=1, max_value=365, value=30, key="ts_forecast_periods")
                    
                    with col3:
                        forecast_type = st.selectbox("Forecast Type:", ["Time-Series", "Advanced (with CI)"], key="ts_forecast_type")
                    
                    with col4:
                        use_filtered = False
                        if st.session_state.query_results and st.session_state.query_results.get('data') is not None:
                            use_filtered = st.checkbox("Use filtered data", value=False, key="ts_use_filtered")
                    
                    # Generate forecast button
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        forecast_button = st.button("🚀 Generate Time-Series Forecast", type="primary", use_container_width=True)
                    with col2:
                        show_seasonal = st.checkbox("Show Seasonal Analysis", value=True, key="show_seasonal")
                    
                    if forecast_button:
                        with st.spinner("🔄 Generating time-series forecast with seasonal analysis..."):
                            # Choose data source
                            forecast_data = st.session_state.query_results['data'] if use_filtered and st.session_state.query_results else data
                            
                            # Generate appropriate forecast
                            if forecast_type == "Advanced (with CI)":
                                forecast_result, msg = st.session_state.orchestrator.forecast_agent.advanced_forecast(
                                    forecast_data, target_column, forecast_periods, 'listing_date'
                                )
                            else:
                                forecast_result, msg = st.session_state.orchestrator.forecast_agent.time_series_forecast(
                                    forecast_data, target_column, forecast_periods, 'listing_date'
                                )
                            
                            if forecast_result:
                                st.success(msg)
                                
                                # Display forecast visualization
                                st.markdown("#### 📈 Time-Series Forecast")
                                
                                fig = go.Figure()
                                
                                # Historical data
                                hist_data = forecast_result['historical']
                                fig.add_trace(go.Scatter(
                                    x=hist_data['date'],
                                    y=hist_data['actual'],
                                    mode='lines+markers',
                                    name='Historical Data',
                                    line=dict(color='blue', width=2),
                                    marker=dict(size=4)
                                ))
                                
                                # Forecast data
                                forecast_data_viz = forecast_result['forecast']
                                fig.add_trace(go.Scatter(
                                    x=forecast_data_viz['date'],
                                    y=forecast_data_viz['forecast'],
                                    mode='lines+markers',
                                    name='Forecast',
                                    line=dict(color='red', dash='dash', width=2),
                                    marker=dict(size=4)
                                ))
                                
                                # Add confidence intervals if available
                                if 'upper_bound' in forecast_data_viz.columns:
                                    fig.add_trace(go.Scatter(
                                        x=forecast_data_viz['date'],
                                        y=forecast_data_viz['upper_bound'],
                                        mode='lines',
                                        name='Upper Bound (95%)',
                                        line=dict(color='rgba(255,0,0,0.3)', width=1),
                                        showlegend=False
                                    ))
                                    fig.add_trace(go.Scatter(
                                        x=forecast_data_viz['date'],
                                        y=forecast_data_viz['lower_bound'],
                                        mode='lines',
                                        name='Lower Bound (95%)',
                                        line=dict(color='rgba(255,0,0,0.3)', width=1),
                                        fill='tonexty',
                                        fillcolor='rgba(255,0,0,0.1)',
                                        showlegend=True
                                    ))
                                
                                fig.update_layout(
                                    title=f"Time-Series Forecast: {target_column.title()}",
                                    xaxis_title="Date",
                                    yaxis_title=target_column.title(),
                                    hovermode='x unified',
                                    height=500,
                                    showlegend=True
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Forecast statistics
                                st.markdown("#### 📊 Forecast Statistics")
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Model R² Score", f"{forecast_result['r2_score']:.4f}")
                                with col2:
                                    st.metric("Forecast Period", f"{forecast_periods} days")
                                with col3:
                                    st.metric("Data Range", forecast_result.get('date_range', 'N/A'))
                                with col4:
                                    st.metric("Historical Points", len(forecast_result['historical']))
                                
                                # Seasonal Analysis
                                if show_seasonal and 'seasonal_analysis' in forecast_result:
                                    st.markdown("#### 🌊 Seasonal Analysis")
                                    seasonal = forecast_result['seasonal_analysis']
                                    
                                    if 'error' not in seasonal:
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            # Monthly patterns
                                            if 'monthly_patterns' in seasonal:
                                                st.markdown("**📅 Monthly Patterns**")
                                                monthly_data = seasonal['monthly_patterns']['data']
                                                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                                                
                                                fig_monthly = px.bar(
                                                    x=month_names,
                                                    y=[monthly_data.get(i+1, 0) for i in range(12)],
                                                    title="Average by Month",
                                                    labels={'x': 'Month', 'y': f'Avg {target_column}'}
                                                )
                                                st.plotly_chart(fig_monthly, use_container_width=True)
                                                
                                                # Monthly insights
                                                peak_month = month_names[seasonal['monthly_patterns']['peak_month'] - 1]
                                                low_month = month_names[seasonal['monthly_patterns']['low_month'] - 1]
                                                variation = seasonal['monthly_patterns']['variation']
                                                
                                                st.info(f"📈 **Peak Month:** {peak_month}")
                                                st.info(f"📉 **Low Month:** {low_month}")
                                                st.info(f"📊 **Monthly Variation:** {variation:.1f}%")
                                        
                                        with col2:
                                            # Weekly patterns
                                            if 'weekly_patterns' in seasonal:
                                                st.markdown("**📅 Weekly Patterns**")
                                                weekly_data = seasonal['weekly_patterns']['data']
                                                
                                                fig_weekly = px.bar(
                                                    x=list(weekly_data.keys()),
                                                    y=list(weekly_data.values()),
                                                    title="Average by Day of Week",
                                                    labels={'x': 'Day', 'y': f'Avg {target_column}'}
                                                )
                                                st.plotly_chart(fig_weekly, use_container_width=True)
                                                
                                                # Weekly insights
                                                peak_day = seasonal['weekly_patterns']['peak_day']
                                                low_day = seasonal['weekly_patterns']['low_day']
                                                
                                                st.info(f"📈 **Peak Day:** {peak_day}")
                                                st.info(f"📉 **Low Day:** {low_day}")
                                    else:
                                        st.error(f"Seasonal analysis error: {seasonal['error']}")
                                
                                # Data tables
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("#### 📋 Recent Historical Data")
                                    recent_hist = hist_data.tail(10).copy()
                                    recent_hist['date'] = recent_hist['date'].dt.strftime('%Y-%m-%d')
                                    st.dataframe(recent_hist, height=300, hide_index=True)
                                
                                with col2:
                                    st.markdown("#### 🔮 Forecast Data")
                                    forecast_display = forecast_data_viz[['date', 'forecast']].head(10).copy()
                                    forecast_display['date'] = forecast_display['date'].dt.strftime('%Y-%m-%d')
                                    forecast_display['forecast'] = forecast_display['forecast'].round(2)
                                    st.dataframe(forecast_display, height=300, hide_index=True)
                                    
                                    # Download forecast data
                                    forecast_csv = forecast_data_viz.copy()
                                    forecast_csv['date'] = forecast_csv['date'].dt.strftime('%Y-%m-%d')
                                    forecast_csv_str = forecast_csv.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download Forecast",
                                        data=forecast_csv_str,
                                        file_name=f"timeseries_forecast_{target_column}_{int(time.time())}.csv",
                                        mime="text/csv"
                                    )
                            else:
                                st.error(msg)
        
        with tab5:
            st.markdown("### 📈 Advanced Analytics")
            
            # Data overview
            st.markdown("#### 📊 Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", data.shape[0])
            with col2:
                st.metric("Total Columns", data.shape[1])
            with col3:
                if 'make' in data.columns:
                    st.metric("Unique Makes", data['make'].nunique())
            with col4:
                if 'year' in data.columns:
                    year_span = data['year'].max() - data['year'].min()
                    st.metric("Year Span", f"{year_span} years")


if __name__ == "__main__":
    main()