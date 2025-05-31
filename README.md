PROJECT TITLE
Vehicle Data Management Application

PROBLEM STATEMENT
In the Automotive world, customer feedback / data is not directly available to manufacturing OEMs; this is an important channel of data for OEMs to respond to customers

PROJECT DESCRIPTION
This project has been worked upon in two parts:
1. Part A: Uses OpenAI LLM calls and Pinecone for RAG to generate data that is representative of customer feedback from the internet
2. Part B: Uses the data from Part A to allow users to query for various analytics, plots and forecasts
3. Streamlit UI
4. I have been unable to upload Part A correctly. Will do so in a day

APPLICATION FEATURES
- OpenAI based dataset generated for the application
- Pinecone RAG
- Natural language query processing for vehicle data
- Multi-agent system architecture
- Data visualization with Plotly
- Time-series forecasting with seasonal analysis
- Comprehensive vehicle filtering (make, model, year, fuel type, state, body style)
- Real-time data analysis and charts
- Monthly, weekly, and quarterly seasonal pattern analysis
- Confidence intervals for forecasts

INSTALLATION:
pip install -r requirements.txt

USAGE:
streamlit run vehicle_analysis.py

SCREENSHOTS:
A powerpoint with screenshots is provided:
1. Shows csv from part A upload
2. UI for providing NLP based query
3. Example Queries
4. Forecast capabilities (Time-series)
4. Charts and Data based on Query

STREAMLIT URL
https://veh-mgmt-app.streamlit.app/


