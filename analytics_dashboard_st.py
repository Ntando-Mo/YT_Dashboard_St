# IMPORTS

import pandas as pd 
import numpy as np 
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime

# Helper Functions

def style_negative(v, props=''):
    """Applies red color to negative values in a DataFrame style object."""
    try: 
        # Robustly handle potential formatted strings (e.g., currency, percent)
        if isinstance(v, str):
            v = float(v.strip('$%').replace(',', '').strip('%'))
        return props if v < 0 else None
    except:
        return None
    
def style_positive(v, props=''):
    """Applies green color to positive values in a DataFrame style object."""
    try: 
        if isinstance(v, str):
            v = float(v.strip('$%').replace(',', '').strip('%'))
        return props if v > 0 else None
    except:
        return None    

def audience_simple(country):
    """Simplifies country codes into major regions or 'Other'."""
    if country in ['US', 'CA']:
        return 'USA/Canada'
    elif country in ['IN']:
        return 'India'
    elif country in ['GB', 'DE', 'FR', 'IT', 'ES']:
        return 'Europe'
    else:
        return 'Other'

# Data loading and initial engineering

@st.cache
def load_data():
    """Loads raw data, renames columns, and performs core type conversions."""
    # Load and clean df_agg
    df_agg = pd.read_csv('Aggregated_Metrics_By_Video.csv').iloc[1:,:]
    df_agg.columns = ['Video','Video title','Video publish time','Comments added','Shares','Dislikes','Likes',
                      'Subscribers lost','Subscribers gained','RPM(USD)','CPM(USD)','Average % viewed','Average view duration',
                      'Views','Watch time (hours)','Subscribers','Your estimated revenue (USD)','Impressions','Impressions ctr(%)']
    
    # Type Conversions
    df_agg['Video publish time'] = pd.to_datetime(df_agg['Video publish time'], format='%b %d, %Y')
    df_agg['Average view duration'] = df_agg['Average view duration'].apply(lambda x: datetime.strptime(x,'%H:%M:%S'))
    
    # Calculate Avg_duration
    df_agg['Avg_duration'] = df_agg['Average view duration'].apply(lambda x: x.second + x.minute*60 + x.hour*3600)
    
    # Calculate Engagement and Views per Subscriber Gained
    df_agg['Engagement_ratio'] = (df_agg['Comments added'] + df_agg['Shares'] + df_agg['Dislikes'] + df_agg['Likes']) / df_agg.Views
    df_agg['Views / sub gained'] = df_agg['Views'] / df_agg['Subscribers gained']
    df_agg.sort_values('Video publish time', ascending=False, inplace=True)
    
    # Load other dataframes
    df_agg_sub = pd.read_csv('Aggregated_Metrics_By_Country_And_Subscriber_Status.csv')
    df_comments = pd.read_csv('Aggregated_Metrics_By_Video.csv')
    df_time = pd.read_csv('Video_Performance_Over_Time.csv')
    
    # Ensure df_time date column is correct
    df_time['Date'] = pd.to_datetime(df_time['Date'], format='mixed', dayfirst=True)
    
    return df_agg, df_agg_sub, df_comments, df_time

# Load dataframes
df_agg, df_agg_sub, df_comments, df_time = load_data()


# Data engineering for dashboard views
# Overall Median for Individual Video Comparison
all_median = df_agg.median(numeric_only=True)

# Baseline 12-Month Median (used for Aggregate Metrics Delta denominator)
metric_date_12mo = df_agg['Video publish time'].max() - pd.DateOffset(months=12)
median_agg_12mo = df_agg[df_agg['Video publish time'] >= metric_date_12mo].median(numeric_only=True)


# Time-Series Data for Individual Analysis 

# Join df_time with video publish time from df_agg
df_time['Video publish time'] = df_time['Video Title'].map(df_agg.set_index('Video title')['Video publish time'])

# FIX: Calculate days_published on full datetime objects to avoid type error
df_time_diff = df_time.copy()
df_time_diff['days_published'] = (df_time_diff['Date'] - df_time_diff['Video publish time']).dt.days


# Calculate views_cumulative (Percentiles for Trend Chart)
views_cumulative = df_time_diff.groupby(['Video Title','days_published']).agg({'Views':'sum'}).reset_index()
views_cumulative['cumulative_views'] = views_cumulative.groupby('Video Title')['Views'].cumsum()

# Group by days_published to calculate channel benchmarks
views_cumulative = views_cumulative.groupby('days_published')['cumulative_views'].agg([
    lambda x: x.quantile(0.2), 
    'median',                  
    lambda x: x.quantile(0.8)  
]).reset_index()

views_cumulative.columns = ['days_published', '20pct_views', 'median_views', '80pct_views']

# Streamlit app layout

# Sidebar Selection
st.sidebar.title("Navigation")
add_sidebar = st.sidebar.selectbox('Select View:', ('Aggregate Metrics','Individual Video Analysis'))

# Aggregate Metrics View

if add_sidebar == 'Aggregate Metrics':
    st.title("Ntando The Analyst YouTube Aggregated Data 📈")

    display_metrics = [
        'Views', 'Likes', 'Subscribers', 'Shares', 'Comments added', 
        'RPM(USD)', 'Average % viewed', 'Avg_duration', 
        'Engagement_ratio', 'Views / sub gained'
    ]

    # Calculate 6-month and 12-month medians
    df_agg_filtered = df_agg[['Video publish time'] + display_metrics].copy()
    metric_date_6mo = df_agg_filtered['Video publish time'].max() - pd.DateOffset(months=6)
    metrics_6mo_data = df_agg_filtered[df_agg_filtered['Video publish time'] >= metric_date_6mo][display_metrics]
    metric_medians6mo = metrics_6mo_data.median()
    metric_medians12mo = median_agg_12mo # Use pre-calculated 12mo median
    
    st.subheader('6-Month Performance vs. Prior 6 Months (Median)')
    col1, col2, col3, col4, col5 = st.columns(5)
    columns = [col1, col2, col3, col4, col5]
    count = 0
    
    # Display Metric Cards
    for i in display_metrics:
        if i in metric_medians6mo.index and metric_medians12mo[i] != 0:
            with columns[count]:
                delta = (metric_medians6mo[i] - metric_medians12mo[i]) / metric_medians12mo[i]
                
                # Conditional Formatting
                if i == 'Average % viewed':
                    # 1. Multiply by 100 to get the percentage value (e.g., 0.4122 * 100 = 41.22)
                    current_value = metric_medians6mo[i]
                    
                    # 2. Format the calculated number into the final string, e.g., "41.22%"
                    formatted_value_str = "{:.2f}%".format(current_value) 
                    
                    st.metric(label=i, 
                              value=formatted_value_str,  # <-- Pass the definitive string
                              delta="{:.2%}".format(delta))
                    
                elif 'RPM' in i or 'CPM' in i or 'revenue' in i:
                    current_value = round(metric_medians6mo[i], 2)
                    st.metric(label=i, value="${:,.2f}".format(current_value), delta="{:.2%}".format(delta))
                    
                elif i == 'Avg_duration':
                    # Format seconds into MM:SS (e.g., 175 seconds -> 02:55)
                    seconds = int(metric_medians6mo[i])
                    minutes = seconds // 60
                    seconds_left = seconds % 60
                    duration_str = f"{minutes:02d}:{seconds_left:02d}"
                    st.metric(label=i, value=duration_str, delta="{:.2%}".format(delta))
                    
                else:
                    current_value = round(metric_medians6mo[i], 1)
                    st.metric(label=i, value="{:,.1f}".format(current_value), delta="{:.2%}".format(delta))
                
                count += 1
                if count >= 5:
                    count = 0

    # Individual Video Performance Table (Matching Image Style)
    st.markdown('---')
    st.subheader('Individual Video Performance vs. Channel Median')

    df_agg_diff = df_agg.copy()
    df_agg_diff['Publish_date'] = df_agg_diff['Video publish time'].dt.date
    
    df_for_display = df_agg_diff[['Video title','Publish_date']].copy()
    comparison_cols = ['Views','Likes','Subscribers','Shares','Comments added','RPM(USD)','Average % viewed',
                       'Engagement_ratio','Views / sub gained']

    # Calculate Percentage Difference Ratio: (Video Value / Overall Median) - 1
    for col in comparison_cols:
        if all_median[col] != 0:
            df_for_display[col] = (df_agg_diff[col] / all_median[col]) - 1
        else:
             df_for_display[col] = 0.0

    df_for_display['Avg_duration'] = df_agg_diff['Avg_duration']
    
    # Define DataFrame Formatting
    df_to_format = {}
    for col in comparison_cols:
        df_to_format[col] = '{:.6f}' # Format ratio as large float
        
    df_to_format['Avg_duration'] = lambda x: f"{int(x // 60):02d}:{int(x % 60):02d}"

    # Display the DataFrame with conditional styling
    st.dataframe(
        df_for_display.style
            .hide(axis="index")
            .applymap(style_negative, props='color:red;', subset=comparison_cols) 
            .applymap(style_positive, props='color:green;', subset=comparison_cols) 
            .format(df_to_format)
    )

# Individual Video Analysis View

if add_sidebar == 'Individual Video Analysis':
    st.title("Individual Video Performance Analysis 🔎")
    
    videos = tuple(df_agg['Video title'].unique())
    st.write("### Audience and Trend Breakdown")
    video_select = st.selectbox('Pick a Video:', videos)
    
    # Audience Chart (Chart 1)
    st.subheader(f'Audience Breakdown for: {video_select}')
    agg_sub_filtered = df_agg_sub[df_agg_sub['Video Title'] == video_select].copy()
    
    # Use .loc to avoid SettingWithCopyWarning
    agg_sub_filtered.loc[:, 'Country'] = agg_sub_filtered['Country Code'].apply(audience_simple)
    agg_sub_filtered.sort_values('Is Subscribed', inplace=True) 
    
    if not agg_sub_filtered.empty:
        fig = px.bar(
            agg_sub_filtered, 
            x='Views', 
            y='Is Subscribed', 
            color='Country', 
            orientation='h',
            title='Views by Subscriber Status and Region'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No audience data found for '{video_select}'.")

    st.markdown('---')

    # Trend Comparison Chart (Chart 2)
    st.subheader('30-Day View Trend Comparison')

    agg_time_filtered = df_time_diff[df_time_diff['Video Title'] == video_select]
    first_30 = agg_time_filtered[agg_time_filtered['days_published'].between(0,30)]
    first_30 = first_30.sort_values('days_published')
    
    if not first_30.empty:
        fig2 = go.Figure()
        
        # Add Percentile/Median reference lines
        fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['20pct_views'],
                            mode='lines', name='20th percentile', line=dict(color='purple', dash='dash')))
        fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['median_views'],
                                mode='lines', name='50th percentile', line=dict(color='white', dash='dash')))
        fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['80pct_views'],
                                mode='lines', name='80th percentile', line=dict(color='royalblue', dash='dash')))
        
        # Add Current Video trend line
        fig2.add_trace(go.Scatter(x=first_30['days_published'], y=first_30['Views'].cumsum(),
                                mode='lines', name='Current Video', line=dict(color='firebrick', width=6)))
            
        fig2.update_layout(
            title='Cumulative Views (First 30 Days) vs. Channel Benchmarks',
            xaxis_title='Days Since Published',
            yaxis_title='Cumulative Views',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning(f"Time-series data for '{video_select}' is incomplete or missing.")