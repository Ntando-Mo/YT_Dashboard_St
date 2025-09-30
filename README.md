# Ntando The Analyst - YouTube Dashboard 📈

This is a Streamlit application designed to visualize and analyze aggregated and time-series performance data from a YouTube channel. The dashboard allows for comparison of overall channel performance metrics against defined benchmarks (like 6-month and 12-month medians) and provides a detailed breakdown for individual video performance and audience demographics.

## Features

* **Aggregate Metrics View:** Displays key channel metrics (Views, Engagement, RPM, etc.) comparing the last 6 months' median performance against the previous 12 months' median.
* **Individual Video Analysis:** Allows users to select a specific video to see:
    * **Audience Breakdown:** Views segmented by subscriber status and simplified geographical region.
    * **Trend Comparison:** 30-day cumulative view trend compared against channel percentiles (20th, 50th, 80th).
* **Data Cleaning and Robustness:** Implements robust data loading logic to handle common CSV data inconsistencies (e.g., stripping whitespace from column names and video titles, safe date conversion).

## Project Structure

The project relies on a single Python script and three core CSV data files.

| File Name | Description |
| :--- | :--- |
| `analytics_dashboard_st.py` | The main Streamlit Python script containing all data loading, engineering, and dashboard layout logic. |
| `requirements.txt` | Lists all necessary Python packages (`streamlit`, `pandas`, `numpy`, `plotly`). |
| `Aggregated_Metrics_By_Video.csv` | Contains lifetime performance metrics for each video (used for `df_agg` and `df_comments`). |
| `Aggregated_Metrics_By_Country_And_Subscriber_Status.csv` | Contains audience breakdown data (used for `df_agg_sub`). |
| `Video_Performance_Over_Time.csv` | Contains daily or periodic time-series data for videos (used for `df_time`). |

## Installation and Local Setup

To run this dashboard locally, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/Ntando-Mo/YT_Dashboard_St
cd yt_dashboard_st
```

### 2. Set up the Python Environment

# Create a virtual environment (using conda or venv)
conda create -n YT_Dashboard_St python=3.9 
conda activate YT_Dashboard_St

# OR using venv
python -m venv venv
source venv/bin/activate # On Windows, use `venv\Scripts\activate`

### 3. Install Dependencies

pip install -r requirements.txt

### 4. Add Data Files

Ensure the three required CSV files (Aggregated_Metrics_By_Video.csv, Aggregated_Metrics_By_Country_And_Subscriber_Status.csv, and Video_Performance_Over_Time.csv) are placed directly in the project's root directory.

### 5. Run the application

streamlit run analytics_dashboard_st.py
