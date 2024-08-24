import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Load the data
benin_data = pd.read_csv('data/benin-malanville.csv')
togo_data = pd.read_csv('data/togo-dapaong_qc.csv')
sierraleone_data = pd.read_csv('data/sierraleone-bumbuna.csv')

# Add 'Country' column to each DataFrame
benin_data['Country'] = 'Benin Malanville'
sierraleone_data['Country'] = 'Sierra Leone Bumbuna'
togo_data['Country'] = 'Togo Dapaong'

def check_null_values(data):
    null_count = data.isnull().sum()
    null_percentage = (data.isnull().sum() / len(data)) * 100
    null_report = pd.DataFrame({
        'Missing Values': null_count,
        'Percentage (%)': null_percentage
    })
    null_report = null_report[null_report['Missing Values'] > 0]
    return null_report

def convert_to_datetime(df, column_name='Timestamp'):
    df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
    return df

def extract_datetime_components(df, timestamp_column='Timestamp'):
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')
    df["Year"] = df[timestamp_column].dt.year
    df["Month"] = df[timestamp_column].dt.month
    df["Day"] = df[timestamp_column].dt.day
    df["Hour"] = df[timestamp_column].dt.hour
    return df

def replace_negative_values(df, columns):
    for column in columns:
        df[column] = df[column].apply(lambda x: max(x, 0))
    return df

def plot_time_series(df, country_name, timestamp_column='Timestamp', columns=['GHI', 'DNI', 'DHI']):
    st.write(f"Time Series Plot for {country_name}")
    fig, ax = plt.subplots(figsize=(12, 6))
    for column in columns:
        ax.plot(df[timestamp_column], df[column], label=column)
    ax.set_xlabel('Time')
    ax.set_ylabel('Solar Radiation (W/m²)')
    ax.set_title(f'Solar Radiation Over Time for {country_name}')
    ax.legend()
    st.pyplot(fig)

def plot_histograms(df, country_name, columns, bins=15, figsize=(12, 8)):
    num_columns = len(columns)
    num_rows = (num_columns + 2) // 3
    fig, axes = plt.subplots(num_rows, 3, figsize=figsize)
    axes = axes.flatten()
    for i, column in enumerate(columns):
        axes[i].hist(df[column].dropna(), bins=bins, edgecolor='black')
        axes[i].set_title(column)
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
    fig.suptitle(f'Histogram of Solar Radiation Metrics for {country_name}')
    st.pyplot(fig)

def plot_correlation_heatmap(df, country_name, columns=None):
    if columns is None:
        columns = df.columns.tolist()
    correlation_matrix = df[columns].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    ax.set_title(f'Correlation Heatmap for {country_name}')
    st.pyplot(fig)

def plot_boxplot_by_country(df, country_column='Country', value_column='GHI'):
    if country_column not in df.columns or value_column not in df.columns:
        st.warning(f"Columns '{country_column}' or '{value_column}' not found in the DataFrame.")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x=country_column, y=value_column, data=df)
    ax.set_title(f"{value_column} Distribution by Country")
    ax.set_xlabel('Country')
    ax.set_ylabel(value_column)
    st.pyplot(fig)

def compare_mods(df, country_name):
    # Check if 'ModA' and 'ModB' columns are present in the DataFrame
    if 'ModA' not in df.columns or 'ModB' not in df.columns:
        st.warning(f"Columns 'ModA' and 'ModB' not found in the DataFrame for {country_name}.")
        return
    
    # Scatter plot
    st.write(f"ModA vs. ModB in {country_name}")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['ModA'], df['ModB'])
    ax.set_xlabel('ModA (W/m²)')
    ax.set_ylabel('ModB (W/m²)')
    ax.set_title(f'ModA vs. ModB in {country_name}')
    st.pyplot(fig)
    
    # Calculate correlation
    mod_correlation = df[['ModA', 'ModB']].corr()
    st.write(f"Correlation between ModA and ModB for {country_name}:")
    st.write(mod_correlation)

def correlation_with_ghi(df, variables, country_name):
    # Check if 'GHI' is in the DataFrame
    if 'GHI' not in df.columns:
        st.warning(f"'GHI' column not found in the DataFrame for {country_name}.")
        return
    
    # Add 'GHI' to the list of variables to ensure it is included in the correlation calculation
    variables_with_ghi = ['GHI'] + variables
    
    # Check if all specified variables are in the DataFrame
    missing_variables = [var for var in variables if var not in df.columns]
    if missing_variables:
        st.warning(f"Columns {missing_variables} not found in the DataFrame for {country_name}.")
        return
    
    # Calculate the correlation matrix and extract the correlation with 'GHI'
    correlation_matrix = df[variables_with_ghi].corr()
    correlation_with_ghi = correlation_matrix['GHI']
    
    # Display the correlation results
    st.write(f"Correlation between GHI and other variables for {country_name}:")
    st.write(correlation_with_ghi)

def plot_monthly_average_ghi(df, country_name):
    # Check if required columns are in the DataFrame
    if not all(col in df.columns for col in ['Year', 'Month', 'GHI']):
        st.warning(f"Required columns 'Year', 'Month', and 'GHI' not found in the DataFrame for {country_name}.")
        return
    
    # Group by year and month, then calculate the mean GHI
    monthly_ghi = df.groupby(['Year', 'Month'])['GHI'].mean().unstack()
    
    # Plotting
    st.write(f"Monthly Average GHI for {country_name}")
    fig, ax = plt.subplots(figsize=(12, 6))
    monthly_ghi.plot(ax=ax)
    ax.set_title(f'Monthly Average GHI for {country_name}')
    ax.set_xlabel('Year')
    ax.set_ylabel('Average GHI (W/m²)')
    ax.legend(title='Month')
    st.pyplot(fig)

def plot_diurnal_pattern_ghi(df, country_name):
    # Check if required columns are in the DataFrame
    if not all(col in df.columns for col in ['Hour', 'GHI']):
        st.warning(f"Required columns 'Hour' and 'GHI' not found in the DataFrame for {country_name}.")
        return
    
    # Group by hour, then calculate the mean GHI
    hourly_ghi = df.groupby('Hour')['GHI'].mean()
    
    # Plotting
    st.write(f"Diurnal Pattern of GHI for {country_name}")
    fig, ax = plt.subplots(figsize=(12, 6))
    hourly_ghi.plot(ax=ax)
    ax.set_title(f'Average GHI by Hour of the Day for {country_name}')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average GHI (W/m²)')
    st.pyplot(fig)

def process_and_recommend_investment(benin_data, sierraleone_data, togo_data):
    # Concatenate all data into a single DataFrame
    combined_data = pd.concat([benin_data, sierraleone_data, togo_data], ignore_index=True)
    
    # Calculate average metrics by country
    average_metrics = combined_data.groupby('Country')[['GHI', 'DNI', 'DHI']].mean()
    
    # Sort by GHI in descending order and add rank
    average_metrics['Rank'] = average_metrics['GHI'].rank(ascending=False)
    high_potential = average_metrics.sort_values(by='GHI', ascending=False)
    
    # Print high potential countries with their ranks
    st.subheader("High Potential Countries based on GHI")
    st.write(high_potential)
    
    # Recommendations based on GHI
    top_countries = high_potential.head(3)  # Top 3 countries with highest average GHI
    st.subheader("Recommended Countries for Solar Investment")
    recommendations = []
    for rank, (country, row) in enumerate(top_countries.iterrows(), start=1):
        recommendations.append(f"Rank {rank}: {country} - Average GHI: {row['GHI']:.2f} W/m²")
    st.write("\n".join(recommendations))

# Streamlit App layout
st.title('Solar Radiation Data Analysis')

st.sidebar.header("Choose Dataset")
dataset_option = st.sidebar.selectbox(
    'Select the dataset',
    ('Benin Malanville', 'Togo Dapaong', 'Sierra Leone Bumbuna')
)

if dataset_option == 'Benin Malanville':
    data = benin_data
    country_name = 'Benin Malanville'
elif dataset_option == 'Togo Dapaong':
    data = togo_data
    country_name = 'Togo Dapaong'
else:
    data = sierraleone_data
    country_name = 'Sierra Leone Bumbuna'

st.write(f"Data Analysis for {country_name}")

# Data Preprocessing
data = convert_to_datetime(data)
data = extract_datetime_components(data)
data = replace_negative_values(data, ['GHI', 'DNI', 'DHI'])

# Display Missing Values
st.subheader("Missing Values Report")
st.write(check_null_values(data))

# Summary Statistics
st.subheader("Summary Statistics")
st.write(data[['GHI', 'DNI', 'DHI']].describe())

# Plot Time Series
st.subheader("Time Series Plot")
plot_time_series(data, country_name)

# Plot Histograms
st.subheader("Histograms")
plot_histograms(data, country_name, ['GHI', 'DNI', 'DHI', 'Tamb', 'RH'])

# Correlation Heatmap
st.subheader("Correlation Heatmap")
plot_correlation_heatmap(data, country_name, columns=['GHI', 'DNI', 'DHI', 'Tamb', 'RH', 'WS', 'WSgust', 'WSstdev', 'WD', 'WDstdev', 'BP', 'Cleaning', 'Precipitation', 'TModA', 'TModB'])

# Plot Comparison of ModA and ModB
st.subheader("Comparison of ModA and ModB")
compare_mods(data, country_name)

# Correlation with GHI
st.subheader("Correlation with GHI")
correlation_with_ghi(data, ['DNI', 'DHI', 'Tamb', 'RH', 'WS', 'WSgust', 'WSstdev', 'WD', 'WDstdev', 'BP', 'Cleaning', 'Precipitation', 'TModA', 'TModB'], country_name)

# Monthly Average GHI
st.subheader("Monthly Average GHI")
plot_monthly_average_ghi(data, country_name)

# Diurnal Pattern of GHI
st.subheader("Diurnal Pattern of GHI")
plot_diurnal_pattern_ghi(data, country_name)

# Investment Recommendations
st.subheader("Investment Recommendations")
process_and_recommend_investment(benin_data, sierraleone_data, togo_data)
