import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
def clean_and_merge_data(GDP_file, population_file, output_file="cleaned_and_merge_data.csv"):
    """
    Clean and merge GDP and population data, and calculate GDP per capita.
    Parameters:
    - gdp_file: Path to the GDP data file
    - population_file: Path to the population data file
    - output_file: Path to save the processed data
    Returns:
    - cleaned_data: Processed DataFrame
    """
    # Read data
    gdp_data = pd.read_csv(GDP_file)
    population_data = pd.read_csv(population_file)

    # Rename columns
    gdp_data.rename(columns={'Value': 'GDP'}, inplace=True)
    population_data.rename(columns={'Value': 'Population'}, inplace=True)

    # Merge data
    merged_data = pd.merge(gdp_data, population_data, on=['Country Name', 'Time'])

    # Calculate GDP per capita
    merged_data['GDP_per_capita'] = merged_data['GDP'] / merged_data['Population']

    # Keep only the required columns
    cleaned_data = merged_data[['Country Name', 'Time', 'GDP', 'Population', 'GDP_per_capita']]

    # Save the data
    cleaned_data.to_csv(output_file, index=False)
    return cleaned_data


def calculate_gdp_growth(data, output_file="processed_data.csv"):
    """
    Calculate GDP growth rate and save the results.
    Parameters:
    - data: Input DataFrame
    - output_file: Path to save the results
    Returns:
    - data: DataFrame with GDP growth rate
    """
    # Calculate GDP growth rate
    data['GDP_growth_rate'] = data.groupby('Country Name')['GDP'].pct_change(fill_method=None) * 100

    # Save the data
    data.to_csv(output_file, index=False)
    return data



def plot_gdp_trend(data):

    # Subnational data
    countries = data['Country Name'].unique()
    plt.figure(figsize=(10, 6))
    for country in countries:
        country_data = data[data['Country Name'] == country]
        plt.plot(country_data['Time'], country_data['GDP'], label=f'{country} GDP')
    plt.xlabel('Year')
    plt.ylabel('GDP (constant LCU)')
    plt.title('GDP Trend')
    plt.legend()
    plt.show()


def plot_gdp_per_capita(data):
    """
    Draw a bar chart of per capita GDP comparison.
    """
    countries = data['Country Name'].unique()
    plt.figure(figsize=(10, 6))
    for country in countries:
        country_data = data[data['Country Name'] == country]
        plt.bar(country_data['Time'], country_data['GDP_per_capita'], label=f'{country} GDP per Capita', alpha=0.7)
    plt.xlabel('Year')
    plt.ylabel('GDP per Capita')
    plt.title('GDP per Capita Comparison')
    plt.legend()
    plt.show()


def plot_gdp_growth_rate(data):
    """
    Plot the trend of GDP growth rate.
    Parameters:
    - data: indicates the data DataFrame
    """
    countries = data['Country Name'].unique()
    plt.figure(figsize=(10, 6))
    for country in countries:
        country_data = data[data['Country Name'] == country]
        plt.plot(country_data['Time'], country_data['GDP_growth_rate'], label=f'{country} GDP Growth Rate')
    plt.xlabel('Year')
    plt.ylabel('GDP Growth Rate (%)')
    plt.title('GDP Growth Rate Trend')
    plt.legend()
    plt.show()


def calculate_cagr(data):
    """
    Calculate the compound annual growth rate (CAGR) of GDP for each country.
    Parameters:
    - data: DataFrame, containing 'Country Name', 'Time', and 'GDP'.
    Back:
    - cagr_df: Contains the DataFrame for the country and corresponding CAGR
    """
    cagr_results = []
    countries = data['Country Name'].unique()

    for country in countries:
        country_data = data[data['Country Name'] == country]
        starting_gdp = country_data.iloc[0]['GDP']
        ending_gdp = country_data.iloc[-1]['GDP']
        num_years = country_data['Time'].nunique() - 1

        cagr = ((ending_gdp / starting_gdp) ** (1 / num_years) - 1) * 100
        cagr_results.append({'Country Name': country, 'CAGR': cagr})

    cagr_df = pd.DataFrame(cagr_results)
    return cagr_df

def plot_cagr(cagr_data):
    """
    CAGR comparisons between countries are plotted.
    Parameters:
    -cagr_data: DataFrame containing 'Country Name' and 'CAGR'
    """
    plt.figure(figsize=(8, 5))
    plt.bar(cagr_data['Country Name'], cagr_data['CAGR'], color=['blue', 'green'], alpha=0.7)
    plt.xlabel('Country')
    plt.ylabel('CAGR (%)')
    plt.title('Compound Annual Growth Rate (CAGR) Comparison')
    plt.show()


def merge_gdp_iiag(gdp_file, iiag_file, output_file="merged_data.csv"):
    """
    Merge GDP data and IIAG data, and filter data for the years 2012-2021.
    Parameters:
    - gdp_file: Path to the GDP data file
    - iiag_file: Path to the IIAG data file
    - output_file: Path to save the merged data
    Returns:
    - merged_data: Merged DataFrame
    """
    # Read GDP data and IIAG data
    gdp_data = pd.read_csv(gdp_file)
    iiag_data = pd.read_csv(iiag_file)

    # Rename the year column in GDP data
    gdp_data.rename(columns={"Time": "Year"}, inplace=True)

    # Convert the year column to integer type
    gdp_data["Year"] = gdp_data["Year"].astype(int)
    iiag_data["Year"] = iiag_data["Year"].astype(int)

    # Filter data for the years 2012-2021
    gdp_data = gdp_data[(gdp_data["Year"] >= 2012) & (gdp_data["Year"] <= 2021)]
    iiag_data = iiag_data[(iiag_data["Year"] >= 2012) & (iiag_data["Year"] <= 2021)]

    # Merge data
    merged_data = pd.merge(
        gdp_data,
        iiag_data,
        on=["Country Name", "Year"],
        how="inner"
    )

    # Select required columns
    merged_data = merged_data[[
        "Country Name", "Year", "Population", "GDP", "GDP_per_capita", "GDP_growth_rate", "Indicator", "Value"
    ]]

    # Save the merged result
    merged_data.to_csv(output_file, index=False)

    return merged_data



def plot_scaled_lines_by_year(data, indicator_name, economic_metric, output_file=None, scale_method="normalize"):
    """
    Plot line charts with standardized or log-transformed data.
    Parameters:
    - data: The merged DataFrame
    - indicator_name: Specified IIAG indicator name (e.g., 'Rule of Law & Justice (IIAG)')
    - economic_metric: Specified economic metric (e.g., 'GDP_per_capita')
    - output_file: Path to save the chart (optional)
    - scale_method: Scaling method, either "normalize" or "log"
    """
    # Filter data for the specified indicator
    indicator_data = data[data["Indicator"] == indicator_name].copy()

    # Scale up the IIAG values
    indicator_data["Scaled_Value"] = indicator_data["Value"] * 100

    # Scaling method
    if scale_method == "normalize":
        # Normalize to [0, 1]
        for col in [economic_metric, "Scaled_Value"]:
            indicator_data[col] = (indicator_data[col] - indicator_data[col].min()) / (
                indicator_data[col].max() - indicator_data[col].min()
            )
    elif scale_method == "log":
        # Log transformation
        for col in [economic_metric, "Scaled_Value"]:
            indicator_data[col] = np.log1p(indicator_data[col])

    # Get unique years and countries
    years = sorted(indicator_data["Year"].unique())
    countries = indicator_data["Country Name"].unique()

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot line charts
    for country in countries:
        # Economic metric line
        country_data = indicator_data[indicator_data["Country Name"] == country]
        ax.plot(
            country_data["Year"],
            country_data[economic_metric],
            marker='o',
            label=f"{country} {economic_metric} (Scaled)",
            alpha=0.8
        )

        # Scaled IIAG value line
        ax.plot(
            country_data["Year"],
            country_data["Scaled_Value"],
            marker='x',
            linestyle='--',
            label=f"{country} {indicator_name} (Scaled)",
            alpha=0.8
        )

    # Set X-axis and labels
    ax.set_xticks(years)
    ax.set_xlabel("Year")
    ax.set_ylabel("Scaled Value")
    ax.set_title(f"{indicator_name} and {economic_metric} by Year (Scaled)")
    ax.legend()

    # Show grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    # Save or display the chart
    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()

def compare_iiag_values(summary_file, country_file, output_file="iiag_comparison.csv"):
    """
    Compare a country's IIAG indicator values with the African average and median.

    Parameters:
    - summary_file: Path to the file containing the African average and median for each indicator by year.
    - country_file: Path to the file containing a country's specific values for each indicator by year.
    - output_file: Path to save the output file. Default is "iiag_comparison.csv".

    Returns:
    - result: DataFrame with merged and calculated data.
    """
    # Read data
    summary_data = pd.read_csv(summary_file)
    country_data = pd.read_csv(country_file)

    # Merge data
    merged_data = pd.merge(
        country_data,
        summary_data,
        on=["Indicator", "Year"],
        how="inner"
    )

    # Calculate differences
    merged_data["Value Compared with Mean"] = merged_data["Value"] - merged_data["Mean"]
    merged_data["Value Compared with Median"] = merged_data["Value"] - merged_data["Median"]
    merged_data["Min(Value Compared)"] = merged_data[[
        "Value Compared with Mean", "Value Compared with Median"
    ]].min(axis=1)

    # Select required columns
    result = merged_data[[
        "Country Name", "Indicator", "Year", "Mean", "Median", "Value", "Min(Value Compared)"
    ]].copy()
    result.rename(columns={
        "Mean": "Africa Mean",
        "Median": "Africa Median",
        "Value": "Country Value"
    }, inplace=True)

    # Save the result
    result.to_csv(output_file, index=False)
    return result

def plot_two_indicators(data, indicator1, indicator2, output_file=None):
    """
    Plot a single figure containing two subplots, each showing the Min(Value Compared) for two indicators.

    Parameters:
    - data: DataFrame containing country, indicators, years, and their differences from the African average.
    - indicator1: Name of the first indicator (e.g., 'Rule of Law & Justice (IIAG)').
    - indicator2: Name of the second indicator (e.g., 'Security & Safety (IIAG)').
    - output_file: Path to save the figure (optional).
    """
    # Filter data for the two indicators
    data1 = data[data["Indicator"] == indicator1]
    data2 = data[data["Indicator"] == indicator2]

    # Get unique years and countries
    years = sorted(data["Year"].unique())
    countries = data["Country Name"].unique()

    # Set bar width and X-axis positions
    bar_width = 0.4
    x_indices = np.arange(len(years))

    # Create the figure and subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot the first indicator
    for i, country in enumerate(countries):
        country_data = data1[data1["Country Name"] == country]
        axes[0].bar(
            x_indices + (i - 0.5) * bar_width,
            country_data["Min(Value Compared)"],
            bar_width,
            label=country
        )
    axes[0].set_title(f"{indicator1} - Min(Value Compared)")
    axes[0].set_ylabel("Min(Value Compared)")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)

    # Plot the second indicator
    for i, country in enumerate(countries):
        country_data = data2[data2["Country Name"] == country]
        axes[1].bar(
            x_indices + (i - 0.5) * bar_width,
            country_data["Min(Value Compared)"],
            bar_width,
            label=country
        )
    axes[1].set_title(f"{indicator2} - Min(Value Compared)")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Min(Value Compared)")
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)

    # Set the X-axis
    axes[1].set_xticks(x_indices)
    axes[1].set_xticklabels(years, rotation=45)

    # Adjust layout
    plt.tight_layout()

    # Save or display the chart
    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()


def analyze_hdi_gpi(data, country1, country2):
    """
    Analyze the relationship between HDI and GPI data for two countries and plot trends.

    Parameters:
    - data: Merged DataFrame containing columns ["Country Name", "year", "value_hdi", "value_gpi"].
    - country1: Name of the first country.
    - country2: Name of the second country.

    Returns:
    - No return value, but generates trend plots and prints correlation coefficients.
    """
    # Standardize the data
    scaler = MinMaxScaler()
    data[["value_hdi", "value_gpi"]] = scaler.fit_transform(data[["value_hdi", "value_gpi"]])

    # Retrieve data for the two countries
    country1_data = data[data["Country Name"] == country1]
    country2_data = data[data["Country Name"] == country2]

    # Calculate correlation
    corr1, _ = pearsonr(country1_data["value_hdi"], country1_data["value_gpi"])
    corr2, _ = pearsonr(country2_data["value_hdi"], country2_data["value_gpi"])
    print(f"{country1} HDI and GPI Correlation Coefficient: {corr1:.2f}")
    print(f"{country2} HDI and GPI Correlation Coefficient: {corr2:.2f}")

    # Plot trends
    plt.figure(figsize=(14, 8))

    # Country 1
    plt.subplot(2, 1, 1)
    plt.plot(country1_data["year"], country1_data["value_hdi"], label=f"{country1} HDI", marker="o")
    plt.plot(country1_data["year"], country1_data["value_gpi"], label=f"{country1} GPI", marker="o")
    plt.title(f"{country1}: HDI vs GPI (Standardized)")
    plt.ylabel("Standardized Value")
    plt.legend()
    plt.grid()

    # Country 2
    plt.subplot(2, 1, 2)
    plt.plot(country2_data["year"], country2_data["value_hdi"], label=f"{country2} HDI", marker="o")
    plt.plot(country2_data["year"], country2_data["value_gpi"], label=f"{country2} GPI", marker="o")
    plt.title(f"{country2}: HDI vs GPI (Standardized)")
    plt.xlabel("Year")
    plt.ylabel("Standardized Value")
    plt.legend()
    plt.grid()

    # Display the plot
    plt.tight_layout()
    plt.show()




cleaned_data = clean_and_merge_data("GDP.csv", "population.csv")
processed_data = calculate_gdp_growth(cleaned_data)
cagr_data = calculate_cagr(processed_data)
gdp_file = "processed_data.csv"
iiag_file = "iiag_transformed.csv"
merged_data = merge_gdp_iiag(gdp_file, iiag_file)
plot_gdp_trend(processed_data)
plot_gdp_per_capita(processed_data)
plot_gdp_growth_rate(processed_data)
plot_cagr(cagr_data)


summary_file = "iiag_indicator_summary.csv"
country_file = "iiag_transformed.csv"
output_file = "iiag_comparison.csv"

result = compare_iiag_values(summary_file, country_file, output_file)
plot_two_indicators(
    data=result,
    indicator1="Rule of Law & Justice (IIAG)",
    indicator2="Security & Safety  (IIAG)"
)

plot_scaled_lines_by_year(
    merged_data,
    indicator_name="Rule of Law & Justice (IIAG)",
    economic_metric="GDP",
    scale_method="normalize"
)


plot_scaled_lines_by_year(
    merged_data,
    indicator_name="Security & Safety  (IIAG)",
    economic_metric="GDP",
    scale_method="normalize"
)

data = pd.read_csv("hdi_gpi_combined.csv")
analyze_hdi_gpi(data=data, country1="Zambia", country2="Botswana")