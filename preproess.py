import pandas as pd
def calculate_by_indicator(file_path, output_file="iiag_indicator_summary.csv"):
    """
    Calculate the median and mean for each indicator by year.
    Parameters:
    - file_path: Path to the input file
    - output_file: Path to the output file
    Returns:
    - summary_data: DataFrame containing the median and mean values
    """
    # Read the data
    data = pd.read_csv(file_path)

    # Extract the year columns
    year_columns = data.columns[8:]
    indicator_column = "Indicator"

    # Create an empty DataFrame to store the results
    results = []

    # Iterate through each indicator
    for indicator in data[indicator_column].unique():
        indicator_data = data[data[indicator_column] == indicator]
        for year in year_columns:
            median = indicator_data[year].median()
            mean = indicator_data[year].mean()
            results.append({"Indicator": indicator, "Year": year, "Median": median, "Mean": mean})

    # Convert results to a DataFrame
    summary_data = pd.DataFrame(results)

    # Save the results to a CSV file
    summary_data.to_csv(output_file, index=False)

    return summary_data


def transform_iiag(file_path, output_file="iiag_transformed.csv"):
    """
    Transform the IIAG file into the required format.
    Parameters:
    - file_path: Path to the input file
    - output_file: Path to the output file
    Returns:
    - transformed_data: Transformed DataFrame
    """
    # Read the IIAG data
    data = pd.read_csv(file_path)
    columns_to_drop = ["Economy ISO3", "Indicator ID", "Attribute 1", "Attribute 2", "Attribute 3", "Partner"]
    data = data.drop(columns=columns_to_drop, errors="ignore")

    # Filter data for Zambia and Botswana
    selected_countries = ["Zambia", "Botswana"]
    filtered_data = data[data["Economy Name"].isin(selected_countries)]

    # Filter data for selected indicators
    selected_indicators = ["Security & Safety  (IIAG)", "Rule of Law & Justice (IIAG)"]
    filtered_data = filtered_data[filtered_data["Indicator"].isin(selected_indicators)]

    # Transform the wide format to long format
    transformed_data = filtered_data.melt(
        id_vars=["Economy Name", "Indicator"],  # Fixed columns
        var_name="Year",  # Generated year column
        value_name="Value"  # Generated value column
    )

    # Rename columns
    transformed_data.rename(columns={"Economy Name": "Country Name"}, inplace=True)

    # Save the results
    transformed_data.to_csv(output_file, index=False)

    return transformed_data



def process_hdi_data(file_path, country_name):
    """
    Process HDI data, filter for a specific year range, add a country name column, and save the output with the country name as the filename.

    Parameters:
    - file_path: Path to the original HDI data file.
    - country_name: Name of the country to be added as a column in the result.

    Returns:
    - filtered_data: Processed DataFrame.
    """
    # Load the data
    hdi_data = pd.read_csv(file_path)

    # Filter data for the years 2012 to 2021 and create a copy
    filtered_data = hdi_data.iloc[27:37].copy()  # Rows 28-37 (Python indexing starts at 0)

    # Extract the year
    filtered_data["year"] = filtered_data["key"].str.extract(r'(\d{4})')  # Extract the year
    filtered_data["year"] = filtered_data["year"].astype(int)  # Convert to integer

    # Add the country name column
    filtered_data["Country Name"] = country_name

    # Rearrange column order
    filtered_data = filtered_data[["Country Name", "year", "value"]]

    # Automatically generate the output filename
    output_file = f"{country_name}_HDI.csv"

    # Save the result
    filtered_data.to_csv(output_file, index=False)

    return filtered_data

def merge_hdi_gpi(hdi_file1: object, hdi_file2: object, gpi_file1: object, gpi_file2: object, country1: object, country2: object, output_file="merged_hdi_gpi.csv") -> object:
    """
    Merge HDI and GPI data for two countries.

    Parameters:
    - hdi_file1: HDI file path for Country 1.
    - hdi_file2: HDI file path for Country 2.
    - gpi_file1: GPI file path for Country 1.
    - gpi_file2: GPI file path for Country 2.
    - country1: Name of Country 1.
    - country2: Name of Country 2.

    Returns:
    - merged_data: Merged DataFrame containing HDI and GPI data for both countries.
    """
    # Load data
    hdi1 = pd.read_csv(hdi_file1)
    hdi2 = pd.read_csv(hdi_file2)
    gpi1 = pd.read_csv(gpi_file1)
    gpi2 = pd.read_csv(gpi_file2)

    # Add country names
    hdi1["country"] = country1
    hdi2["country"] = country2
    gpi1["country"] = country1
    gpi2["country"] = country2

    # Concatenate HDI and GPI data
    hdi = pd.concat([hdi1, hdi2], ignore_index=True)
    gpi = pd.concat([gpi1, gpi2], ignore_index=True)

    # Merge HDI and GPI data by country and year
    merged_data = pd.merge(hdi, gpi, on=["year", "Country Name"], suffixes=("_hdi", "_gpi"))
    merged_data = merged_data.drop(columns=["country_hdi", "country_gpi"])
    merged_data.to_csv(output_file, index=False)

    return merged_data




indicator_summary = calculate_by_indicator("iiag.csv", output_file="iiag_indicator_summary.csv")
iiag_file = "iiag.csv"
transformed_data = transform_iiag(iiag_file)
processed_data = process_hdi_data(
    file_path="Zambia.csv",
    country_name="Zambia"
)

merged_data = merge_hdi_gpi(
    hdi_file1="Zambia_HDI.csv",
    hdi_file2="Botswana_HDI.csv",
    gpi_file1="Zambia_GPI.csv",
    gpi_file2="Botswana_GPI.csv",
    country1="Zambia",
    country2="Botswana",
    output_file="hdi_gpi_combined.csv"
)

