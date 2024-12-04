Project Title: The Impact of War and Peace Environments on Economic Development
Project Overview
This project analyzes the data of Zambia and Botswana to study the relationship between peaceful environments (measured by GPI - Global Peace Index) and economic development (measured by GDP, GDP per capita, and HDI - Human Development Index). By comparing the economic performances of these two countries under different peace conditions, the project explores the significant impact of peace on economic growth.

Project Structure
The project is divided into the following modules and files:

1. Main Code Files
main.py
Contains the core analysis and visualization functions, including:

Analyzing the relationship between HDI and GPI.
Plotting GDP and IIAG indicators' trends.
Calculating correlation coefficients and generating line charts.
preproess.py
Contains data cleaning and preprocessing modules, including:

Cleaning GDP and population data.
Merging HDI, GPI, and IIAG indicator data.
Generating standardized CSV files for analysis.

2. Data Folder
Contains the raw data and intermediate CSV files used in the project:
GDP.csv and Population.csv: GDP and population data.
iiag.csv: IIAG indicator data.
Zambia_HDI.csv and Botswana_HDI.csv: HDI data.
Zambia_GPI.csv and Botswana_GPI.csv: GPI data.
Installation and Execution
Environment Requirements
Python Version: Python 3.8 or higher is recommended.
Dependencies:
pandas
matplotlib
numpy
scipy
scikit-learn
Installation Steps
Clone the Project

git clone https://github.com/your-repository
cd your-repository

Install Dependencies
pip install -r requirements.txt
Execution Steps

Data Cleaning and Merging:
Run preproess.py to clean and merge the raw data.
python preproess.py

Data Analysis and Visualization:
Use main.py to perform analysis and generate visualizations.
python main.py

Features
Data Preprocessing
Cleaning GDP and population data and calculating GDP per capita.
Merging HDI and GPI data to create a standardized analysis table.
Organizing IIAG indicators and extracting related economic metrics for analysis.
Data Analysis
Relationship between GDP Growth and IIAG Indicators

Use regression analysis to study how changes in IIAG indicator values impact GDP and GDP per capita.
Correlation Between HDI and GPI

Calculate the correlation coefficient between HDI and GPI for both countries and visualize their relationship using trend charts.
Comparison of Economic Indicators

Compare the differences between the two countries in GDP per capita and peace index, highlighting the impact of peace on economic development.
Visualization
Trend Charts: Show trends of HDI, GPI, and GDP over time.
Bar Charts: Compare annual changes in GDP and IIAG indicator values.
Correlation Line Charts: Display the correlation between various indicators.

Notes
Ensure all raw data files are in the correct paths specified in the code.
If additional sample code execution or parameter adjustments are needed, refer to comments in main.py or preproess.py.
To avoid execution errors, check that all dependencies are installed, and file formats match the expected structure.
