# -Global-Food-Waste-Analysis

## Overview
This project focuses on analyzing global food waste during the storage stage of the food supply chain, using data from the FAO's Food Loss and Waste Platform. The goal is to identify key causes of food loss and propose data-driven strategies to reduce waste, thereby improving food security, economic stability, and environmental sustainability.

## Features
- Analysis of food loss by commodity, activity, and cause of loss.
- Visualizations to highlight trends and patterns in food loss.
- Machine learning models to predict loss percentages and evaluate effective interventions.
- Recommendations for reducing storage-related food loss.

## Data
- **Source:** FAO's Food Loss and Waste Platform ([Link](https://www.fao.org/platform-food-loss-waste/flw-data/en/))
- **Dataset:** Includes variables such as country, region, commodity, year, loss percentage, cause of loss, and treatment methods.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/global-food-waste-analysis.git
   ```
2. Navigate to the project directory:
   ```bash
   cd global-food-waste-analysis
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Ensure the dataset (`data.csv`) is in the project directory.
2. Run the data exploration script:
   ```bash
   python data_exploration.py
   ```
3. Train the machine learning models:
   ```bash
   python model_training.py
   ```
4. Generate visualizations:
   ```bash
   python visualization.py
   ```

## Project Structure
- `data_exploration.py`: Script for cleaning and exploring the dataset.
- `model_training.py`: Scripts for building and evaluating machine learning models (Linear Regression and Random Forest).
- `visualization.py`: Generates bar charts, and trend lines to illustrate key findings.
- `README.md`: Documentation for the project.

## Key Findings
- **Commodity Analysis:** Snails, grapefruit juice, and meat of pig with the bone had the highest loss percentages.
- **Activity Analysis:** Harvesting, packaging, sorting, and storage activities contributed significantly to losses.
- **Cause of Loss:** Rodent infestations were a leading cause, with over 50% of losses in some cases.
- **Modeling Results:** Random Forest outperformed Linear Regression with an RMSE of 2.997 and an R² value of 0.693.

## Recommendations
1. Build modern, pest-resistant storage facilities.
2. Train farmers on effective storage and pest control practices.
3. Implement cost-effective trapping and monitoring systems.
4. Leverage real-time data technologies to track and mitigate food loss.

