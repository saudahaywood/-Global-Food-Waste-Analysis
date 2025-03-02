
# **Global Food Waste Analysis**

## **Project Overview**
This project aims to analyze **global food waste trends**, identify major causes, and predict food loss percentages using machine learning models. The analysis is based on a dataset that details **food loss percentages, supply chain stages, causes of loss, and treatments applied**. The primary goal is to build a predictive model to estimate **food loss percentages** and identify strategies to reduce waste.

## **Dataset**
- **Source:** FAO's Food Loss and Waste Platform
- **Features:**
  - **m49_code:** Numerical country/region code.
  - **country:** Name of the country.
  - **region:** Geographical region.
  - **cpc_code:** Commodity classification code.
  - **commodity:** Type of food commodity (e.g., rice, meat, fruits).
  - **year:** Year of data collection.
  - **loss_percentage:** Percentage of food lost.
  - **loss_quantity:** Quantity of food lost (e.g., in kilograms).
  - **activity:** Related activities (e.g., storage, packaging, retail).
  - **food_supply_stage:** Supply chain stage (e.g., storage, retail).
  - **treatment:** Storage interventions applied.
  - **cause_of_loss:** Factors causing food loss (e.g., rodents, poor storage).
  - **sample_size:** Sample size for data collection.
  - **method_data_collection:** Data collection methodology.
  - **reference & URL:** Data source references.

The **target variable** for the predictive model is **`loss_percentage`**.

## **Methodology**
### **1. Data Exploration**
- **Loaded dataset and checked missing values.**
- **Analyzed key trends** in food loss by **country, commodity, and supply chain stage.**
- **Identified leading causes of food waste** and **treatments used to reduce waste**.

#### **1.1 Missing Data Handling**
- **Columns with excessive missing values were dropped.**
- **Categorical variables were one-hot encoded.**
- **Missing `cause_of_loss` values were replaced with `'Unknown'`.**

#### **1.2 Key Insights from Data Analysis**
- **Most affected food categories:** Snails, grapefruit juice, and pork had the highest food loss percentages.
- **Top causes of food waste:** Marine shipment losses, fruit flies, and import rejections were among the leading causes.
- **Most vulnerable supply chain stage:** Post-harvest and household-level losses were significantly high.
- **Trends over time:** Food loss percentages fluctuated based on external factors like **storage conditions, handling, and market demand**.

---

### **2. Model Building**
#### **2.1 Data Preprocessing**
```python
# Drop unnecessary columns with excessive missing values
data = data.drop(columns=['region', 'method_data_collection', 'loss_quantity',
                          'sample_size', 'reference', 'url', 'notes'])

# Replace missing values
data.fillna({'cause_of_loss': 'Unknown'}, inplace=True)

# One-hot encode categorical variables
data_encoded = pd.get_dummies(data, columns=['commodity', 'food_supply_stage', 'cause_of_loss'], drop_first=True)
```

---

### **2.2 Model 1: Linear Regression**
#### **Training and Evaluation**
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Define features (X) and target (y)
X = data_encoded.drop(columns=['loss_percentage'])
y = data_encoded['loss_percentage']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = lr_model.predict(X_test)

# Calculate evaluation metrics
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Linear Regression RMSE: {rmse}')
print(f'Linear Regression MAE: {mae}')
print(f'Linear Regression R²: {r2}')
```
#### **Results**
- **RMSE:** **16,396,338.78** (Extremely high error, indicating poor model performance).
- **R² Score:** **-9,178,495,110,209.982** (Negative, showing the model is not suitable).

---

### **2.3 Model 2: Random Forest Regressor**
```python
from sklearn.ensemble import RandomForestRegressor

# Train the Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf_model.predict(X_test)

# Calculate evaluation metrics
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f'Random Forest RMSE: {rmse}')
print(f'Random Forest R²: {r2}')
```
#### **Results**
- **RMSE:** **2.997** (Much lower error, indicating better performance).
- **R² Score:** **0.693** (Explains 69.3% of variance, significantly better than linear regression).

#### **Conclusion:** **Random Forest is the better model for this dataset.**

---

## **Findings & Recommendations**
### **Key Insights**
- **Food loss is highest in post-harvest storage, households, and exports.**
- **Snails, grapefruit juice, and pork are among the most wasted commodities.**
- **Rodents, improper storage, and transportation issues are major causes of food waste.**
- **Interventions such as proper storage, pest control, and handling improvements can reduce food loss.**

### **Recommendations**
- **Modern storage facilities**: Invest in **temperature-controlled warehouses** to reduce post-harvest losses.
- **Farmer training**: Educate farmers on best post-harvest practices.
- **Pest control measures**: Improve rodent management strategies in storage.
- **Real-time monitoring**: Use IoT and AI-based tracking systems for early spoilage detection.

---

## **Challenges & Limitations**
- **Missing Data**: Many records lacked complete information.
- **Bias in Data**: Some regions were underrepresented.
- **Complexity of Food Waste**: The model does not account for external factors like climate and transportation issues.

---

## **Future Improvements**
1. **Enhancing Data Quality**:
   - Gather **additional real-world data** from more countries.
   - Include **weather, economic conditions, and food pricing factors**.
   
2. **Experimenting with More Models**:
   - **XGBoost** and **Neural Networks** for better predictive accuracy.
   - Time-series forecasting to track trends over multiple years.

3. **Real-Time Monitoring**:
   - **Use IoT sensors** to track spoilage and prevent food loss.
   - **Integrate blockchain** for transparent supply chain tracking.

---

## **Required Libraries**
To run this project, install the necessary Python libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## **References**
- [FAO Food Loss and Waste Database](https://www.fao.org/platform-food-loss-waste/en/)
- [United Nations Food Waste Report](https://www.un.org/en/observances/end-food-waste-day)



