import streamlit as st
import pandas as pd
import pickle

# Load the trained model, scaler, and expected column order
with open('gb_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)
# Load test data to build a comparison distribution
X_test = pd.read_csv('X_test.csv')
all_probabilities = model.predict_proba(X_test)[:, 1]
st.title("Bank Customer Churn Risk Calculator")
st.write("Enter customer details to predict churn risk.")

st.header("Customer Details")

credit_score = st.slider("Credit Score", 300, 900, 650)
age = st.slider("Age", 18, 92, 40)
tenure = st.slider("Tenure (years with bank)", 0, 10, 5)
balance = st.number_input("Account Balance", min_value=0.0, value=50000.0)
num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
salary = st.number_input("Estimated Salary", min_value=0.0, value=60000.0)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])

if st.button("Calculate Churn Risk"):

    # Build the engineered features, same as in our notebooks
    balance_salary_ratio = balance / salary
    high_product_density = 1 if num_products >= 3 else 0
    is_active_num = 1 if is_active == "Yes" else 0
    engagement_product_interaction = is_active_num * num_products
    age_tenure_interaction = age * tenure

    # Build a dictionary matching our training columns exactly
    input_dict = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': 1 if has_cr_card == "Yes" else 0,
        'IsActiveMember': is_active_num,
        'EstimatedSalary': salary,
        'BalanceSalaryRatio': balance_salary_ratio,
        'HighProductDensity': high_product_density,
        'EngagementProductInteraction': engagement_product_interaction,
        'AgeTenureInteraction': age_tenure_interaction,
        'Geography_Germany': 1 if geography == "Germany" else 0,
        'Geography_Spain': 1 if geography == "Spain" else 0,
        'Gender_Male': 1 if gender == "Male" else 0
    }

    # Convert to a DataFrame with the exact column order the model expects
    input_df = pd.DataFrame([input_dict])[feature_columns]

    # Scale the same numeric columns we scaled during training
    numeric_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                     'EstimatedSalary', 'BalanceSalaryRatio', 'EngagementProductInteraction',
                     'AgeTenureInteraction']
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Predict
    probability = model.predict_proba(input_df)[0][1]
    st.session_state['last_probability'] = probability
    prediction = model.predict(input_df)[0]

    st.header("Result")
    st.write(f"Churn Probability: **{probability:.1%}**")

    if prediction == 1:
        st.error("High Risk: This customer is likely to churn.")
    else:
        st.header("What-If Scenario Simulator")
    st.write("Adjust engagement or products below to see how risk would change:")

    whatif_active = st.selectbox("What if Active Member?", ["Yes", "No"], key="whatif_active")
    whatif_products = st.selectbox("What if Number of Products?", [1, 2, 3, 4], key="whatif_products")

    whatif_active_num = 1 if whatif_active == "Yes" else 0
    whatif_engagement = whatif_active_num * whatif_products

    whatif_dict = input_dict.copy()
    whatif_dict['IsActiveMember'] = whatif_active_num
    whatif_dict['NumOfProducts'] = whatif_products
    whatif_dict['HighProductDensity'] = 1 if whatif_products >= 3 else 0
    whatif_dict['EngagementProductInteraction'] = whatif_engagement

    whatif_df = pd.DataFrame([whatif_dict])[feature_columns]
    whatif_df[numeric_cols] = scaler.transform(whatif_df[numeric_cols])

    whatif_probability = model.predict_proba(whatif_df)[0][1]

    st.write(f"Original Churn Probability: **{probability:.1%}**")
    st.write(f"What-If Churn Probability: **{whatif_probability:.1%}**")             
    st.write(f"Change: **{(whatif_probability - probability)*100:+.1f} percentage points**")
    st.success("Low Risk: This customer is likely to stay.")
    st.header("What Drives Churn? (Feature Importance)")

importances = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)
st.bar_chart(importances)
st.header("How Does This Compare to Other Customers?")

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.hist(all_probabilities, bins=30, color='skyblue', edgecolor='black')
if st.session_state.get('last_probability') is not None:
    ax.axvline(st.session_state['last_probability'], color='red', linestyle='--', linewidth=2, label='This Customer')
    ax.legend()
ax.set_xlabel('Churn Probability')
ax.set_ylabel('Number of Customers')
st.pyplot(fig)