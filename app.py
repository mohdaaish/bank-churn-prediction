import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load the trained model, scaler, and expected column order
with open('gb_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

numeric_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                 'EstimatedSalary', 'BalanceSalaryRatio', 'EngagementProductInteraction',
                 'AgeTenureInteraction']

# ---- Load test data (scaled, for model input) ----
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')['Exited']
all_probabilities = model.predict_proba(X_test)[:, 1]
all_predictions = model.predict(X_test)

# ---- Rebuild the UNSCALED version of X_test, so we have real Balance values in euros ----
df_full = pd.read_csv('European_Bank_features.csv')
df_full = df_full.drop(columns=['CustomerId', 'Surname', 'Year'])
df_full = pd.get_dummies(df_full, columns=['Geography', 'Gender'], drop_first=True)

from sklearn.model_selection import train_test_split
X_full = df_full.drop(columns=['Exited'])
y_full = df_full['Exited']
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_full, y_full, test_size=0.2, stratify=y_full, random_state=42
)
real_balance = X_test_raw['Balance'].reset_index(drop=True)

st.title("Bank Customer Churn Risk Calculator")

tab1, tab2 = st.tabs(["Churn Risk Calculator", "Financial Impact Dashboard"])

# =========================================================
# TAB 1: CHURN RISK CALCULATOR (existing functionality)
# =========================================================
with tab1:
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
        balance_salary_ratio = balance / salary
        high_product_density = 1 if num_products >= 3 else 0
        is_active_num = 1 if is_active == "Yes" else 0
        engagement_product_interaction = is_active_num * num_products
        age_tenure_interaction = age * tenure

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

        input_df = pd.DataFrame([input_dict])[feature_columns]
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        probability = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]

        st.session_state['calculated'] = True
        st.session_state['probability'] = probability
        st.session_state['prediction'] = prediction
        st.session_state['input_dict'] = input_dict

    if st.session_state.get('calculated'):

        probability = st.session_state['probability']
        prediction = st.session_state['prediction']
        input_dict = st.session_state['input_dict']

        st.header("Result")
        st.write(f"Churn Probability: **{probability:.1%}**")

        if prediction == 1:
            st.error("High Risk: This customer is likely to churn.")
        else:
            st.success("Low Risk: This customer is likely to stay.")

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

        st.header("What Drives Churn? (Feature Importance)")
        importances = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)
        st.bar_chart(importances)

        st.header("How Does This Compare to Other Customers?")
        fig, ax = plt.subplots()
        ax.hist(all_probabilities, bins=30, color='skyblue', edgecolor='black')
        ax.axvline(probability, color='red', linestyle='--', linewidth=2, label='This Customer')
        ax.legend()
        ax.set_xlabel('Churn Probability')
        ax.set_ylabel('Number of Customers')
        st.pyplot(fig)

# =========================================================
# TAB 2: FINANCIAL IMPACT DASHBOARD (new)
# =========================================================
with tab2:
    st.header("Financial Impact of Predicted Churn")
    st.write(
        "This dashboard estimates the financial exposure represented by customers "
        "the model predicts will churn, based on adjustable business assumptions. "
        "All figures below are estimates driven by the sliders you set — not verified bank data."
    )

    st.subheader("High-Value Customer Definition")
    hv_percentile = st.slider("High-value threshold (percentile of balance)", 50, 95, 75, step=5,
                               help="Customers at or above this percentile of account balance are classified as high-value.")
    high_value_threshold = real_balance.quantile(hv_percentile / 100)
    is_high_value = real_balance >= high_value_threshold

    st.write(f"**High-value threshold:** €{high_value_threshold:,.2f} "
             f"(top {100 - hv_percentile}% of customers by balance)")
    st.write(f"**Number of high-value customers in this dataset:** {is_high_value.sum()} out of {len(real_balance)}")

    st.subheader("Customer Lifetime Value (CLV) Assumptions")
    col1, col2, col3 = st.columns(3)
    with col1:
        net_interest_margin = st.slider("Net Interest Margin (%/year)", 0.5, 5.0, 2.0, step=0.5) / 100
    with col2:
        expected_years = st.slider("Expected Customer Horizon (years)", 1, 10, 5)
    with col3:
        discount_rate = st.slider("Discount Rate (%/year)", 1.0, 10.0, 5.0, step=0.5) / 100

    clv = pd.Series(0.0, index=real_balance.index)
    for year in range(1, expected_years + 1):
        clv += (real_balance * net_interest_margin) / ((1 + discount_rate) ** year)

    financial_df = pd.DataFrame({
        'Balance': real_balance,
        'IsHighValue': is_high_value,
        'PredictedChurn': all_predictions,
        'ActualChurn': y_test.reset_index(drop=True),
        'EstimatedCLV': clv
    })

    predicted_churners = financial_df[financial_df['PredictedChurn'] == 1]
    revenue_at_risk = predicted_churners['EstimatedCLV'].sum()
    high_value_churners = predicted_churners[predicted_churners['IsHighValue']]

    st.subheader("Revenue at Risk")
    col1, col2, col3 = st.columns(3)
    col1.metric("Customers Predicted to Churn", f"{len(predicted_churners):,}")
    col2.metric("Estimated Revenue at Risk", f"€{revenue_at_risk:,.0f}")
    col3.metric("High-Value Customers at Risk", f"{len(high_value_churners):,}")

    st.caption(
        "Revenue at Risk = sum of estimated CLV across all customers the model predicts will churn. "
        "CLV = discounted sum of (Balance x Net Interest Margin) over the expected customer horizon."
    )

    st.subheader("Retention Campaign ROI Estimator")
    col1, col2 = st.columns(2)
    with col1:
        retention_cost = st.slider("Retention Cost per Customer (€)", 10, 500, 50, step=10)
    with col2:
        success_rate = st.slider("Retention Success Rate (%)", 5, 80, 30, step=5) / 100

    total_campaign_cost = len(predicted_churners) * retention_cost
    expected_retained = len(predicted_churners) * success_rate
    expected_value_saved = revenue_at_risk * success_rate
    roi = (expected_value_saved - total_campaign_cost) / total_campaign_cost if total_campaign_cost > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Campaign Cost", f"€{total_campaign_cost:,.0f}")
    col2.metric("Expected Value Saved", f"€{expected_value_saved:,.0f}")
    col3.metric("Estimated ROI", f"{roi:.0%}")

    st.caption(
        "ROI = (Expected Value Saved - Total Campaign Cost) / Total Campaign Cost. "
        "These figures are illustrative and highly sensitive to the assumptions above — "
        "adjust the sliders to reflect your own institution's real costs and margins."
    )