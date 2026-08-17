import streamlit as st
import requests

st.set_page_config(page_title="SuperKart Demand Forecaster", layout="wide")
st.title("🛒 SuperKart Sales Prediction App")

st.markdown("Enter product and store attributes to forecast total sales revenue.")

col1, col2 = st.columns(2)

with col1:
    mrp = st.number_input("Product MRP ($)", min_value=1.0, max_value=1000.0, value=140.0)
    weight = st.number_input("Product Weight (g)", min_value=0.1, max_value=100.0, value=12.5)
    area = st.number_input("Product Allocated Area Ratio", min_value=0.001, max_value=1.0, value=0.06, format="%.4f")
    sugar = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
    prod_type = st.selectbox("Product Type", [
        "Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household",
        "Baking Goods", "Snack Foods", "Frozen Foods", "Breakfast", "Health and Hygiene",
        "Hard Drinks", "Canned", "Breads", "Starchy Foods", "Others", "Seafood"
    ])

with col2:
    store_age = st.number_input("Store Age (Years)", min_value=0, max_value=100, value=15)
    store_size = st.selectbox("Store Size", ["High", "Medium", "Small"])
    city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
    store_type = st.selectbox("Store Type", [
        "Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"
    ])

if st.button("Predict Store Sales", use_container_width=True):
    payload = {
        "Product_MRP": mrp,
        "Product_Weight": weight,
        "Product_Allocated_Area": area,
        "Product_Sugar_Content": sugar,
        "Product_Type": prod_type,
        "Store_Age_Years": store_age,
        "Store_Size": store_size,
        "Store_Location_City_Type": city_type,
        "Store_Type": store_type
    }
    
    try:
        response = requests.post("http://flask-backend:5000/predict", json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            st.success(f"### Predicted Total Sales: ${result['predicted_sales']:,.2f}")
        else:
            st.error(f"Error {response.status_code}: {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Could not reach the Flask backend service. Ensure both containers are active on the Docker bridge network.")
