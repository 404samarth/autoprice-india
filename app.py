import streamlit as st
import numpy as np
import pickle
import gzip

# ── Load model and encoders ──────────────────────────
@st.cache_resource
def load_files():
    model      = pickle.load(open('car_model (1).pkl', 'rb'))
    le_brand   = pickle.load(open('le_brand.pkl', 'rb'))
    le_car     = pickle.load(open('le_car.pkl', 'rb'))
    le_variant = pickle.load(open('le_variant.pkl', 'rb'))
    le_fuel    = pickle.load(open('le_fuel.pkl', 'rb'))
    le_trans   = pickle.load(open('le_transmission.pkl', 'rb'))
    le_owner   = pickle.load(open('le_owner.pkl', 'rb'))

    mapping = pickle.load(gzip.open('car_mapping (1).pkl', 'rb'))

    return (
        model,
        le_brand,
        le_car,
        le_variant,
        le_fuel,
        le_trans,
        le_owner,
        mapping
    )

# ── Load everything ──────────────────────────────────
model, le_brand, le_car, le_variant, le_fuel, le_trans, le_owner, mapping = load_files()

# ── Page Setup ───────────────────────────────────────
st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Indian Car Price Predictor")
st.caption("Get estimated resale price based on real data")
st.caption("Created by Samarth")
st.divider()

# ── Step 1: Brand ────────────────────────────────────
st.markdown("### Select Brand")
brand = st.selectbox("Brand", sorted(mapping.keys()), label_visibility="collapsed")

# ── Step 2: Car Model ────────────────────────────────
st.markdown("### Select Car Model")
car = st.selectbox("Car Model", sorted(mapping[brand].keys()), label_visibility="collapsed")

# ── Step 3: Variant ──────────────────────────────────
st.markdown("### Select Variant")
variant = st.selectbox("Variant", mapping[brand][car], label_visibility="collapsed")

st.divider()

# ── Step 4: Other Details ────────────────────────────
st.markdown("### Enter More Details")

col1, col2 = st.columns(2)

with col1:
    year      = st.number_input("Year of Purchase", min_value=1992, max_value=2020, value=2015)
    km_driven = st.number_input("KM Driven", min_value=500, max_value=500000, value=40000, step=1000)
    fuel      = st.selectbox("Fuel Type", sorted(le_fuel.classes_.tolist()))

with col2:
    transmission = st.selectbox("Transmission", sorted(le_trans.classes_.tolist()))
    owner        = st.selectbox("Owner Type", sorted(le_owner.classes_.tolist()))

st.divider()

# ── Predict ──────────────────────────────────────────
if st.button("🔍 Predict Resale Price", use_container_width=True):
    try:
        brand_enc   = le_brand.transform([brand])[0]
        car_enc     = le_car.transform([car])[0]
        variant_enc = le_variant.transform([variant])[0]
        fuel_enc    = le_fuel.transform([fuel])[0]
        trans_enc   = le_trans.transform([transmission])[0]
        owner_enc   = le_owner.transform([owner])[0]

        input_data = np.array([[brand_enc, car_enc, variant_enc, year,
                                 km_driven, fuel_enc, trans_enc, owner_enc]])

        price = model.predict(input_data)[0]
        lakhs = price / 100000

        st.success(f"### 💰 Estimated Resale Price: ₹{price:,.0f}")
        st.info(f"That is approximately **₹ {lakhs:.2f} Lakhs**")

        # Show input summary
        with st.expander("View your inputs"):
            st.write(f"**Car:** {brand} {car} {variant}")
            st.write(f"**Year:** {year}")
            st.write(f"**KM Driven:** {km_driven:,} km")
            st.write(f"**Fuel:** {fuel}")
            st.write(f"**Transmission:** {transmission}")
            st.write(f"**Owner:** {owner}")

    except Exception as e:
        st.error(f"Something went wrong: {e}")

st.caption("Built with CarDekho dataset · Random Forest Model · Made by samarth404")
sr.caption("To view more projects: https://github.com/404samarth") 
