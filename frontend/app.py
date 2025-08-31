import streamlit as st
import requests
from PIL import Image
import io
import base64
import pandas as pd
import plotly.express as px
import os
import json

# ------------------ CONFIG ------------------
API_URL = "http://127.0.0.1:8000/predict"
st.set_page_config(page_title="🔥 Forest Fire Detection", layout="centered")

# ------------------ HEADER ------------------
st.title("🔥 Forest Fire Detection System")
st.write("Upload an image and the system will predict whether there is a fire or not, with a heatmap overlay.")

# ------------------ IMAGE UPLOAD & FIRE DETECTION ------------------
uploaded_file = st.file_uploader("Upload a forest image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Detect Fire"):
        with st.spinner("Analyzing..."):
            try:
                # Send image to backend
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(API_URL, files=files)

                if response.status_code == 200:
                    result = response.json()

                    # Extract values
                    prediction = result["prediction"].strip().lower()
                    confidence = float(result["confidence"])
                    fire_prob = float(result["class_probabilities"].get("Fire", 0.0))
                    nofire_prob = float(result["class_probabilities"].get("No Fire", 0.0))

                    st.subheader("📊 Prediction Result")

                    # Use probability for final decision
                    final_decision = "fire" if fire_prob >= 0.5 else "no fire"

                    if final_decision == "fire":
                        st.error(f"🔥 Fire Detected! (Fire Probability: {fire_prob:.2%})")
                    else:
                        st.success(f"✅ No Fire Detected (No Fire Probability: {nofire_prob:.2%})")

                    # Warning if fire probability is significant but not dominant
                    if 0.3 < fire_prob < 0.5:
                        st.warning(f"⚠️ Warning: Fire Probability = {fire_prob:.2%}")

                    # Show class probabilities
                    st.write("### Class Probabilities")
                    st.write(f"🔥 Fire: **{fire_prob:.2%}**")
                    st.progress(min(max(fire_prob, 0.0), 1.0))
                    st.write(f"🌲 No Fire: **{nofire_prob:.2%}**")
                    st.progress(min(max(nofire_prob, 0.0), 1.0))

                    # Grad-CAM heatmap (if available)
                    heatmap_base64 = result.get("heatmap", None)
                    if heatmap_base64:
                        st.write("### 🔥 Fire Detection Heatmap")
                        heatmap_bytes = base64.b64decode(heatmap_base64)
                        heatmap_img = Image.open(io.BytesIO(heatmap_bytes))
                        st.image(heatmap_img, caption="Grad-CAM Heatmap", use_container_width=True)
                    else:
                        st.warning("⚠️ No heatmap received from backend.")

                else:
                    st.error(f"❌ Backend Error: {response.status_code} - {response.text}")

            except Exception as e:
                st.error(f"⚠️ Could not connect to backend: {e}")
# ------------------ HISTORICAL DATA SECTION ------------------
st.write("---")
st.header("📊 Indian Forest Fire Trends (2000–2025)")

csv_path = os.path.join("../data/CSV_data/india_forest_fires_2000_2025.csv")

try:
    df = pd.read_csv(csv_path)

    expected_cols = ["Year", "Month", "Forest", "State", "Cause", "Area_Burned_km2"]
    if not all(col in df.columns for col in expected_cols):
        st.error(f"❌ CSV must have columns: {', '.join(expected_cols)}")
    else:
        # Sidebar filters
        st.sidebar.header("🔎 Filters")
        year_filter = st.sidebar.multiselect("Select Year(s)", options=sorted(df["Year"].unique()), default=sorted(df["Year"].unique()))
        month_filter = st.sidebar.multiselect("Select Month(s)", options=sorted(df["Month"].unique()), default=sorted(df["Month"].unique()))
        state_filter = st.sidebar.multiselect("Select State(s)", options=sorted(df["State"].unique()), default=sorted(df["State"].unique()))
        forest_filter = st.sidebar.multiselect("Select Forest(s)", options=sorted(df["Forest"].unique()), default=sorted(df["Forest"].unique()))
        cause_filter = st.sidebar.multiselect("Select Cause(s)", options=sorted(df["Cause"].unique()), default=sorted(df["Cause"].unique()))

        # Apply filters
        filtered_df = df[
            (df["Year"].isin(year_filter)) &
            (df["Month"].isin(month_filter)) &
            (df["State"].isin(state_filter)) &
            (df["Forest"].isin(forest_filter)) &
            (df["Cause"].isin(cause_filter))
        ]

        # Graph type selection
        graph_type = st.selectbox(
            "📊 Select Graph Type",
            ["Yearly Trends", "Monthly Trends", "State-wise", "Forest-wise", "Cause-wise"]
        )

        if graph_type == "Yearly Trends":
            yearly_data = filtered_df.groupby("Year", as_index=False)["Area_Burned_km2"].sum()
            fig = px.line(yearly_data, x="Year", y="Area_Burned_km2",
                          markers=True, title="🔥 Yearly Forest Fire Trends",
                          labels={"Area_Burned_km2": "Total Area Burned (km²)"})
            fig.update_traces(line=dict(width=3, color="red"))

        elif graph_type == "Monthly Trends":
            monthly_data = filtered_df.groupby("Month", as_index=False)["Area_Burned_km2"].sum()
            fig = px.bar(monthly_data, x="Month", y="Area_Burned_km2",
                         title="🔥 Monthly Forest Fire Trends",
                         labels={"Area_Burned_km2": "Total Area Burned (km²)"})

        elif graph_type == "State-wise":
            state_data = filtered_df.groupby("State", as_index=False)["Area_Burned_km2"].sum().sort_values(by="Area_Burned_km2", ascending=False)
            fig = px.bar(state_data, x="State", y="Area_Burned_km2",
                         title="🔥 State-wise Forest Fire Damage",
                         labels={"Area_Burned_km2": "Total Area Burned (km²)"})

        elif graph_type == "Forest-wise":
            forest_data = filtered_df.groupby("Forest", as_index=False)["Area_Burned_km2"].sum().sort_values(by="Area_Burned_km2", ascending=False)
            fig = px.bar(forest_data, x="Forest", y="Area_Burned_km2",
                         title="🌲 Forest-wise Fire Damage",
                         labels={"Area_Burned_km2": "Total Area Burned (km²)"})

        elif graph_type == "Cause-wise":
            cause_data = filtered_df.groupby("Cause", as_index=False)["Area_Burned_km2"].sum().sort_values(by="Area_Burned_km2", ascending=False)
            fig = px.pie(cause_data, names="Cause", values="Area_Burned_km2",
                         title="⚡ Fire Causes Distribution")

        # Style
        fig.update_layout(plot_bgcolor="white",
                          xaxis=dict(showgrid=True, gridcolor="lightgray"),
                          yaxis=dict(showgrid=True, gridcolor="lightgray"),
                          title_font=dict(size=22, color="darkred"),
                          hovermode="x unified")
        fig.update_yaxes(tickformat=",")

        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"⚠️ Could not load CSV data from {csv_path}: {e}")

# ------------------ MAP VISUALIZATION ------------------
st.write("---")
st.header("🗺️ Forest Fire Intensity Map (2000–2025)")

try:
    geojson_path = r"E:\IPL\pythonProject\forest-fire-detection\data\Maps\india_states.geojson"
    with open(geojson_path, "r", encoding="utf-8") as f:
        india_states = json.load(f)

    # Normalize state names
    state_name_map = {
        "Orissa": "Odisha",
        "Pondicherry": "Puducherry",
        "Uttaranchal": "Uttarakhand",
        "Chattisgarh": "Chhattisgarh",
        "NCT of Delhi": "Delhi",
        "Jammu & Kashmir": "Jammu and Kashmir"
    }
    df["State"] = df["State"].replace(state_name_map)

    years = ["All Years"] + sorted(df["Year"].unique().tolist())
    selected_year = st.selectbox("📅 Select Year", years)

    if selected_year != "All Years":
        df_filtered = df[df["Year"] == selected_year]
    else:
        df_filtered = df.copy()

    map_data = df_filtered.groupby("State", as_index=False)["Area_Burned_km2"].sum()

    fig_map = px.choropleth(map_data,
                            geojson=india_states,
                            featureidkey="properties.ST_NM",
                            locations="State",
                            color="Area_Burned_km2",
                            color_continuous_scale="YlOrRd",
                            title=f"🔥 State-wise Forest Fire Damage in India ({selected_year})",
                            labels={"Area_Burned_km2": "Total Area Burned (km²)"})

    fig_map.update_geos(fitbounds="locations",
                        visible=True,
                        showcountries=True,
                        countrycolor="black",
                        showsubunits=True,
                        subunitcolor="black",
                        subunitwidth=0.5,
                        countrywidth=1)

    fig_map.update_traces(hovertemplate="<b>%{location}</b><br>🔥 Burned Area: %{z:,} km²<extra></extra>")
    fig_map.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0},
                          coloraxis_colorbar=dict(title="Burned Area (km²)", ticks="outside"))

    st.plotly_chart(fig_map, use_container_width=True)

except Exception as e:
    st.error(f"⚠️ Could not generate map visualization: {e}")
