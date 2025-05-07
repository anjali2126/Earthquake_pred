import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.geocoders import Nominatim
from math import radians, cos, sin, asin, sqrt

# ---- Geocoding ----
def geocode_location(place):
    geolocator = Nominatim(user_agent="eq_dashboard")
    location = geolocator.geocode(place)
    if location:
        return location.latitude, location.longitude
    return None, None

# ---- Haversine Distance ----
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

# ---- EarthquakeCNN Model ----
class EarthquakeCNN(nn.Module):
    def __init__(self, input_channels=3):
        super(EarthquakeCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(2, 2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool1d(2, 2)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool1d(2, 2)
        self.flat_features = 512 * 375
        self.fc1 = nn.Linear(self.flat_features, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return torch.sigmoid(self.fc3(x))

# ---- Load Model ----
@st.cache_resource
def load_model():
    model = EarthquakeCNN()
    model.load_state_dict(torch.load("earthquake_cnn_weights.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

# ---- Load Metadata ----
@st.cache_data
def load_metadata():
    return pd.read_csv("merged.csv")

# ---- Load Trace Data ----
def load_trace(hdf5_path, trace_name):
    with h5py.File(hdf5_path, 'r') as f:
        return f[f"data/{trace_name}"][:]

# ---- Prediction ----
def predict(model, signal):
    tensor_input = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return model(tensor_input).item()

# ---- Streamlit UI ----
st.set_page_config(page_title="Earthquake Predictor", layout="wide")
st.title("🌍 Earthquake Prediction Dashboard")
st.markdown("""<style>
    .st-emotion-cache-1kyxreq {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
</style>""", unsafe_allow_html=True)

model = load_model()
metadata = load_metadata()

with st.container():
    place = st.text_input("📍 Enter a place (e.g., California, Delhi, Tokyo):")

    if place:
        lat, lon = geocode_location(place)

        if lat is None:
            st.error("❌ Location not found.")
        else:
            st.success(f"✅ Found location: {place} at ({lat:.2f}, {lon:.2f})")

            metadata['distance_km'] = metadata.apply(
                lambda row: haversine(lat, lon, row['source_latitude'], row['source_longitude']), axis=1
            )
            nearest = metadata.sort_values('distance_km').iloc[0]
            trace_name = nearest['trace_name']
            event_time = nearest['source_origin_time']  
            magnitude = nearest['source_magnitude']
            st.markdown(f"### Nearest trace: `{trace_name}` ({nearest['distance_km']:.2f} km away)")
            st.markdown(f"previous recorded Magnitude: {magnitude}")

            try:
                signal = load_trace("merged.hdf5", trace_name)
                probability = predict(model, signal)
                label = "🌋 Earthquake likely to occur" if probability >= 0.5 else "✅ No earthquake detected for now"

                st.subheader("📊 Prediction Result")
                st.success(f"**{label}**\n\n**Probability:** `{probability:.3f}`")

                # Plot waveform
                st.subheader("📈 Seismic Waveform")
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(signal[:, 0], label='Channel 1')
                ax.plot(signal[:, 1], label='Channel 2')
                ax.plot(signal[:, 2], label='Channel 3')
                ax.set_title("Seismic Trace - 3000 Samples x 3 Channels")
                ax.set_xlabel("Time (samples)")
                ax.set_ylabel("Amplitude")
                ax.legend()
                st.pyplot(fig)

                # Heatmap visualization (correlation)
                st.subheader("🧠 Channel Correlation Heatmap")
                corr = pd.DataFrame(signal).corr()
                fig2, ax2 = plt.subplots()
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
                ax2.set_title("Correlation Between Channels")
                st.pyplot(fig2)

                # Show location on map
                st.subheader("🗺️ Trace Epicenter")
                st.map(pd.DataFrame({'lat': [nearest['source_latitude']], 'lon': [nearest['source_longitude']]}))

            except Exception as e:
                st.error(f"❌ Failed to load or predict: {e}")

