
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="TrafficVision AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM FRONTEND DESIGN
# ==========================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #091321, #10243b);
    color: white;
}

[data-testid="stSidebar"] {
    background: #101d30;
    border-right: 1px solid #263c56;
}

.main-title {
    font-size: 40px;
    font-weight: 800;
    text-align: center;
    color: #ffffff;
}

.subtitle {
    text-align: center;
    color: #a9bfd8;
    font-size: 16px;
    margin-bottom: 30px;
}

.section-heading {
    color: #55d6f5;
    font-size: 22px;
    font-weight: bold;
}

.result-card {
    background: linear-gradient(135deg, #153652, #174c65);
    border: 1px solid #37bce0;
    padding: 28px;
    border-radius: 18px;
    text-align: center;
}

.result-name {
    font-size: 30px;
    font-weight: bold;
    color: #70e6ff;
}

.metric-card {
    background: #172a40;
    border: 1px solid #2d4965;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}

.small-text {
    color: #a9bfd8;
    font-size: 13px;
}

div[data-testid="stFileUploader"] {
    background: #14263c;
    border: 1px dashed #48c9ef;
    padding: 18px;
    border-radius: 15px;
}

.stProgress > div > div > div > div {
    background-color: #35c9ed;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# TRAFFIC SIGN CLASS NAMES
# ==========================================

class_names = {
    0: "Speed limit 20 km/h",
    1: "Speed limit 30 km/h",
    2: "Speed limit 50 km/h",
    3: "Speed limit 60 km/h",
    4: "Speed limit 70 km/h",
    5: "Speed limit 80 km/h",
    6: "End of speed limit 80 km/h",
    7: "Speed limit 100 km/h",
    8: "Speed limit 120 km/h",
    9: "No passing",
    10: "No passing for vehicles over 3.5 tons",
    11: "Right-of-way at next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 tons"
}


# ==========================================
# LOAD TRAINED CNN MODEL
# ==========================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "traffic_sign_cnn.keras"
    )


@st.cache_data
def load_mapping():
    with open("class_mapping.json", "r") as file:
        return json.load(file)


try:
    model = load_model()
    class_mapping = load_mapping()

except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("# 🚦 TrafficVision AI")
    st.markdown("### Smart Traffic Sign Detection")

    st.markdown("---")

    st.markdown("## 🧠 About the Project")

    st.write(
        "An AI-powered traffic sign classification "
        "system built using Convolutional Neural Networks."
    )

    st.markdown("---")

    st.markdown("## ⚙️ Model Details")

    st.write("""
    **Architecture:** CNN

    **Dataset:** GTSRB

    **Image Input:** 32 × 32

    **Classes:** 43

    **Test Accuracy:** 97.84%

    **Framework:** TensorFlow
    """)

    st.markdown("---")

    st.markdown("## 📂 Supported Images")
    st.write("JPG | JPEG | PNG | WEBP")

    st.markdown("---")

    st.markdown("## 🔄 Process")

    st.write("""
    1. Upload image
    2. Image preprocessing
    3. CNN analysis
    4. Class prediction
    5. Display results
    """)

    st.markdown("---")

    st.caption("Developed with Python + CNN + Streamlit")


# ==========================================
# MAIN FRONTEND
# ==========================================

st.markdown(
    '<div class="main-title">🚦 Traffic Sign Recognition</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent Traffic Sign Classification Using Deep Learning'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-heading">📤 Upload Your Traffic Sign</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload a traffic sign image. The AI model will "
    "automatically analyze it and generate a prediction."
)


# ==========================================
# IMAGE UPLOADER
# ==========================================

uploaded_file = st.file_uploader(
    "Choose a traffic sign image",
    type=["jpg", "jpeg", "png", "webp"],
    key="traffic_image"
)


# ==========================================
# AUTOMATIC IMAGE ANALYSIS
# ==========================================

if uploaded_file is not None:

    try:

        # Read uploaded image
        image = Image.open(uploaded_file).convert("RGB")

        st.success("Image uploaded successfully!")

        left, right = st.columns(2, gap="large")

        with left:

            st.markdown("### 🖼️ Uploaded Image")

            st.image(
                image,
                caption="Your uploaded traffic sign",
                use_container_width=True
            )

            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>Image Information</h4>
                    <p>Width: {image.width} pixels</p>
                    <p>Height: {image.height} pixels</p>
                    <p>Format: {uploaded_file.type}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Preprocess image
        resized_image = image.resize((32, 32))

        image_array = np.array(
            resized_image,
            dtype=np.float32
        ) / 255.0

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # Automatically run CNN prediction
        with st.spinner("🧠 AI is analyzing your traffic sign..."):

            predictions = model.predict(
                image_array,
                verbose=0
            )[0]

        # Get highest prediction
        predicted_index = int(np.argmax(predictions))

        confidence = float(
            predictions[predicted_index]
        ) * 100

        # Convert model index to original class ID
        if str(predicted_index) not in class_mapping:
            raise ValueError(
                "Predicted index not found in class mapping."
            )

        original_class_id = int(
            class_mapping[str(predicted_index)]
        )

        predicted_name = class_names.get(
            original_class_id,
            "Unknown traffic sign"
        )

        # Display result
        with right:

            st.markdown("### 🎯 AI Analysis Result")

            st.markdown(
                f"""
                <div class="result-card">
                    <p class="small-text">
                        PREDICTED TRAFFIC SIGN
                    </p>
                    <div class="result-name">
                        {predicted_name}
                    </div>
                    <h1 style="color:white;">
                        {confidence:.2f}%
                    </h1>
                    <p class="small-text">
                        Prediction confidence
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("### 📊 Confidence Score")

            st.progress(
                min(confidence / 100, 1.0)
            )

            st.write(
                f"Model confidence: **{confidence:.2f}%**"
            )

        # ==================================
        # TOP 3 PREDICTIONS
        # ==================================

        st.markdown("---")

        st.markdown(
            '<div class="section-heading">'
            '🏆 Top 3 Predictions'
            '</div>',
            unsafe_allow_html=True
        )

        top_indices = np.argsort(
            predictions
        )[-3:][::-1]

        for rank, index in enumerate(top_indices, start=1):

            class_id = int(
                class_mapping[str(int(index))]
            )

            sign_name = class_names.get(
                class_id,
                "Unknown"
            )

            score = float(
                predictions[index]
            ) * 100

            col1, col2 = st.columns([3, 1])

            with col1:
                st.write(f"**{rank}. {sign_name}**")

            with col2:
                st.write(f"**{score:.2f}%**")

            st.progress(
                min(score / 100, 1.0)
            )

        st.markdown("---")

        st.success(
            "✅ Traffic sign analysis completed successfully!"
        )

        st.caption(
            "This is an AI classification result, not an "
            "official road-safety decision."
        )

    except Exception as error:

        st.error(
            f"Image analysis failed: {error}"
        )

else:

    st.markdown("---")

    st.info(
        "👆 Upload an image above. Your CNN model will "
        "automatically analyze it."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <h3>🧠 CNN Intelligence</h3>
                <p>Deep learning image classification.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <h3>🚦 43 Classes</h3>
                <p>Recognizes 43 traffic sign categories.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <h3>⚡ Automatic Analysis</h3>
                <p>Prediction starts immediately after upload.</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown(
    """
    <center style="color:#8fa9c4;">
        TrafficVision AI | CNN-Based Traffic Sign Recognition
        <br>
        Built with Python, TensorFlow and Streamlit
    </center>
    """,
    unsafe_allow_html=True
)
