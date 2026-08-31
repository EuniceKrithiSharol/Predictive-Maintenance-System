import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide"
)


# -------------------------------------------------
# CREATE MACHINE DATASET
# -------------------------------------------------

@st.cache_data
def create_machine_dataset():

    np.random.seed(42)

    samples = 1000


    temperature = np.random.normal(
        75,
        15,
        samples
    )


    vibration = np.random.normal(
        4,
        2,
        samples
    )


    pressure = np.random.normal(
        100,
        20,
        samples
    )


    rotation_speed = np.random.normal(
        1500,
        300,
        samples
    )


    operating_hours = np.random.randint(
        100,
        10000,
        samples
    )


    humidity = np.random.normal(
        50,
        15,
        samples
    )


    # ---------------------------------------------
    # FAILURE LOGIC
    # ---------------------------------------------

    failure_risk = (

        (temperature > 95).astype(int)

        +

        (vibration > 7).astype(int)

        +

        (pressure > 130).astype(int)

        +

        (operating_hours > 8000).astype(int)

        +

        (humidity > 75).astype(int)
    )


    failure = np.where(

        failure_risk >= 2,

        1,

        0
    )


    # Add small randomness

    random_failures = np.random.choice(

        [0, 1],

        size=samples,

        p=[0.95, 0.05]
    )


    failure = np.where(

        random_failures == 1,

        1,

        failure
    )


    data = pd.DataFrame({

        "Temperature": temperature,

        "Vibration": vibration,

        "Pressure": pressure,

        "Rotation_Speed": rotation_speed,

        "Operating_Hours": operating_hours,

        "Humidity": humidity,

        "Failure": failure
    })


    data["Temperature"] = data[
        "Temperature"
    ].clip(
        lower=20
    )


    data["Vibration"] = data[
        "Vibration"
    ].clip(
        lower=0.1
    )


    data["Pressure"] = data[
        "Pressure"
    ].clip(
        lower=20
    )


    data["Humidity"] = data[
        "Humidity"
    ].clip(
        lower=0,
        upper=100
    )


    return data


machine_data = create_machine_dataset()


# -------------------------------------------------
# FEATURES
# -------------------------------------------------

features = [

    "Temperature",

    "Vibration",

    "Pressure",

    "Rotation_Speed",

    "Operating_Hours",

    "Humidity"
]


# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------

@st.cache_resource
def train_model(data):

    X = data[
        features
    ]


    y = data[
        "Failure"
    ]


    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42,

        stratify=y
    )


    model = RandomForestClassifier(

        n_estimators=200,

        random_state=42
    )


    model.fit(

        X_train,

        y_train
    )


    predictions = model.predict(

        X_test
    )


    accuracy = accuracy_score(

        y_test,

        predictions
    )


    return model, accuracy


model, model_accuracy = train_model(

    machine_data
)


# -------------------------------------------------
# TITLE
# -------------------------------------------------

st.title(
    "⚙️ Predictive Maintenance System"
)


st.markdown(
    "Use Machine Learning to analyze equipment sensor data "
    "and predict potential machine failure."
)


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.header(
    "🤖 How It Works"
)


st.sidebar.info(
    """
    1. Machine sensor data is collected.

    2. Important operating features are analyzed.

    3. A Random Forest model learns failure patterns.

    4. Current machine conditions are entered.

    5. The system predicts maintenance risk.
    """
)


# -------------------------------------------------
# DASHBOARD METRICS
# -------------------------------------------------

st.subheader(
    "📊 Machine Dataset Overview"
)


total_machines = len(
    machine_data
)


normal_machines = len(

    machine_data[
        machine_data[
            "Failure"
        ] == 0
    ]
)


high_risk_machines = len(

    machine_data[
        machine_data[
            "Failure"
        ] == 1
    ]
)


col1, col2, col3, col4 = st.columns(4)


col1.metric(

    "Total Machine Records",

    total_machines
)


col2.metric(

    "Normal Records",

    normal_machines
)


col3.metric(

    "Potential Failures",

    high_risk_machines
)


col4.metric(

    "Model Accuracy",

    f"{model_accuracy * 100:.1f}%"
)


# -------------------------------------------------
# DATA PREVIEW
# -------------------------------------------------

st.subheader(
    "📁 Machine Sensor Data"
)


display_data = machine_data.copy()


display_data[
    "Status"
] = np.where(

    display_data[
        "Failure"
    ] == 1,

    "Potential Failure",

    "Normal"
)


st.dataframe(

    display_data.head(20),

    use_container_width=True
)


# -------------------------------------------------
# MACHINE STATUS DISTRIBUTION
# -------------------------------------------------

st.subheader(
    "📊 Machine Status Distribution"
)


status_counts = (

    display_data[
        "Status"
    ]
    .value_counts()
    .reset_index()
)


status_counts.columns = [

    "Status",

    "Machines"
]


fig_status = px.pie(

    status_counts,

    names="Status",

    values="Machines",

    title="Machine Operating Status"
)


st.plotly_chart(

    fig_status,

    use_container_width=True
)


# -------------------------------------------------
# SENSOR ANALYSIS
# -------------------------------------------------

st.subheader(
    "📈 Sensor Data Analysis"
)


fig_sensor = px.scatter(

    display_data,

    x="Temperature",

    y="Vibration",

    color="Status",

    size="Operating_Hours",

    hover_data=[

        "Pressure",

        "Rotation_Speed",

        "Humidity"
    ],

    title="Temperature vs Vibration Analysis"
)


st.plotly_chart(

    fig_sensor,

    use_container_width=True
)


# -------------------------------------------------
# PREDICTIVE MAINTENANCE SECTION
# -------------------------------------------------

st.divider()


st.header(
    "🔧 Analyze Machine Condition"
)


col1, col2 = st.columns(2)


with col1:

    temperature = st.number_input(

        "Machine Temperature",

        min_value=20.0,

        max_value=150.0,

        value=75.0
    )


    vibration = st.number_input(

        "Vibration Level",

        min_value=0.1,

        max_value=20.0,

        value=4.0
    )


    pressure = st.number_input(

        "Operating Pressure",

        min_value=20.0,

        max_value=250.0,

        value=100.0
    )


with col2:

    rotation_speed = st.number_input(

        "Rotation Speed (RPM)",

        min_value=100.0,

        max_value=5000.0,

        value=1500.0
    )


    operating_hours = st.number_input(

        "Total Operating Hours",

        min_value=0,

        max_value=50000,

        value=3000
    )


    humidity = st.slider(

        "Environmental Humidity",

        min_value=0.0,

        max_value=100.0,

        value=50.0
    )


# -------------------------------------------------
# PREDICTION
# -------------------------------------------------

if st.button(
    "🔍 Predict Maintenance Risk"
):

    input_data = pd.DataFrame({

        "Temperature": [

            temperature
        ],

        "Vibration": [

            vibration
        ],

        "Pressure": [

            pressure
        ],

        "Rotation_Speed": [

            rotation_speed
        ],

        "Operating_Hours": [

            operating_hours
        ],

        "Humidity": [

            humidity
        ]
    })


    prediction = model.predict(

        input_data
    )[0]


    probability = model.predict_proba(

        input_data
    )[0][1]


    st.divider()


    st.subheader(
        "🤖 Predictive Maintenance Result"
    )


    col1, col2 = st.columns(2)


    col1.metric(

        "Failure Risk",

        f"{probability * 100:.2f}%"
    )


    if prediction == 1:

        col2.metric(

            "Machine Status",

            "Maintenance Required"
        )


        st.error(
            "⚠️ High Risk of Machine Failure Detected"
        )


        st.warning(
            "The machine sensor values indicate abnormal "
            "operating conditions. Maintenance inspection "
            "is recommended."
        )


    else:

        col2.metric(

            "Machine Status",

            "Operating Normally"
        )


        st.success(
            "✅ Machine Operating Conditions Appear Normal"
        )


        st.info(
            "The Machine Learning model predicts a low "
            "risk of immediate failure."
        )


# -------------------------------------------------
# FEATURE IMPORTANCE
# -------------------------------------------------

st.divider()


st.subheader(
    "🧠 Machine Learning Feature Importance"
)


feature_importance = pd.DataFrame({

    "Feature": features,

    "Importance": model.feature_importances_
})


feature_importance = feature_importance.sort_values(

    by="Importance",

    ascending=False
)


fig_importance = px.bar(

    feature_importance,

    x="Importance",

    y="Feature",

    orientation="h",

    title="Features Influencing Machine Failure Prediction"
)


st.plotly_chart(

    fig_importance,

    use_container_width=True
)


# -------------------------------------------------
# CSV UPLOAD
# -------------------------------------------------

st.divider()


st.header(
    "📤 Analyze Machine Sensor CSV"
)


uploaded_file = st.file_uploader(

    "Upload machine sensor data",

    type=["csv"]
)


if uploaded_file is not None:

    uploaded_data = pd.read_csv(

        uploaded_file
    )


    missing_columns = [

        feature

        for feature in features

        if feature not in uploaded_data.columns
    ]


    if missing_columns:

        st.error(

            "Missing required columns: "

            + ", ".join(
                missing_columns
            )
        )


    else:

        predictions = model.predict(

            uploaded_data[
                features
            ]
        )


        probabilities = model.predict_proba(

            uploaded_data[
                features
            ]
        )[:, 1]


        uploaded_data[
            "Failure_Risk"
        ] = probabilities


        uploaded_data[
            "Prediction"
        ] = np.where(

            predictions == 1,

            "Maintenance Required",

            "Normal"
        )


        st.subheader(
            "Machine Prediction Results"
        )


        st.dataframe(

            uploaded_data,

            use_container_width=True
        )


        maintenance_count = len(

            uploaded_data[

                uploaded_data[
                    "Prediction"
                ] == "Maintenance Required"

            ]
        )


        st.warning(

            f"Machines requiring attention: "
            f"{maintenance_count}"
        )


# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.divider()


st.caption(

    "Predictive Maintenance System | "

    "Python • Machine Learning • "

    "Random Forest • Predictive Analytics • "

    "Industrial AI • Streamlit"
)
