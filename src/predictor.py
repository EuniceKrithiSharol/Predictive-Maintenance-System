import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split


FEATURES = [

    "Temperature",

    "Vibration",

    "Pressure",

    "Rotation_Speed",

    "Operating_Hours",

    "Humidity"
]


def train_predictive_model(data):

    X = data[
        FEATURES
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


    return model


def predict_machine_failure(

    machine_data,

    model
):

    machine_df = pd.DataFrame(

        [machine_data]
    )


    prediction = model.predict(

        machine_df[
            FEATURES
        ]
    )[0]


    probability = model.predict_proba(

        machine_df[
            FEATURES
        ]
    )[0][1]


    if prediction == 1:

        result = "Maintenance Required"

    else:

        result = "Normal"


    return {

        "Prediction": result,

        "Failure_Risk": float(
            probability
        )
    }
