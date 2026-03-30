import joblib
import pandas as pd

model = joblib.load("models/yield_model.pkl")
encoder = joblib.load("models/crop_encoder.pkl")


def predict_yield(crop, rainfall, pesticide, temperature):

    crop_encoded = encoder.transform([crop])[0]

    features = pd.DataFrame([{
        "Crop": crop_encoded,
        "Rainfall": rainfall,
        "Pesticide": pesticide,
        "Temperature": temperature
    }])

    prediction = model.predict(features)[0]
    return round(prediction, 2)


if __name__ == "__main__":
    print(predict_yield("Wheat", 1000, 200, 25))