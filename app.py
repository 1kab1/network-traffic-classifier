
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(
    page_title="Network Traffic Classifier",
    page_icon="🛡️",
    layout="wide",
)

MODEL_PATH = Path("models/four_class_unweighted.pkl")

LABELS = {
    0: "BENIGN",
    1: "DDoS",
    2: "PortScan",
    3: "Web Attack",
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("🛡️ Network Traffic Classifier")
st.caption("Machine Learning-Based Network Intrusion Detection System")

st.info(
    "Educational machine-learning prototype. "
    "Predictions are not a substitute for production security monitoring."
)

st.divider()

if not MODEL_PATH.exists():
    st.error(
        "Trained model not found at "
        f"{MODEL_PATH}. Please train the unweighted model first."
    )
    st.stop()

try:
    model_data = load_model()
    model = model_data["model"]
    feature_names = model_data["features"]
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()

st.success("Random Forest model loaded successfully!")

st.subheader("Upload Network Traffic Data")

uploaded_file = st.file_uploader(
    "Choose a CIC-IDS2017 network traffic CSV file",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()

        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        missing_features = [
            feature for feature in feature_names
            if feature not in df.columns
        ]

        if missing_features:
            st.error(
                f"Missing {len(missing_features)} required features."
            )
            st.write("Missing columns:", missing_features)
            st.stop()

        X = df[feature_names].copy()

        X = X.apply(pd.to_numeric, errors="coerce")

        invalid_rows = (
            X.isna().any(axis=1)
            | X.isin([float("inf"), float("-inf")]).any(axis=1)
        )

        if invalid_rows.any():
            st.warning(
                f"Removed {invalid_rows.sum():,} rows "
                "containing missing or infinite feature values."
            )
            X = X.loc[~invalid_rows]

        if X.empty:
            st.error("No valid network traffic rows remain.")
            st.stop()

        predictions = model.predict(X)

        results = df.loc[X.index].copy()
        results["Predicted Traffic"] = [
            LABELS[int(prediction)]
            for prediction in predictions
        ]

        total_flows = len(results)
        benign_count = (
            results["Predicted Traffic"] == "BENIGN"
        ).sum()
        attack_count = total_flows - benign_count

        st.divider()
        st.subheader("Traffic Analysis Overview")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Flows", f"{total_flows:,}")
        col2.metric("Benign Flows", f"{benign_count:,}")
        col3.metric("Predicted Attacks", f"{attack_count:,}")

        st.subheader("Predicted Traffic Distribution")

        distribution = (
            results["Predicted Traffic"]
            .value_counts()
            .reindex(LABELS.values(), fill_value=0)
        )

        st.bar_chart(distribution)

        st.subheader("Classification Results")

        if "Label" in df.columns:
            st.divider()
            st.subheader("Model Evaluation")

            st.caption(
                "Evaluation on the uploaded dataset. "
                "If these rows overlap with training data, "
                "the results are not an independent performance estimate."
            )

            true_labels = pd.to_numeric(
                df.loc[X.index, "Label"], errors="coerce"
            )

            valid_labels = true_labels.isin(LABELS.keys())

            if valid_labels.any():
                y_true = true_labels.loc[valid_labels].astype(int)
                y_pred = pd.Series(
                    predictions, index=X.index
                ).loc[valid_labels].astype(int)

                accuracy = accuracy_score(y_true, y_pred)

                st.metric("Accuracy", f"{accuracy:.2%}")

                st.subheader("Confusion Matrix")

                cm = confusion_matrix(
                    y_true, y_pred, labels=list(LABELS.keys())
                )

                cm_df = pd.DataFrame(
                    cm,
                    index=[f"Actual {name}" for name in LABELS.values()],
                    columns=[
                        f"Predicted {name}"
                        for name in LABELS.values()
                    ],
                )

                st.dataframe(cm_df, use_container_width=True)

                st.subheader("Classification Report")

                report = classification_report(
                    y_true,
                    y_pred,
                    labels=list(LABELS.keys()),
                    target_names=list(LABELS.values()),
                    output_dict=True,
                    zero_division=0,
                )

                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)
            else:
                st.warning("No valid numeric class labels found.")

        st.dataframe(results, use_container_width=True)

        csv_data = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Classification Results",
            data=csv_data,
            file_name="network_traffic_predictions.csv",
            mime="text/csv",
        )

    except Exception as error:
        st.error(f"Error processing uploaded file: {error}")

else:
    st.info("Upload a network traffic CSV file to begin analysis.")

st.divider()
st.caption(
    "Built with Python, Scikit-learn, and Streamlit | "
    "Dataset: CIC-IDS2017"
)
