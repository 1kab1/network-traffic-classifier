
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Network Traffic Classifier",
    page_icon="🛡️",
    layout="wide",
)

MODEL_PATH = Path("models/four_class_unweighted.pkl")
DEMO_PATH = Path("data/demo_sample.csv")

LABELS = {
    0: "BENIGN",
    1: "DDoS",
    2: "PortScan",
    3: "Web Attack",
}


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("🛡️ Network Traffic Classifier")
st.caption(
    "Machine Learning-Based Network Intrusion Detection System"
)

st.info(
    "Educational machine-learning prototype. "
    "Predictions are not a substitute for production security monitoring."
)

st.divider()

if not MODEL_PATH.exists():
    st.error(
        f"Trained model not found at {MODEL_PATH}. "
        "Please make sure the model file exists."
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


# --------------------------------------------------
# SELECT DATA SOURCE
# --------------------------------------------------

st.subheader("Try the Classifier")

data_source = st.radio(
    "Choose how to analyze network traffic:",
    ["🚀 Try Sample Data", "📁 Upload Your Own CSV"],
    horizontal=True,
)

df = None

if data_source == "🚀 Try Sample Data":
    st.write(
        "Explore the classifier using 100 sample flows: "
        "25 BENIGN, 25 DDoS, 25 PortScan, and 25 Web Attack."
    )

    if st.button("🚀 Run Sample Analysis", type="primary"):
        st.session_state["run_demo"] = True

    if st.session_state.get("run_demo", False):
        if DEMO_PATH.exists():
            try:
                df = pd.read_csv(DEMO_PATH)
                st.success(
                    f"Demo dataset loaded successfully: "
                    f"{len(df):,} network traffic flows!"
                )
            except Exception as error:
                st.error(f"Could not load demo data: {error}")
        else:
            st.error(
                "Demo dataset not found at data/demo_sample.csv."
            )

else:
    st.session_state["run_demo"] = False

    uploaded_file = st.file_uploader(
        "Choose a CIC-IDS2017 network traffic CSV file",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as error:
            st.error(f"Could not read uploaded CSV: {error}")


# --------------------------------------------------
# PROCESS DATA AND MAKE PREDICTIONS
# --------------------------------------------------

if df is not None:
    try:
        df.columns = df.columns.str.strip()

        st.divider()
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        missing_features = [
            feature
            for feature in feature_names
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
            | np.isinf(X.to_numpy()).any(axis=1)
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

        # ------------------------------------------
        # TRAFFIC STATISTICS
        # ------------------------------------------

        st.divider()
        st.subheader("Traffic Analysis Overview")

        total_flows = len(results)

        benign_count = (
            results["Predicted Traffic"] == "BENIGN"
        ).sum()

        attack_count = total_flows - benign_count

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Flows",
            f"{total_flows:,}",
        )

        col2.metric(
            "Benign Flows",
            f"{benign_count:,}",
        )

        col3.metric(
            "Predicted Attacks",
            f"{attack_count:,}",
        )

        # ------------------------------------------
        # TRAFFIC DISTRIBUTION CHART
        # ------------------------------------------

        st.subheader("Predicted Traffic Distribution")

        distribution = (
            results["Predicted Traffic"]
            .value_counts()
            .reindex(list(LABELS.values()), fill_value=0)
        )

        st.bar_chart(distribution)

        # ------------------------------------------
        # MODEL EVALUATION
        # ------------------------------------------

        if "Label" in df.columns:
            st.divider()
            st.subheader("Model Evaluation")

            st.caption(
                "Evaluation on the selected dataset. "
                "Demo samples come from model-development data "
                "and may overlap with training examples. "
                "These results are not an independent "
                "real-world performance estimate."
            )

            true_labels = pd.to_numeric(
                df.loc[X.index, "Label"],
                errors="coerce",
            )

            valid_labels = true_labels.isin(LABELS.keys())

            if valid_labels.any():
                y_true = (
                    true_labels.loc[valid_labels]
                    .astype(int)
                )

                y_pred = (
                    pd.Series(
                        predictions,
                        index=X.index,
                    )
                    .loc[valid_labels]
                    .astype(int)
                )

                accuracy = accuracy_score(
                    y_true,
                    y_pred,
                )

                st.metric(
                    "Accuracy",
                    f"{accuracy:.2%}",
                )

                st.subheader("Confusion Matrix")

                cm = confusion_matrix(
                    y_true,
                    y_pred,
                    labels=list(LABELS.keys()),
                )

                cm_df = pd.DataFrame(
                    cm,
                    index=[
                        f"Actual {name}"
                        for name in LABELS.values()
                    ],
                    columns=[
                        f"Predicted {name}"
                        for name in LABELS.values()
                    ],
                )

                st.dataframe(
                    cm_df,
                    use_container_width=True,
                )

                st.subheader("Classification Report")

                report = classification_report(
                    y_true,
                    y_pred,
                    labels=list(LABELS.keys()),
                    target_names=list(LABELS.values()),
                    output_dict=True,
                    zero_division=0,
                )

                report_df = pd.DataFrame(
                    report
                ).transpose()

                st.dataframe(
                    report_df,
                    use_container_width=True,
                )

            else:
                st.warning(
                    "No valid numeric class labels found."
                )

        # ------------------------------------------
        # CLASSIFICATION RESULTS
        # ------------------------------------------

        st.divider()
        st.subheader("Classification Results")

        st.dataframe(
            results,
            use_container_width=True,
        )

        csv_data = results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download Classification Results",
            data=csv_data,
            file_name="network_traffic_predictions.csv",
            mime="text/csv",
        )

    except Exception as error:
        st.error(
            f"Error processing network traffic data: {error}"
        )

else:
    st.info(
        "Click Run Sample Analysis or upload "
        "a network traffic CSV file to begin."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Built with Python, Scikit-learn, and Streamlit | "
    "Dataset: CIC-IDS2017"
)
