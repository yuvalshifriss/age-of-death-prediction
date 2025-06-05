import streamlit as st
import pandas as pd
import plotly.express as px
import os
from typing import List

from common_utils import load_data


def show_summary_stats(df: pd.DataFrame) -> None:
    st.subheader("📊 Summary of Deceased Patients")
    st.dataframe(df.describe())


def show_condition_percentages(df: pd.DataFrame) -> None:
    st.subheader("🩺 Condition Percentages (Deceased Patients)")
    conditions = {
        "Anaemia": df['anaemia'].mean() * 100,
        "Diabetes": df['diabetes'].mean() * 100,
        "High Blood Pressure": df['high_blood_pressure'].mean() * 100,
        "Smoking": df['smoking'].mean() * 100
    }
    condition_df = pd.DataFrame(conditions.items(), columns=["Condition", "Percentage"])
    fig = px.bar(
        condition_df,
        x="Percentage",
        y="Condition",
        orientation='h',
        title="Condition Prevalence",
        text="Percentage"
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
    st.plotly_chart(fig, use_container_width=True)


def show_feature_distributions(df: pd.DataFrame, selected_features: List[str]) -> None:
    st.subheader("📈 Feature Distributions (based on selected features)")
    for feature in selected_features:
        fig = px.histogram(
            df,
            x=feature,
            nbins=20,
            title=f'Distribution of {feature}'
        )
        st.plotly_chart(fig, use_container_width=True)


def show_correlation_explorer(df: pd.DataFrame) -> None:
    st.subheader("📌 Correlation Explorer")
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    corr_matrix = df[numeric_cols].corr().round(2)
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Correlation Heatmap of Numeric Features"
    )
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(layout="wide", page_title="Heart Failure Clinical Data Explorer")

    st.title("💓 Heart Failure Clinical Data Explorer")
    st.markdown("This app explores data for **patients who died** in a heart failure study.")

    file_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'data', 'heart_failure_clinical_records.csv'))
    df = load_data(file_path, only_dead=True)
    df = df.drop(columns=['DEATH_EVENT', 'time'])

    # Sidebar controls
    st.sidebar.header("🔧 Feature Distribution Options")
    selected_features = st.sidebar.multiselect(
        "Select features to show distributions for:",
        ['age', 'ejection_fraction', 'serum_sodium', 'serum_creatinine'],
        default=['age', 'ejection_fraction']
    )

    # Main display sections
    show_summary_stats(df)
    show_condition_percentages(df)
    show_feature_distributions(df, selected_features)
    show_correlation_explorer(df)


if __name__ == '__main__':
    main()
