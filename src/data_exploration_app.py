import streamlit as st
import pandas as pd
import plotly.express as px
import os


def main():
    st.set_page_config(layout="wide", page_title="Heart Failure Clinical Data Explorer")

    # Load data
    @st.cache_data
    def load_data():
        file_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'data', 'heart_failure_clinical_records.csv'))
        return pd.read_csv(file_path)

    df = load_data()
    dead_df = df[df['DEATH_EVENT'] == 1]

    st.title("💓 Heart Failure Clinical Data Explorer")
    st.markdown("This app explores data for **patients who died** in a heart failure study.")

    # Sidebar controls
    st.sidebar.header("🔧 Feature Distribution Options")
    selected_features = st.sidebar.multiselect(
        "Select features to show distributions for:",
        ['age', 'ejection_fraction', 'serum_sodium', 'serum_creatinine'],
        default=['age', 'ejection_fraction']
    )

    # Summary stats
    st.subheader("📊 Summary of Deceased Patients")
    st.dataframe(dead_df.describe())

    # Condition percentages (excluding "Male")
    st.subheader("🩺 Condition Percentages (Deceased Patients)")
    conditions = {
        "Anaemia": dead_df['anaemia'].mean() * 100,
        "Diabetes": dead_df['diabetes'].mean() * 100,
        "High Blood Pressure": dead_df['high_blood_pressure'].mean() * 100,
        "Smoking": dead_df['smoking'].mean() * 100
    }
    condition_df = pd.DataFrame(conditions.items(), columns=["Condition", "Percentage"])
    fig_bar = px.bar(
        condition_df,
        x="Percentage",
        y="Condition",
        orientation='h',
        title="Condition Prevalence",
        text="Percentage"
    )
    fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_bar.update_layout(margin=dict(l=40, r=40, t=50, b=40))
    st.plotly_chart(fig_bar, use_container_width=True)

    # Feature distributions
    st.subheader("📈 Feature Distributions (based on selected features)")
    for feature in selected_features:
        fig = px.histogram(
            dead_df,
            x=feature,
            nbins=20,
            title=f'Distribution of {feature}'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Correlation Explorer
    st.subheader("📌 Correlation Explorer")

    # Numeric columns
    numeric_cols = dead_df.select_dtypes(include='number').columns.tolist()
    corr_matrix = dead_df[numeric_cols].corr().round(2)

    # Correlation Heatmap
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Correlation Heatmap of Numeric Features"
    )
    st.plotly_chart(fig_corr, use_container_width=True)


if __name__ == '__main__':
    main()
