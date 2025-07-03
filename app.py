import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Utility function for plot download
def fig_to_bytes(fig):
    """Convert matplotlib figure to bytes for download"""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf

# Page configuration
st.set_page_config(
    page_title="Data Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #f5f5f5;
        background-image: linear-gradient(315deg, #f5f5f5 0%, #e5e5e5 74%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #2c3e50 !important;
        color: white !important;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] .st-cq {
        color: white !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50 !important;
    }
    
    /* Buttons */
    .st-b7 {
        background-color: #3498db !important;
        color: white !important;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Cards */
    .st-eb {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Title section
with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📊 Data Analysis Application")
        st.subheader("Explore datasets with Streamlit and Seaborn")
    with col2:
        st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=150)

# Sidebar for dataset selection and controls
with st.sidebar:
    st.header("Dataset Selection")
    dataset_name = st.selectbox(
        "Choose a dataset", ("Iris", "Titanic", "Tips")
    )
    
    with st.expander("🎛️ Visualization Controls", expanded=True):
        # Color theme selector
        theme = st.selectbox("Color Theme", ["Default", "Dark", "Pastel", "Bright"])
        
        # Plot size control
        plot_width = st.slider("Plot Width", 400, 1000, 600)
        plot_height = st.slider("Plot Height", 300, 800, 400)
        
        # Animation toggle
        animate = st.checkbox("Enable animations", True)

# Load the selected dataset
def load_data(dataset):
    if dataset == "Iris":
        data = sns.load_dataset("iris")
    elif dataset == "Titanic":
        data = sns.load_dataset("titanic")
    elif dataset == "Tips":
        data = sns.load_dataset("tips")
    return data

data = load_data(dataset_name)

# Main content
with st.container():
    st.write("---")
    
    # Dataset overview in a card-like container
    with st.expander("📁 Dataset Overview", expanded=True):
        st.write(f"Displaying the **{dataset_name}** dataset:")
        st.dataframe(data.style.background_gradient(cmap='Blues'), height=300)
        
        # Display summary statistics
        st.subheader("📋 Summary Statistics")
        st.write(data.describe())
    
    st.write("---")
    
    # Visualizations in tabs
    tab1, tab2, tab3 = st.tabs(["📈 Basic Visualizations", "📊 Advanced Analysis", "🔍 Interactive Explorer"])
    
    with tab1:
        st.header("Basic Visualizations")
        
        if dataset_name == "Iris":
            with st.expander("🌺 Iris Scatter Plot", expanded=True):
                fig, ax = plt.subplots(figsize=(plot_width/100, plot_height/100))
                scatter = sns.scatterplot(
                    x="sepal_length", 
                    y="sepal_width", 
                    hue="species", 
                    data=data, 
                    ax=ax,
                    s=100 if animate else 50,
                    alpha=0.7
                )
                plt.title("Iris Dataset - Sepal Dimensions", fontsize=14)
                plt.xlabel("Sepal Length (cm)", fontsize=12)
                plt.ylabel("Sepal Width (cm)", fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.3)
                st.pyplot(fig)
                
                # Add download button
                st.download_button(
                    label="Download Plot",
                    data=fig_to_bytes(fig),
                    file_name="iris_scatter.png",
                    mime="image/png"
                )
        
        elif dataset_name == "Titanic":
            with st.expander("🚢 Titanic Survival Count", expanded=True):
                fig, ax = plt.subplots(figsize=(plot_width/100, plot_height/100))
                sns.countplot(
                    x="survived", 
                    hue="class", 
                    data=data, 
                    ax=ax,
                    palette="viridis" if theme == "Default" else None
                )
                plt.title("Titanic Survival by Class", fontsize=14)
                plt.xlabel("Survived", fontsize=12)
                plt.ylabel("Count", fontsize=12)
                plt.legend(title="Class")
                st.pyplot(fig)
                
                st.download_button(
                    label="Download Plot",
                    data=fig_to_bytes(fig),
                    file_name="titanic_survival.png",
                    mime="image/png"
                )
        
        elif dataset_name == "Tips":
            with st.expander("🍽️ Tips Analysis", expanded=True):
                fig, ax = plt.subplots(figsize=(plot_width/100, plot_height/100))
                sns.scatterplot(
                    x="total_bill", 
                    y="tip", 
                    hue="sex", 
                    data=data, 
                    ax=ax,
                    s=80 if animate else 40,
                    alpha=0.7
                )
                plt.title("Tips vs Total Bill", fontsize=14)
                plt.xlabel("Total Bill ($)", fontsize=12)
                plt.ylabel("Tip ($)", fontsize=12)
                st.pyplot(fig)
                
                st.download_button(
                    label="Download Plot",
                    data=fig_to_bytes(fig),
                    file_name="tips_scatter.png",
                    mime="image/png"
                )
    
    with tab2:
        st.header("Advanced Analysis")
        
        # Correlation Heatmap
        with st.expander("🔥 Correlation Heatmap", expanded=True):
            if dataset_name in ["Iris", "Titanic", "Tips"]:
                if dataset_name == "Titanic":
                    numeric_data = data.select_dtypes(include=np.number)
                else:
                    numeric_data = data.select_dtypes(include=np.number)

                if not numeric_data.empty:
                    corr = numeric_data.corr()
                    fig, ax = plt.subplots(figsize=(plot_width/100, plot_height/100))
                    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                    plt.title("Correlation Matrix", fontsize=14)
                    st.pyplot(fig)
                    
                    st.download_button(
                        label="Download Heatmap",
                        data=fig_to_bytes(fig),
                        file_name="correlation_heatmap.png",
                        mime="image/png"
                    )
                else:
                    st.write("No numeric features to compute correlation for this dataset.")
        
        # Pair Plots (for Iris)
        if dataset_name == "Iris":
            with st.expander("🔄 Pair Plots", expanded=True):
                fig = sns.pairplot(data, hue="species", height=2)
                st.pyplot(fig)
                
                st.download_button(
                    label="Download Pair Plot",
                    data=fig_to_bytes(fig.fig),
                    file_name="iris_pairplot.png",
                    mime="image/png"
                )
        
        # Distribution Plots
        with st.expander("📊 Distribution Plots", expanded=False):
            if dataset_name in ["Iris", "Titanic", "Tips"]:
                selected_col = st.selectbox(
                    "Select column for distribution:",
                    options=data.select_dtypes(include=np.number).columns
                )
                
                if selected_col:
                    fig, ax = plt.subplots(figsize=(plot_width/100, plot_height/100))
                    sns.histplot(data[selected_col], kde=True, ax=ax)
                    plt.title(f"Distribution of {selected_col}", fontsize=14)
                    plt.xlabel(selected_col, fontsize=12)
                    plt.ylabel("Frequency", fontsize=12)
                    st.pyplot(fig)
                    
                    st.download_button(
                        label="Download Distribution Plot",
                        data=fig_to_bytes(fig),
                        file_name=f"{selected_col}_distribution.png",
                        mime="image/png"
                    )
        
        # Feature interaction for Tips
        if dataset_name == "Tips":
            with st.expander("🧩 Feature Interaction", expanded=True):
                data["bill_per_person"] = data["total_bill"] / data["size"]
                
                fig, ax = plt.subplots(figsize=(plot_width/100, plot_height/100))
                sns.scatterplot(
                    x="bill_per_person", 
                    y="tip", 
                    hue="sex", 
                    data=data, 
                    ax=ax,
                    s=80 if animate else 40,
                    alpha=0.7
                )
                plt.title("Tip vs. Bill per Person by Gender", fontsize=14)
                plt.xlabel("Bill per Person ($)", fontsize=12)
                plt.ylabel("Tip ($)", fontsize=12)
                st.pyplot(fig)
                
                st.write(
                    "Created 'bill_per_person' by dividing 'total_bill' by 'size' to analyze spending per person."
                )
                
                st.download_button(
                    label="Download Interaction Plot",
                    data=fig_to_bytes(fig),
                    file_name="tips_interaction.png",
                    mime="image/png"
                )
    
    with tab3:
        st.header("Interactive Explorer")
        
        # Interactive Scatter Plot
        with st.expander("🖱️ Interactive Scatter Plot", expanded=True):
            if dataset_name in ["Iris", "Titanic", "Tips"]:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    x_col = st.selectbox(
                        "Select X-axis", 
                        options=data.columns, 
                        index=0
                    )
                
                with col2:
                    y_col = st.selectbox(
                        "Select Y-axis", 
                        options=data.columns, 
                        index=1
                    )
                
                with col3:
                    hue_col = st.selectbox(
                        "Select Hue (color)", 
                        options=["None"] + list(data.columns), 
                        index=0
                    )

                if x_col and y_col:
                    fig, ax = plt.subplots(figsize=(plot_width/100, plot_height/100))
                    sns.scatterplot(
                        x=x_col, 
                        y=y_col, 
                        hue=None if hue_col == "None" else hue_col, 
                        data=data, 
                        ax=ax,
                        s=80 if animate else 40,
                        alpha=0.7
                    )
                    plt.title(f"{y_col} vs {x_col}", fontsize=14)
                    plt.xlabel(x_col, fontsize=12)
                    plt.ylabel(y_col, fontsize=12)
                    st.pyplot(fig)
                    
                    st.download_button(
                        label="Download Custom Plot",
                        data=fig_to_bytes(fig),
                        file_name="custom_scatter.png",
                        mime="image/png"
                    )

# Footer
st.markdown("---")
footer = """<div style="text-align: center; padding: 10px; background-color: #2c3e50; color: white; border-radius: 5px;">
    <p>Created with ❤️ by zakihandsome | Powered by Streamlit</p>
</div>"""
st.markdown(footer, unsafe_allow_html=True)

# Hide Streamlit default menu and footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)