import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import io

# Page configuration
st.set_page_config(page_title="Simple Linear Regression App", layout="wide")

# Title and description
st.title("Simple Linear Regression Web Application")
st.markdown("""
This application implements simple linear regression analysis on any CSV dataset.
Upload your data and follow the workflow to build and evaluate a linear regression model.
""")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False


# Reset function
def reset_application():
    """Reset all session state variables to initial values"""
    # Increment the uploader key to force file uploader reset
    st.session_state.uploader_key += 1
    # Clear all other session state variables
    for key in list(st.session_state.keys()):
        if key != 'uploader_key':
            del st.session_state[key]
    st.session_state.df = None
    st.session_state.model = None
    st.session_state.model_trained = False


# Sidebar for file upload and reset
st.sidebar.header("1. Load Data")
uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV file",
    type="csv",
    key=f"file_uploader_{st.session_state.uploader_key}"
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.sidebar.success(f"✅ Data loaded successfully! Shape: {df.shape}")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")

# Reset button in sidebar
st.sidebar.markdown("---")
st.sidebar.header("Application Controls")
if st.sidebar.button("🔄 Reset Application", type="secondary", use_container_width=True):
    reset_application()
    st.rerun()

# Main content area
if st.session_state.df is not None:
    df = st.session_state.df

    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview",
        "📈 Feature Selection & Visualization",
        "🔧 Model Training",
        "📉 Model Evaluation",
        "🎯 Predictions"
    ])

    # Tab 1: Data Overview
    with tab1:
        st.header("Data Overview")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("First 5 Rows")
            st.dataframe(df.head())

        with col2:
            st.subheader("Random Sample (5 rows)")
            st.dataframe(df.sample(min(5, len(df))))

        st.subheader("Statistical Summary")
        st.dataframe(df.describe())

        st.subheader("Data Types")
        st.dataframe(pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum()
        }))

    # Tab 2: Feature Selection and Visualization
    with tab2:
        st.header("Feature Selection and Visualization")

        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            st.error("Need at least 2 numeric columns for regression analysis")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Select Features for Analysis")
                selected_features = st.multiselect(
                    "Choose features to explore:",
                    numeric_cols,
                    default=numeric_cols[:min(4, len(numeric_cols))]
                )

            with col2:
                st.subheader("Select Target Variable (Y)")
                target_variable = st.selectbox(
                    "Choose the target variable to predict:",
                    numeric_cols,
                    index=len(numeric_cols) - 1 if len(numeric_cols) > 0 else 0
                )

            if selected_features:
                # Create subset dataframe
                analysis_df = df[selected_features].copy()

                # Histograms
                st.subheader("Feature Distributions")
                fig, axes = plt.subplots(1, len(selected_features), figsize=(4 * len(selected_features), 4))
                if len(selected_features) == 1:
                    axes = [axes]

                for idx, col in enumerate(selected_features):
                    axes[idx].hist(analysis_df[col].dropna(), bins=20, edgecolor='black')
                    axes[idx].set_title(col)
                    axes[idx].set_xlabel('Value')
                    axes[idx].set_ylabel('Frequency')

                plt.tight_layout()
                st.pyplot(fig)

                # Scatter plots against target
                if target_variable in selected_features:
                    other_features = [f for f in selected_features if f != target_variable]

                    if other_features:
                        st.subheader(f"Scatter Plots: Features vs {target_variable}")

                        fig, axes = plt.subplots(1, len(other_features), figsize=(5 * len(other_features), 4))
                        if len(other_features) == 1:
                            axes = [axes]

                        for idx, feature in enumerate(other_features):
                            axes[idx].scatter(df[feature], df[target_variable], alpha=0.5)
                            axes[idx].set_xlabel(feature)
                            axes[idx].set_ylabel(target_variable)
                            axes[idx].set_title(f"{feature} vs {target_variable}")

                            # Calculate correlation
                            corr = df[[feature, target_variable]].corr().iloc[0, 1]
                            axes[idx].text(0.05, 0.95, f'Corr: {corr:.3f}',
                                           transform=axes[idx].transAxes,
                                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                        plt.tight_layout()
                        st.pyplot(fig)

    # Tab 3: Model Training
    with tab3:
        st.header("Model Training")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            st.error("Need at least 2 numeric columns for regression")
        else:
            col1, col2, col3 = st.columns(3)

            with col1:
                feature_col = st.selectbox(
                    "Select Input Feature (X):",
                    numeric_cols,
                    key="feature_select"
                )

            with col2:
                target_col = st.selectbox(
                    "Select Target Variable (Y):",
                    [col for col in numeric_cols if col != feature_col],
                    key="target_select"
                )

            with col3:
                test_size = st.slider(
                    "Test Set Size (%):",
                    min_value=10,
                    max_value=40,
                    value=20,
                    step=5
                ) / 100

            random_state = st.number_input(
                "Random State (for reproducibility):",
                min_value=0,
                max_value=1000,
                value=42
            )

            if st.button("Train Model", type="primary"):
                # Prepare data
                X = df[feature_col].values
                y = df[target_col].values

                # Remove any NaN values
                mask = ~(np.isnan(X) | np.isnan(y))
                X = X[mask]
                y = y[mask]

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )

                # Train model
                model = LinearRegression()
                model.fit(X_train.reshape(-1, 1), y_train)

                # Store in session state
                st.session_state.model = model
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.feature_col = feature_col
                st.session_state.target_col = target_col
                st.session_state.model_trained = True

                st.success("✅ Model trained successfully!")

            # Display model results if a model has been trained
            if 'model_trained' in st.session_state and st.session_state.model_trained:
                model = st.session_state.model
                X_train = st.session_state.X_train
                X_test = st.session_state.X_test
                y_train = st.session_state.y_train
                y_test = st.session_state.y_test
                feature_col = st.session_state.feature_col
                target_col = st.session_state.target_col

                # Display model parameters
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Coefficient (Slope)", f"{model.coef_[0]:.4f}")
                with col2:
                    st.metric("Intercept", f"{model.intercept_:.4f}")

                st.latex(f"y = {model.intercept_:.4f} + {model.coef_[0]:.4f} \\times x")

                # Visualize the fitted model
                st.subheader("Model Visualization")

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                # Training data plot
                ax1.scatter(X_train, y_train, alpha=0.5, label='Training Data')
                ax1.plot(X_train, model.predict(X_train.reshape(-1, 1)),
                         'r-', label='Fitted Line', linewidth=2)
                ax1.set_xlabel(feature_col)
                ax1.set_ylabel(target_col)
                ax1.set_title('Training Data with Fitted Line')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # Test data plot
                ax2.scatter(X_test, y_test, alpha=0.5, label='Test Data')
                ax2.plot(X_test, model.predict(X_test.reshape(-1, 1)),
                         'r-', label='Model Prediction', linewidth=2)
                ax2.set_xlabel(feature_col)
                ax2.set_ylabel(target_col)
                ax2.set_title('Test Data with Model Prediction')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)

    # Tab 4: Model Evaluation
    with tab4:
        st.header("Model Evaluation")

        if st.session_state.model is not None:
            model = st.session_state.model
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test

            # Make predictions
            y_pred = model.predict(X_test.reshape(-1, 1))

            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            # Display metrics
            st.subheader("Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Mean Absolute Error", f"{mae:.4f}")
            with col2:
                st.metric("Mean Squared Error", f"{mse:.4f}")
            with col3:
                st.metric("Root Mean Squared Error", f"{rmse:.4f}")
            with col4:
                st.metric("R² Score", f"{r2:.4f}")

            # Residual analysis
            st.subheader("Residual Analysis")

            residuals = y_test - y_pred

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Actual vs Predicted
            axes[0].scatter(y_test, y_pred, alpha=0.5)
            axes[0].plot([y_test.min(), y_test.max()],
                         [y_test.min(), y_test.max()],
                         'r--', lw=2)
            axes[0].set_xlabel('Actual Values')
            axes[0].set_ylabel('Predicted Values')
            axes[0].set_title('Actual vs Predicted')
            axes[0].grid(True, alpha=0.3)

            # Residuals vs Predicted
            axes[1].scatter(y_pred, residuals, alpha=0.5)
            axes[1].axhline(y=0, color='r', linestyle='--')
            axes[1].set_xlabel('Predicted Values')
            axes[1].set_ylabel('Residuals')
            axes[1].set_title('Residual Plot')
            axes[1].grid(True, alpha=0.3)

            # Histogram of residuals
            axes[2].hist(residuals, bins=20, edgecolor='black')
            axes[2].set_xlabel('Residuals')
            axes[2].set_ylabel('Frequency')
            axes[2].set_title('Distribution of Residuals')
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

            # Detailed predictions table
            if st.checkbox("Show Detailed Predictions"):
                pred_df = pd.DataFrame({
                    st.session_state.feature_col: X_test,
                    f'Actual {st.session_state.target_col}': y_test,
                    f'Predicted {st.session_state.target_col}': y_pred,
                    'Residual': residuals,
                    'Absolute Error': np.abs(residuals)
                })
                st.dataframe(pred_df)
        else:
            st.info("Please train a model first in the 'Model Training' tab.")

    # Tab 5: Make Predictions
    with tab5:
        st.header("Make New Predictions")

        if st.session_state.model is not None:
            model = st.session_state.model
            feature_col = st.session_state.feature_col
            target_col = st.session_state.target_col

            st.subheader(f"Predict {target_col} based on {feature_col}")

            # Single prediction
            st.write("**Single Value Prediction**")
            input_value = st.number_input(
                f"Enter {feature_col} value:",
                value=float(df[feature_col].mean())
            )

            if st.button("Predict"):
                prediction = model.predict([[input_value]])[0]
                st.success(f"Predicted {target_col}: **{prediction:.4f}**")

            # Batch predictions
            st.write("**Batch Predictions**")
            st.write("Enter multiple values (comma-separated):")
            batch_input = st.text_input("Values:", placeholder="e.g., 2.5, 3.0, 3.5, 4.0")

            if batch_input and st.button("Predict Batch"):
                try:
                    values = [float(x.strip()) for x in batch_input.split(',')]
                    predictions = model.predict(np.array(values).reshape(-1, 1))

                    results_df = pd.DataFrame({
                        feature_col: values,
                        f'Predicted {target_col}': predictions
                    })
                    st.dataframe(results_df)

                    # Visualization
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.scatter(values, predictions, color='green', s=100,
                               label='New Predictions', zorder=5)

                    # Add the regression line
                    x_range = np.linspace(min(values), max(values), 100)
                    y_range = model.predict(x_range.reshape(-1, 1))
                    ax.plot(x_range, y_range, 'r-', label='Model', linewidth=2)

                    ax.set_xlabel(feature_col)
                    ax.set_ylabel(f'Predicted {target_col}')
                    ax.set_title('Batch Predictions')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)

                except ValueError:
                    st.error("Please enter valid numeric values separated by commas.")
        else:
            st.info("Please train a model first in the 'Model Training' tab.")

else:
    # Initial state - no data loaded
    st.info("👈 Please upload a CSV file from the sidebar to begin the analysis.")

    st.markdown("""
    ### Getting Started

    1. **Upload your CSV file** using the file uploader in the sidebar
    2. **Explore your data** in the Data Overview tab
    3. **Select features** and visualize relationships
    4. **Train a model** with your chosen input and target variables
    5. **Evaluate performance** with comprehensive metrics
    6. **Make predictions** on new data

    The application supports any CSV dataset with numeric columns for regression analysis.
    """)

# Footer
st.markdown("---")
st.markdown("**Simple Linear Regression Web App** - Built with Streamlit")
st.markdown("This application follows the workflow of simple linear regression analysis, "
            "allowing you to explore data, train models, and make predictions on any CSV dataset.")