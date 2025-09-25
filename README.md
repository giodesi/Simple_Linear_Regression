# Simple Linear Regression Analysis Project

## Project Overview

This project provides a comprehensive implementation of simple linear regression analysis through both a Jupyter notebook for educational exploration and a production-ready web application. The project demonstrates the complete workflow of building, training, evaluating, and deploying linear regression models for predictive analytics on any dataset.

The implementation focuses on accessibility and generalization, enabling users to apply linear regression techniques to their own data without requiring extensive programming knowledge or statistical expertise. The project maintains scientific rigor while presenting complex concepts through intuitive visualizations and clear metrics.

## Quick Links

### Live Application
Access the fully functional web application deployed on Hugging Face Spaces: [**Simple Linear Regression App**](https://huggingface.co/spaces/giodesi/Simple_Linear_Regression)

### Project Resources
The complete implementation includes both educational materials and production-ready code. The Jupyter notebook provides a comprehensive walkthrough of linear regression concepts with detailed explanations and visualizations, available at [`Simple-Linear-Regression.ipynb`](https://github.com/giodesi/Simple_Linear_Regression/blob/v1.0.2/Simple-Linear-Regression.ipynb). The source code for the Streamlit web application can be found in [`app.py`](https://github.com/giodesi/Simple_Linear_Regression/blob/v1.0.2/app.py), which implements the same functionality with an interactive interface suitable for any CSV dataset.

### Version Information
This documentation corresponds to version 1.0.2 of the Simple Linear Regression toolkit. The stable release tag ensures consistent behavior between the documentation and deployed application. Future updates will maintain backward compatibility while extending functionality based on user feedback and requirements.

## Components

### Jupyter Notebook: Simple-Linear-Regression.ipynb

The Jupyter notebook serves as an educational foundation that demonstrates the complete linear regression workflow using Python and scikit-learn. The notebook provides a structured learning experience through hands-on implementation with a fuel consumption dataset, though the techniques generalize to any regression problem.

The notebook begins by establishing the theoretical foundation of simple linear regression before progressing through practical implementation. It loads and explores the FuelConsumptionCo2.csv dataset, which contains model-specific fuel consumption ratings and estimated carbon dioxide emissions for light-duty vehicles in Canada. Through systematic data exploration, the notebook demonstrates how to identify linear relationships between features using statistical summaries and visualizations.

The implementation showcases critical machine learning practices including proper train-test splitting, model training using scikit-learn's LinearRegression, and comprehensive model evaluation. The notebook emphasizes understanding model coefficients and their interpretation, visualizing fitted regression lines, and calculating performance metrics including Mean Absolute Error, Mean Squared Error, Root Mean Squared Error, and R-squared scores. The notebook concludes with comparative analysis, demonstrating how different feature selections impact model performance.

### Streamlit Web Application: app.py

The web application transforms the notebook's static analysis into an interactive platform that accommodates any CSV dataset. Built with Streamlit, the application provides a professional interface for conducting linear regression analysis without writing code.

The application architecture follows a logical workflow divided into five distinct sections. The Data Overview tab provides immediate insight into uploaded datasets through statistical summaries, data type information, and sample visualization. The Feature Selection and Visualization tab enables dynamic exploration of relationships between variables through automated histogram generation and scatter plot matrices with correlation coefficients.

The Model Training tab implements the core regression functionality, allowing users to select any numeric columns as input features and target variables. The interface provides control over train-test split ratios and random state for reproducibility. Upon training, the application displays the regression equation, model coefficients, and dual visualizations showing the fitted line on both training and test data.

The Model Evaluation tab delivers comprehensive performance assessment through multiple metrics and residual analysis. The evaluation includes three critical visualizations: actual versus predicted values to assess prediction accuracy, residual plots to detect heteroscedasticity and non-linearity, and residual distribution histograms to verify normality assumptions. The Predictions tab enables practical application of trained models through both single-value and batch prediction capabilities with real-time visualization.

## Installation and Requirements

The project requires Python 3.7 or higher (3.12.9 suggested) with several scientific computing libraries. To establish the environment, first ensure Python is properly installed on your system. Then install the required dependencies using pip:

```bash
pip install streamlit==1.44.0
pip install pandas==2.2.3
pip install numpy==2.2.4
pip install matplotlib==3.10.6
pip install scikit-learn==1.6.1
```

For Jupyter notebook execution, additionally install:

```bash
pip install jupyter notebook
```

## Usage Instructions

### Running the Jupyter Notebook

Navigate to the project directory and launch Jupyter Notebook through the command line using `jupyter notebook`. Open Simple-Linear-Regression.ipynb from the Jupyter interface and execute cells sequentially using Shift+Enter. The notebook includes detailed markdown explanations accompanying each code cell to guide understanding.

### Running the Web Application

Launch the Streamlit application from the terminal using the command `streamlit run app.py`. The application opens automatically in your default web browser at http://localhost:8501. 

To begin analysis, upload any CSV file containing numeric columns using the sidebar file uploader. The application automatically detects numeric columns suitable for regression analysis. Navigate through the tabs sequentially to explore data, select features, train models, evaluate performance, and generate predictions. The reset button in the sidebar allows you to clear all data and begin a new analysis session.

## Data Requirements

The application accepts CSV files with standard formatting and requires at least two numeric columns for regression analysis. The system automatically handles missing values through appropriate filtering during model training. Column headers should be descriptive as they appear throughout the interface for feature selection and result interpretation.

The application performs best with datasets containing clear linear relationships between variables. While it handles any numeric data, datasets with strong linear correlations produce more accurate and interpretable models. The system supports datasets of varying sizes, though very large files may require additional processing time.

## Technical Architecture

The web application leverages Streamlit's session state management to maintain data persistence across user interactions. This architecture ensures that uploaded data, trained models, and analysis results remain accessible throughout the session without redundant recomputation.

The implementation follows software engineering best practices including modular function design, comprehensive error handling, and clear separation between data processing and visualization logic. The reset functionality implements a key-based system for the file uploader widget, ensuring complete state clearance when users initiate a new analysis session.

Model training utilizes scikit-learn's optimized LinearRegression implementation with automatic handling of data reshaping for single-feature regression. The application implements proper data splitting with configurable test set sizes and random state control for reproducibility. All visualizations utilize matplotlib with consistent styling and informative labeling to maintain professional presentation standards.

## Model Evaluation Methodology

The application implements comprehensive model evaluation through multiple complementary approaches. Performance metrics provide quantitative assessment through Mean Absolute Error for interpretable average deviation, Mean Squared Error for outlier-sensitive evaluation, Root Mean Squared Error for scale-appropriate comparison, and R-squared for variance explanation assessment.

Residual analysis offers deeper diagnostic insight through three visualizations. The actual versus predicted plot reveals systematic bias patterns, the residual plot detects heteroscedasticity and non-linearity, and the residual distribution histogram verifies the normality assumption crucial for statistical inference. These diagnostic tools collectively ensure users understand not only model accuracy but also the validity of underlying assumptions.

## Future Enhancements

The current implementation provides a robust foundation for simple linear regression analysis while maintaining opportunities for expansion. Potential enhancements include support for multiple linear regression with several input features, polynomial regression for non-linear relationships, advanced feature engineering capabilities, and cross-validation for more robust performance estimation.

Additional statistical enhancements could incorporate confidence intervals for predictions, hypothesis testing for coefficient significance, influence diagnostics for outlier detection, and automated assumption checking with corrective recommendations. The interface could expand to include model comparison capabilities, experiment tracking across sessions, and export functionality for models and visualizations.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. This permissive license allows for commercial use, modification, distribution, and private use while requiring only attribution and limiting liability.

## Attribution

The fuel consumption dataset used in the notebook originates from the Government of Canada's Open Data portal, demonstrating the application of linear regression to real-world environmental data. When using this project or its derivatives, please maintain appropriate attribution to both this project and the original data source.

## Support and Documentation

For optimal results, users should ensure their data contains meaningful linear relationships between variables and verify that residual patterns align with linear regression assumptions. The application provides informative error messages for common issues and includes contextual guidance throughout the interface to support users at all expertise levels.