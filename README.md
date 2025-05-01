# Spine-Related Disorders Classification System

This project is a machine learning-based web application designed to classify spine-related disorders. It uses various machine learning models, including CatBoost, AdaBoost, and Random Forest, to predict whether a patient has a normal spine, disk hernia, or spondylolisthesis based on input features.

## Features

- **Binary Classification**: Classifies spine conditions as either "Normal" or "Abnormal."
- **Multi-Class Classification**: Classifies spine conditions into three categories: "Normal," "Disk Hernia," and "Spondylolisthesis."
- **Model Comparison**: Compares the performance of different models before and after applying SMOTE (Synthetic Minority Oversampling Technique).
- **Interactive Web Interface**: Allows users to input patient data and select a model for prediction.

## Technologies Used

- **Backend**: Flask
- **Frontend**: HTML, CSS, JavaScript
- **Machine Learning**: Scikit-learn, CatBoost, Imbalanced-learn
- **Visualization**: Matplotlib, Seaborn

## Dataset

The project uses two datasets:

1. `column_2C_weka.csv` for binary classification.
2. `column_3C_weka.csv` for multi-class classification.

## How to Run the Project

### Prerequisites

1. Install Python (version 3.8 or higher).
2. Install the required Python libraries:

   ```bash
   pip install -r requirements.txt
   ```

### Steps to Run

1. **Clone the Repository**:

   ```bash
   git clone <repository-url>
   cd spine-related-disorders-classification-ml-model
   ```

2. **Prepare the Datasets**:

   - Place the datasets (`column_2C_weka.csv` and `column_3C_weka.csv`) in the appropriate directory.

3. **Train and Save Models**:

   - Open the Jupyter Notebook `spine-related disorders classification.ipynb`.
   - Run all cells to train the models and save them in the `models` directory.

4. **Set Up Environment Variables**:

   - Create a .env file in the project root with the following content:

     ```bash
      MODEL_DIR=models
      DEBUG=True
     ```

5. **Start the Flask Server**:

   - Run the Flask application:

     ```bash
     python run.py
     ```

6. **Access the Web Application**:

   - Open your browser and navigate to `http://127.0.0.1:5000`.

7. **Use the Application**:

   - Select a model from the available options.
   - Input patient data (e.g., Pelvic Incidence, Pelvic Tilt, etc.).
   - Click "Submit" to get the prediction.

8. **Use the API**:

   - You can also use the `/api/predict` endpoint for programmatic predictions. Example:

   ```bash
         curl -X POST http://127.0.0.1:5000/api/predict \
      -H "Content-Type: application/json" \
      -d '{
         "model": "CatBoost",
         "pelvic_incidence": 45.0,
         "pelvic_tilt": 10.0,
         "lumbar_lordosis_angle": 35.0,
         "sacral_slope": 30.0,
         "pelvic_radius": 120.0,
         "degree_spondylolisthesis": 5.0
      }'
   ```

## Running the Project with Docker

1. Build the Docker image:

   ```bash
   docker build -t spine-classification-app .
   ```

2. Run the Docker container:

   ```bash
   docker run -p 5000:5000 --env-file .env spine-classification-app
   ```

3. Access the application in your browser at `http://127.0.0.1:5000`.

## Project Structure

```bash
spine-related-disorders-classification-ml-model/
├── app/
│   ├── __init__.py             # Flask app factory
│   ├── app.py                  # Main application logic
│   ├── models.py               # Model loading and prediction logic
│   ├── routes.py               # Route definitions
│   ├── templates/
│   │   ├── index.html          # Main page
│   │   ├── result.html         # Result page
│   ├── static/
│   │   ├── styles.css          # CSS for styling
├── data/                       # Directory for datasets
│   ├── column_2C_weka.csv
│   ├── column_3C_weka.csv
│   ├── resampled_column_2C_weka.csv
│   ├── resampled_column_3C_weka.csv
├── models/                     # Directory for saved models
│   ├── CatBoost_multi.pkl
│   ├── AdaBoost_multi.pkl
│   ├── Random Forest_multi.pkl
├── tests/                      # Unit tests
│   ├── test_app.py             # Tests for app.py
│   ├── test_models.py          # Tests for models.py
│   ├── test_routes.py          # Tests for routes.py
├── notebooks/
│   ├── spine-related-disorders-classification.ipynb  # Jupyter Notebook for training models
├── requirements.txt            # Python dependencies
├── run.py                      # Entry point for running the app
├── .env                        # Environment variables
└── README.md                   # Project documentation
```

## Requirements

The required Python libraries are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

## Example Input

- **Pelvic Incidence**: 39.05
- **Pelvic Tilt Numeric**: 10.06
- **Lumbar Lordosis Angle**: 25.01
- **Sacral Slope**: 28.99
- **Pelvic Radius**: 114.40
- **Degree Spondylolisthesis**: 4.56

## Example Output

- **Selected Model**: AdaBoost
- **Predicted Class**: Disk Hernia

## License

This project is licensed under the MIT License.

## Acknowledgments

- The datasets used in this project are publicly available and were sourced from the UCI Machine Learning Repository.
