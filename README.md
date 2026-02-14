# ML_pixel_regression_model

# CNN Pixel Regression Model

A deep learning project that uses Convolutional Neural Networks (CNN) to predict pixel coordinates in grayscale images. This model performs regression to locate specific positions within 50x50 pixel images.

## 📋 Project Overview

This project implements a CNN-based regression model that:
- Takes 50x50 grayscale images as input
- Predicts (x, y) coordinates of pixel locations
- Uses deep learning for spatial coordinate regression
- Integrates with Weights & Biases (wandb) for experiment tracking

## 🎯 Model Architecture

The CNN model consists of:
- Convolutional layers for feature extraction from images
- Pooling layers for dimensionality reduction
- Dense layers for coordinate regression
- Output layer predicting 2D coordinates (x, y)

## 🔧 Requirements

Install the required dependencies:

```bash
pip install tensorflow matplotlib numpy wandb
```

### Main Dependencies:
- **TensorFlow**: 2.20.0 (Deep learning framework)
- **Keras**: 3.13.2 (High-level neural networks API)
- **NumPy**: 2.4.2 (Numerical computing)
- **Matplotlib**: 3.10.8 (Visualization)
- **Weights & Biases**: 0.25.0 (Experiment tracking)

## 🚀 Usage

### 1. Training the Model

Run the Jupyter notebook `ML_Updated.ipynb` to:
- Generate or load training data (50x50 images with coordinate labels)
- Build and compile the CNN model
- Train the model with wandb tracking
- Evaluate performance on test data

### 2. Making Predictions

```python
# Load the trained model
from tensorflow import keras
model = keras.models.load_model('cnn_pixel_regression_model.h5')

# Make predictions on new images
predictions = model.predict(X_new)
```

### 3. Visualizing Results

The notebook includes visualization code to:
- Display original images
- Plot ground truth coordinates (green)
- Overlay predicted coordinates (red)
- Compare model predictions against actual values

## 📊 Model Performance

**Test Set Metrics:**
- Mean Squared Error (MSE): 47.34
- Mean Absolute Error (MAE): 4.58

These metrics indicate the average pixel distance between predicted and actual coordinates.

## 📁 Project Structure

```
.
├── ML_Updated.ipynb              # Main notebook with complete workflow
├── cnn_pixel_regression_model.h5 # Saved trained model
└── README.md                      # Project documentation
```

## 🔍 Key Features

- **Data Generation**: Synthetic dataset of 50x50 images with labeled coordinates
- **CNN Architecture**: Custom convolutional neural network for regression
- **Experiment Tracking**: Integration with wandb for monitoring training
- **Visualization**: Interactive plots showing predictions vs ground truth
- **Model Persistence**: Save and load trained models for inference

## 📈 Training Process

The model training includes:
1. Data preprocessing and normalization
2. Train/validation/test split
3. CNN model compilation with appropriate loss function
4. Training with early stopping and checkpointing
5. Performance evaluation on held-out test set
6. Visualization of predictions

## 🎨 Visualization Examples

The notebook generates comparison plots showing:
- Input grayscale images
- Green markers: Ground truth coordinates
- Red markers: Model predictions
- Coordinate labels for easy comparison

## 🔄 Model Output

The model outputs 2D coordinates in the range [0, 49] for each axis:
- **x-coordinate**: Horizontal pixel position
- **y-coordinate**: Vertical pixel position

## 💾 Saved Model

The trained model is saved as `cnn_pixel_regression_model.h5` in HDF5 format and can be loaded for:
- Making predictions on new images
- Fine-tuning on additional data
- Transfer learning to related tasks

## 🛠️ Development Environment

- **Python Version**: 3.13.7
- **Virtual Environment**: .venv
- **Platform**: Windows (compatible with Linux/macOS)

## 📝 Notes

- The model uses MSE (Mean Squared Error) as the primary loss function
- MAE (Mean Absolute Error) is tracked as an additional metric
- Images should be normalized to [0, 1] range before prediction
- The model expects input shape: (batch_size, 50, 50, 1)

## 🤝 Contributing

To improve this project:
1. Experiment with different CNN architectures
2. Try data augmentation techniques
3. Implement additional evaluation metrics
4. Test on real-world image datasets

## 📧 Support

For questions or issues, please refer to the notebook documentation or create an issue in the project repository.

---

**Last Updated**: February 2026
