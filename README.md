# Handwritten Digit Recognizer (CNN)

A web-based handwritten digit recognition application that uses a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset** to classify handwritten digits from **0 to 9**. Built with Python, TensorFlow/Keras, and Streamlit.

Draw a digit on the canvas, click **Predict**, and view the predicted digit, confidence score, and probability distribution.

## Features

- **Interactive drawing canvas:** Draw a digit using a mouse or touchscreen.
- **CNN-based classification:** Predicts one of the ten MNIST digit classes (0–9).
- **Confidence score:** Displays the confidence of the predicted class.
- **Class probability chart:** Visualizes the model's output probabilities for all ten digits.
- **Cached model loading:** Loads the trained model once using Streamlit resource caching.

## Project Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface, drawing canvas, image preprocessing, and digit prediction |
| `digit_recognizer_model.keras` | Pre-trained CNN model used for inference |
| `requirements.txt` | Python dependencies |
| `render.yaml` | Render deployment configuration |
| `runtime.txt` | Python runtime specification |
| `.python-version` | Python version configuration |
| `.gitignore` | Files excluded from Git |

> **Note:** The repository currently contains the Streamlit inference application and the saved model. It does not include separate `digit_recognizer.py` training or `predict_custom_image.py` scripts.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| TensorFlow / Keras | Loads and runs the CNN model |
| Streamlit | Web application interface |
| streamlit-drawable-canvas | Interactive drawing canvas |
| NumPy | Image array processing and prediction handling |
| Pillow (PIL) | Image conversion and resizing |

## How It Works

1. **Draw:** The user draws a single digit on the canvas.
2. **Capture:** The canvas provides the drawing as an RGBA image.
3. **Grayscale conversion:** The image is converted to grayscale.
4. **Resize:** The image is resized to **28 × 28 pixels**, matching the MNIST input dimensions.
5. **Normalize:** Pixel values are scaled from 0–255 to 0–1.
6. **Reshape:** The image is reshaped to `(1, 28, 28, 1)` for the CNN input.
7. **Predict:** The trained model returns probabilities for the ten digit classes.
8. **Display:** The app selects the class with the highest probability and displays the predicted digit, confidence, and probability chart.

## CNN Concepts Demonstrated

### Why CNN instead of a plain neural network?

A CNN is designed to process image data. Its convolutional filters learn local visual patterns such as edges, curves, and loops. These features help the network recognize digit shapes. Pooling layers reduce the spatial size of feature maps and computation, while dense layers use the extracted features to classify the image.

A fully connected network treats pixels more independently and can require many more parameters. CNNs are generally more effective for image recognition because they learn spatial patterns.

### Important CNN layers

- **Conv2D:** Applies filters across the image to detect features such as edges, curves, and shapes.
- **BatchNormalization:** Normalizes layer activations to help make training more stable.
- **MaxPooling2D:** Reduces feature-map dimensions while retaining strong feature responses.
- **Dropout:** Randomly disables some neurons during training to help reduce overfitting.
- **Dense:** Combines learned features for classification.
- **Softmax:** Converts the final outputs into probabilities across the ten digit classes.

### Image preprocessing

MNIST images are grayscale images of size **28 × 28**. The application converts the canvas drawing to grayscale, resizes it to the same dimensions, normalizes its pixel values, and adds the batch and channel dimensions expected by the model.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

Replace the URL and folder name with your actual GitHub repository details.

### 2. Create a virtual environment (recommended)

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the application

```bash
streamlit run app.py
```

Open the local address printed in the terminal, usually:

```text
http://localhost:8501
```

Make sure `digit_recognizer_model.keras` remains in the project root because `app.py` loads the model from that location.

## Deployment (Render)

The project includes a `render.yaml` Blueprint configuration for Render.

1. Push the project to a GitHub repository.
2. Sign in to [Render](https://render.com/).
3. Choose **New → Blueprint** and connect your repository.
4. Render reads the settings from `render.yaml`.
5. Review and deploy the web service.
6. Open the public URL after deployment completes.

The configured start command runs Streamlit on Render's assigned port and binds it to `0.0.0.0`.

## Evaluation and Limitations

The application uses a model trained on the MNIST handwritten digit dataset. The app describes the model as achieving approximately **99% test accuracy** on MNIST. This benchmark is measured on the dataset and should not be interpreted as a guarantee of the same accuracy on every user drawing.

Predictions can be less reliable when a drawing differs significantly from MNIST examples—for example, if the digit is off-center, unusually thin, rotated, or contains multiple digits.

## Possible Improvements

- Add image-upload support to recognize digits from photographs.
- Improve preprocessing for off-center, rotated, or differently sized handwriting.
- Add a training notebook or script with reproducible training and evaluation.
- Display the top predicted classes with their probabilities.
- Add data augmentation such as rotation, shifting, and zooming.
- Evaluate performance on handwritten images beyond the MNIST test set.
