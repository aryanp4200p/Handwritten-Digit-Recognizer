# Handwritten Digit Recognizer (CNN)

A convolutional neural network that classifies handwritten digits (0-9)
using the MNIST dataset, built with TensorFlow/Keras.

## Files

| File | Purpose |
|---|---|
| `digit_recognizer.py` | Loads MNIST, preprocesses data, builds/trains/evaluates the CNN, saves the model and all plots |
| `predict_custom_image.py` | Loads the saved model and predicts the digit in *your own* handwritten image |
| `requirements.txt` | Python dependencies |

## How to run

### Option A — Google Colab (recommended, free GPU, zero setup)
1. Go to [colab.research.google.com](https://colab.research.google.com), new notebook.
2. Paste the contents of `digit_recognizer.py` into a cell and run it (TensorFlow is preinstalled).
3. To test your own handwriting: upload an image via the folder icon on the left, then run `predict_custom_image.py` in a new cell, e.g. `!python predict_custom_image.py my_digit.png`.

### Option B — Locally
```bash
pip install -r requirements.txt
python digit_recognizer.py                 # trains and evaluates the model
python predict_custom_image.py my_digit.png  # test on your own photo
```

Training takes about 2-5 minutes on CPU, under a minute on GPU, and typically reaches **~99% test accuracy**.

## What the project demonstrates

1. **Data preprocessing** — normalization (pixel values scaled to 0-1), reshaping for CNN input, one-hot encoding of labels.
2. **CNN architecture** — three convolutional blocks (Conv2D → BatchNorm → MaxPooling) that progressively learn edges → strokes → digit shapes, followed by a dense classifier head with Dropout for regularization.
3. **Training** — Adam optimizer, categorical cross-entropy loss, EarlyStopping and ReduceLROnPlateau callbacks to avoid overfitting and auto-tune the learning rate.
4. **Evaluation** — test accuracy/loss, full classification report (precision/recall/F1 per digit), confusion matrix, and a look at misclassified examples.
5. **Real-world testing** — a separate script to classify a photo of your own handwriting, including grayscale conversion and color inversion to match MNIST's white-on-black format.

## Viva / presentation talking points

**Why CNN instead of a plain neural network?**
CNNs use convolutional filters that scan the image and detect local patterns (edges, curves, loops) regardless of where they appear in the image. A plain dense network would need to learn every pixel position independently, using far more parameters and generalizing worse to slight shifts or variations in handwriting.

**What does each layer do?**
- *Conv2D*: slides small filters (e.g. 3x3) across the image to detect features like edges or curves.
- *BatchNormalization*: normalizes activations between layers, speeding up and stabilizing training.
- *MaxPooling2D*: downsamples the feature map, keeping the strongest signals and reducing computation.
- *Dropout*: randomly disables neurons during training to prevent overfitting.
- *Dense (Softmax)*: the final layer outputs a probability for each of the 10 digit classes.

**Why one-hot encode the labels?**
Because the output layer produces a probability distribution over 10 classes; one-hot encoding turns a label like `3` into `[0,0,0,1,0,0,0,0,0,0]` so it can be compared to that distribution via cross-entropy loss.

**How do you know the model isn't just memorizing?**
The validation split (held out during training) and separate test set (never seen during training) both show comparable accuracy to training accuracy — if the model were overfitting, validation/test accuracy would lag well behind training accuracy. Dropout and EarlyStopping further guard against this.

**What's the confusion matrix for?**
It shows which digits get confused with each other (e.g. 4 vs 9, or 3 vs 5), which is more informative than a single accuracy number.

**Possible extensions to mention if asked "how would you improve it?"**
- Data augmentation (rotation, shifting, zoom) to make the model robust to messier handwriting.
- A deeper architecture or transfer learning.
- Deploying it as a live web app (e.g. a drawable canvas with Streamlit or Flask) where users draw a digit and get instant predictions.

## Putting it on GitHub

```bash
cd your-project-folder
git init
git add .
git commit -m "Handwritten digit recognizer CNN"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```
(Create the empty repo on github.com first — "New repository", no README/gitignore, so it doesn't conflict with what you're pushing.)

Make sure `digit_recognizer_model.keras` is committed too (needed for the live demo below) — a MNIST CNN model file is typically only a few MB, well under GitHub's limits.

## Hosting a live demo (Streamlit Community Cloud — free)

`app.py` is a Streamlit web app: it shows a canvas you can draw a digit on and get an instant prediction from your trained model.

1. Run `python digit_recognizer.py` once locally/in Colab to produce `digit_recognizer_model.keras`, and make sure it's in your repo.
2. Push everything to GitHub (including `app.py` and the updated `requirements.txt`) as above.
3. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, click **New app**.
4. Pick your repo, branch `main`, and set the main file path to `app.py`.
5. Click **Deploy**. In a minute or two you'll get a public URL like `https://your-app-name.streamlit.app` that anyone can open and draw digits on.

This live link is what you'd put on your resume/LinkedIn/portfolio — much stronger than a plain code repo.

**To test locally before deploying:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Skills demonstrated (for your resume)
Python · NumPy · TensorFlow/Keras · Convolutional Neural Networks · Image Preprocessing · Model Training & Evaluation · Data Visualization (Matplotlib/Seaborn) · Image Classification
