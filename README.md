# Image Classifier Project (PyTorch)

This project is a command-line image classifier built with **PyTorch** and **Torchvision**.  
It allows users to train a deep neural network on a dataset of images and then use the trained model to predict the class of new images.

The project supports **VGG16 and VGG19**, GPU acceleration, customizable hyperparameters, and top-K predictions. It was developed as part of the AI Programming with Python Nanodegree by Udacity.

---

## Features

- Train a neural network using transfer learning
- Choose between **VGG16** and **VGG19**
- Adjustable learning rate, hidden units, and epochs
- GPU support for training and inference
- Save and load trained model checkpoints
- Predict image classes with probabilities
- Display human-readable class names

---

## Dataset

This project uses the [102 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html). It consists of 102 different species of flowers.

You can download the dataset using the following command:

```bash
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
tar -xzf flower_data.tar.gz
```

This will create a `flowers` directory containing the images, split into `train`, `valid`, and `test` sets.

---

## Installation

To run this project, you'll need Python 3 and the following libraries:

- PyTorch
- Torchvision
- NumPy
- Pillow
- Matplotlib
- Seaborn

You can install them using pip:

```bash
pip install torch torchvision numpy pillow matplotlib seaborn
```

---

## Project Structure

```bash
.
├── train.py
├── predict.py
├── cat_to_name.json
├── README.md
├── flowers/
│   ├── train/
│   ├── valid/
│   └── test/
└── checkpoint.pth (created after training)
```

## Training the Model

The `train.py` script trains a new classifier on the flower dataset. Here's an example of how to run it:

```bash
python train.py flowers \
  --arch vgg16 \
  --learning_rate 0.001 \
  --hidden_units 1024 \
  --epochs 3 \
  --gpu
```

### Training Arguments

- `data_directory`: (Required) Path to the folder with images (e.g., `flowers/`).
- `--save_dir`: Directory to save checkpoints (default: current directory).
- `--arch`: Model architecture, `vgg16` or `vgg19` (default: `vgg16`).
- `--learning_rate`: Learning rate for the optimizer (default: `0.001`).
- `--hidden_units`: Number of hidden units in the classifier (default: `1024`).
- `--epochs`: Number of training epochs (default: `3`).
- `--gpu`: Use GPU for training if available.

The script will save the trained model as `checkpoint.pth` in the current directory.

## Making Predictions

The `predict.py` script uses a trained model to predict the class of an image.

```bash
python predict.py ./flowers/test/20/image_04910.jpg checkpoint.pth \
  --top_k 5 \
  --category_names cat_to_name.json \
  --gpu
```

### Prediction Arguments

- `path_to_image`: (Required) Path to the input image.
- `path_to_checkpoint`: (Required) Path to the model checkpoint.
- `--top_k`: Return top K most likely classes (default: `5`).
- `--category_names`: Path to a JSON file mapping categories to real names.
- `--gpu`: Use GPU for inference if available.

### Example Output

```
Prediction Results:
  geranium: 0.5135
  pelargonium: 0.2410
  petunia: 0.1120
  morning glory: 0.0801
  love in the mist: 0.0185
```

## Model Architecture

The model uses a pre-trained VGG16/VGG19 network from `torchvision.models` as a feature extractor. The original classifier of the VGG16/VGG19 model is replaced with a new feed-forward classifier with the following structure:

- **Input:** 25088 features from the VGG16/VGG19 convolutional base.
- **Hidden Layer:** 1024 units with ReLU activation.
- **Output Layer:** 102 units (one for each flower category) with LogSoftmax activation.

During training, the parameters of the pre-trained VGG16/VGG19 model are frozen, and only the new classifier is trained.

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pillow (PIL)