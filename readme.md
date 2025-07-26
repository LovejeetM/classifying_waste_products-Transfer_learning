# Waste Classification using Transfer Learning with VGG16

This project demonstrates the application of transfer learning for a binary image classification task. The goal is to classify images of waste as either **Organic** ('O') or **Recyclable** ('R'). Two distinct transfer learning techniques are implemented and compared using the VGG16 architecture pre-trained on the ImageNet dataset:

1.  **Feature Extraction**: Using the pre-trained model as a fixed feature extractor.
2.  **Fine-Tuning**: Unfreezing and retraining the top layers of the pre-trained model.

The entire workflow, from data acquisition and preparation to model training, evaluation, and visualization, is implemented using TensorFlow and Keras.

***

## Dependencies

The project relies on several key Python libraries. The main requirements are:

* `tensorflow` 
* `numpy`
* `matplotlib`
* `scikit-learn`
* `requests`
* `tqdm`

***

## Project Workflow

### 1. Data Acquisition and Preparation

* **Dataset**: The project utilizes the "O-vs-R" dataset, which consists of 1200 images. The data is downloaded from a public cloud storage URL and extracted programmatically. After extraction, the data is organized into the following structure:

    o-vs-r-split/
    ├── train/
    │   ├── O/
    │   │   └── ... (organic images)
    │   └── R/
    │       └── ... (recyclable images)
    └── test/
        ├── O/
        │   └── ... (organic images)
        └── R/
            └── ... (recyclable images)

* **Data Generators**: `tensorflow.keras.preprocessing.image.ImageDataGenerator` is used to create data pipelines for the training, validation, and test sets. This approach efficiently loads images from disk in batches. The training data (800 images) is further split, with 80% used for training and 20% for validation.
* **Preprocessing**: All images are resized to $150 \times 150$ pixels and pixel values are rescaled from the `[0, 255]` range to the `[0, 1]` range by dividing by $255.0$.
* **Data Augmentation**: To improve model generalization and combat overfitting, the training dataset is augmented on-the-fly. The augmentations include:
    * Random horizontal flips (`horizontal_flip=True`)
    * Random horizontal shifts (`width_shift_range=0.1`)
    * Random vertical shifts (`height_shift_range=0.1`)

***

## Modeling

The core of this project is the application of the VGG16 model. Two strategies are explored.

### 2. Approach 1: VGG16 as a Feature Extractor

In this approach, the **VGG16** model is used as a fixed feature extractor. The convolutional base, pre-trained on ImageNet, is effective at identifying generic visual features.

* **Model Architecture**:
    1.  The VGG16 model is instantiated without its top classification layer (`include_top=False`).
    2.  All layers of the VGG16 base are **frozen** by setting `layer.trainable = False`. This prevents their 14.7 million parameters from being updated during training.
    3.  A new, custom classifier is added on top. This classifier consists of:
        * A `Flatten` layer to convert the 2D feature maps from the VGG16 base into a 1D vector.
        * Two `Dense` layers with 512 units each and **ReLU** activation.
        * `Dropout` layers with a rate of 0.3 after each dense layer for regularization.
        * A final `Dense` output layer with a single unit and a **sigmoid** activation function for binary classification.
    4.  This results in a model with over 19 million total parameters, but only ~4.4 million are trainable (the weights of the custom classifier).

* **Compilation and Training**: The model is compiled with `binary_crossentropy` loss, the **RMSprop** optimizer (learning rate $1e-4$), and is trained for 10 epochs.

### 3. Approach 2: Fine-Tuning the VGG16 Model

This approach builds upon the first by fine-tuning the top layers of the VGG16 base, allowing the model to learn features more specific to the waste dataset.

* **Model Architecture**:
    1.  The VGG16 base model is instantiated in the same way.
    2.  Instead of freezing all layers, only the initial convolutional blocks are frozen. The top layers, starting from the `block5_conv3` layer, are set to be **trainable**. This allows the model to adjust the more specialized, high-level features while keeping the generic, low-level features fixed.
    3.  The same custom classifier from the feature extraction approach is added on top.
    4.  By unfreezing the last convolutional block, the number of trainable parameters increases, allowing for more granular learning.

* **Compilation and Training**: The model is re-compiled with the same loss and optimizer settings and trained for an additional 10 epochs.

### 4. Training Callbacks

During both training processes, a set of Keras callbacks are used to manage the training loop effectively:
* **`ModelCheckpoint`**: Saves the best version of the model to a file (`.keras`) based on the minimum validation loss (`val_loss`). This ensures that the final model is the one that performed best on unseen validation data, regardless of where training stopped.
* **`EarlyStopping`**: Halts training if `val_loss` does not improve by a minimum delta of 0.01 for a patience of 4 epochs. This is a crucial technique to prevent overfitting.
* **`LearningRateScheduler`**: A custom learning rate schedule implements an exponential decay ($lrate = initial\_lrate \times e^{-k \times epoch}$). This helps the model to converge more efficiently by taking larger steps at the beginning of training and smaller, more precise steps as it approaches a minimum.

***

## Results and Evaluation

The performance of both models—the feature extraction model and the fine-tuned model—is evaluated on the unseen test set.

### Classification Performance

**Feature Extraction Model Results:**

                  precision    recall  f1-score   support

               O       0.79      0.82      0.80        50
               R       0.81      0.78      0.80        50

        accuracy                           0.80       100
       macro avg       0.80      0.80      0.80       100
    weighted avg       0.80      0.80      0.80       100

**Fine-Tuned Model Results:**

                  precision    recall  f1-score   support

               O       0.79      0.84      0.82        50
               R       0.83      0.78      0.80        50

        accuracy                           0.81       100
       macro avg       0.81      0.81      0.81       100
    weighted avg       0.81      0.81      0.81       100

### Analysis

* The **feature extraction** model achieved a solid baseline accuracy of **80%**. This demonstrates the power of the pre-trained VGG16 features even without any modification.
* The **fine-tuned** model showed a slight improvement, reaching an accuracy of **81%**. While the overall accuracy gain is marginal, the F1-score for the 'O' class improved from 0.80 to 0.82, indicating a better balance of precision and recall for that class.
* The training and validation **loss and accuracy curves** plotted for both experiments visually confirmed that the models were learning effectively without significant overfitting.

***

## Conclusion and Future Work

This project successfully demonstrates a complete pipeline for a deep learning-based image classification task. It confirms that **transfer learning** is a highly effective strategy for problems with limited datasets.
