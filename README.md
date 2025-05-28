You got it! Here's the `README.md` for your AbtinCNN library, written in English:

---

## AbtinCNN: A Custom CNN Library in NumPy

This repository contains a collection of Python modules implemented with NumPy, aiming to build a custom and powerful **Convolutional Neural Network (CNN)** library. This library focuses on innovative approaches in image processing, content-aware weighting, adaptive activation functions, and dynamic loss optimization.

---

### Key Features

AbtinCNN is composed of several advanced components, each designed to enhance a specific aspect of CNN processing and training:

1.  **Advanced Image Preprocessing**:
    * **RGB to HSV Conversion**: Transforms images from the RGB color space to HSV (Hue, Saturation, Value), which is beneficial for separating and analyzing color characteristics more akin to human perception.
    * **Hue Quantization**: Divides the Hue component into a specified number (default 9) of bins or categories, simplifying color processing and preparing data for classification.
    * **Layer Splitting**: Converts a labeled image (e.g., the quantized Hue output) into a set of one-hot encoded layers, suitable for the input of CNN convolutional layers.
    * **Dynamic Padding (`CNNPadder`)**: Adds custom padding (borders) to input images. This feature helps control the spatial dimensions of convolutional layer outputs and improves feature extraction from image edges.

2.  **Intelligent Weighting and Activation**:
    * **HSV Resolution Weighting (`HSVResolutionWeight`)**: An advanced approach to calculate pixel-wise weights based on HSV color components (H, S, V) and image resolution (edges). These weights are combined using learnable `alphas` coefficients, allowing the model to be more sensitive to specific visual features.
    * **Dynamic Bias Layer (`dynamic_bias_layer`)**: Adds bias values to feature maps, a fundamental step in neural layers to shift neuron activation thresholds.
    * **Edge-Differentiating Activation Functions (`ActivationFunction`)**: Intelligently applies two different activation functions (GELU for edge regions and Swish for central image regions). This approach allows the model to react to edges and details with different sensitivity compared to the background.

3.  **Advanced Training System**:
    * **Dynamic Loss Trainer (`DynamicLossTrainer`)**: An innovative training system capable of simultaneously training model weights and learning the optimal combination of two loss functions (Mean Squared Error - MSE and Mean Absolute Error - MAE) during training. This capability helps the model achieve better performance in the presence of outliers or in scenarios with various types of errors. This trainer utilizes the Adam optimizer for parameter updates.

---

### Overall Structure

The main class, `Abtin`, integrates all these components into a unified processing pipeline. By creating an instance of this class and calling its various methods (`package.part(settings)`), you can manage the processing and training flow with your desired configurations.
