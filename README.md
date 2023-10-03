# Metastasis_Diagnosis

## Overview

Welcome to the "Precision Cancer Diagnosis" project, where we have developed a robust machine learning model for the classification of histopathologic images extracted from lymph node sections. Our primary goal is to determine the presence of metastatic tissue in these images using advanced deep learning techniques. This project offers a comprehensive analysis of various model architectures, along with code for essential tasks like data preprocessing, model development, and evaluation. We have also provided detailed documentation to help users navigate through the project effectively.

## Business Understanding

Cancer, particularly metastatic cancer, poses a significant threat to human health. Early and accurate detection is crucial for effective treatment and patient outcomes. Traditionally, pathologists analyze histopathologic slides to identify cancerous tissue, a process that is time-consuming and susceptible to human error. By harnessing the power of machine learning models, we aim to streamline this process, leading to faster and more precise cancer diagnoses. Our project addresses the urgent need for automated cancer detection, using the PatchCamelyon dataset to create a benchmark for state-of-the-art machine learning models in the field of histopathology.

## Data Understanding and Exploration 

### The PatchCamelyon Dataset

Our dataset is a comprehensive collection of histopathologic images extracted from lymph node sections, designed for binary classification. It comprises 327,680 high-resolution color images, each measuring 96 x 96 pixels. These images are labeled with binary annotations indicating the presence or absence of metastatic tissue.

### Dataset Information

The dataset used in this project is known as PatchCamelyon (PCam), and it is hosted in the [official PCam GitHub repository](https://github.com/basveeling/pcam). PCam is a valuable resource for machine learning tasks related to image classification, particularly in the field of histopathology. As you may have observed, due to the substantial size of the dataset, it wasn't feasible to upload it directly to GitHub. To retrieve and download the dataset, kindly refer to the 'Download' section available in the official PCam GitHub repository.

### Overview

PCam is a comprehensive collection of high-resolution histopathologic images extracted from lymph node sections. Each image measures 96 x 96 pixels and is provided in color. The dataset is designed for binary image classification tasks, where the goal is to determine the presence or absence of metastatic tissue within these images.

### Dataset Size and Labels

- The PCam dataset consists of an impressive 327,680 images.
- These images are labeled with binary annotations, where:
  - '1' indicates the presence of metastatic tissue.
  - '0' indicates the absence of metastatic tissue.

### Data Distribution

- The dataset is divided into training, validation, and testing sets, ensuring proper evaluation and benchmarking.
- It is important to note that there is no overlap in whole-slide images (WSIs) between the splits.
- The dataset has been carefully balanced, with a 50/50 ratio between positive (metastatic) and negative (non-metastatic) examples in each split.

### Image Annotation

- A positive label (1) indicates that the central 32x32-pixel region of a patch contains at least one pixel of tumor tissue.
- Tumor tissue in the outer region of the patch does not influence the label. This outer region is provided to enable the design of fully-convolutional models.

### Patch Selection

- PCam is derived from the Camelyon16 Challenge and contains data from 400 H&E stained WSIs of sentinel lymph node sections.
- Slides were acquired and digitized at two different centers using a 40x objective, resulting in a pixel resolution of 0.243 microns.
- Data sampling methods, including HSV conversion, blurring, and patch selection, were applied to create the patch-based dataset.

This dataset presents a unique and challenging opportunity for machine learning models, and its large size and balanced distribution make it suitable for a wide range of tasks, including deep learning and image classification.

Please refer to the [official PCam repository](https://github.com/basveeling/pcam) for detailed documentation, usage guidelines, and access to the dataset.

## Note on Model Complexity and Memory Limitations


Please be aware that the utilization of Convolutional Neural Network (CNN) architecture demands substantial computational resources, even exceeding the capabilities of high-end hardware like the Apple M2 Max chip that was used for this project. Consequently, this repository primarily presents the most pertinent outcomes and highlights essential steps involved in hyperparameter tuning. Comprehensive details of the processes responsible for achieving the performance showcased herein, such as Bayesian Optimization, Grid Searches, and Random Searches, are documented separately in a dedicated, auxiliary notebook. It's noteworthy that these processes, besides their computational demands, can also consume a significant portion of local memory, resulting in a slowdown of the overall process. As a result, after determining the optimal hyperparameters, the cells dedicated to hyperparameter tuning were often entirely removed to optimize memory usage, execution time, and overall performance. Nevertheless, this repository serves as a comprehensive guide for the design of a high-performance CNN model capable of accurately diagnosing cancerous tissue.

## Data Preprocessing

In this project, we have undergone several essential data preprocessing steps after partitioning the provided training data into our custom training, validation, and testing sets:

1. To prepare the target variables (y_train, y_test, and y_valid) for utilization in deep learning models, we transformed them into NumPy arrays, adjusted their data type to float32, and reshaped them into column vectors.

2. We standardized the pixel values of the images within the dataset to fall within the range of [0.0, 1.0]. This standardization process involved dividing each pixel value by 255.0. This standardization procedure is a fundamental preprocessing operation frequently applied in deep learning endeavors. Its purpose is to ensure that the pixel values remain consistent and manageable for neural network training, ultimately enhancing model performance.

## Model Performance and Analysis

Our project includes a detailed analysis of the model's performance on both validation and testing datasets. It's essential to highlight some key findings:

The model exhibits commendable overall accuracy on the testing dataset, demonstrating its ability to make correct predictions.
Precision and recall metrics vary between the two classes ("Normal" and "Metastasis"), indicating a trade-off between minimizing false positives and false negatives. The metrics of the model can be observed below:

![metrics](https://github.com/ramses02/Metastasis_Diagnosis/blob/main/Images/metrics.png)

Upon evaluating our best model on the testing dataset, we observed an overall accuracy of 84%. While this accuracy score demonstrates the model's ability to make correct predictions on the test data, a deeper analysis reveals some nuances in its performance across different classes.

The model exhibited a precision of 84% for the "Normal" class, indicating that when it predicted an image as "Normal," it was accurate in doing so 84% of the time. This suggests a reasonable ability to classify "Normal" samples correctly. However, the recall for the "Normal" class was 84%, indicating that out of all the actual "Normal" images, 16% were mistakenly classified as "Metastasis."

Conversely, for the "Metastasis" class, the model demonstrated a recall of 85%, highlighting its proficiency in detecting actual "Metastasis" cases. This means that the model correctly identified 85% of the true "Metastasis" cases in the dataset. 

The calculated F1-scores were 0.84 and 0.84 for the "Normal" and "Metastasis" classes, respectively. These scores indicate a trade-off between precision and recall values for both classes, emphasizing the need to strike a balance between minimizing false positives and false negatives.

It's noteworthy that this model exhibited superior performance on the validation dataset compared to the testing data. Several factors could contribute to this discrepancy: (1) Distributional differences between the validation and testing datasets, where the test data may contain patterns or edge cases not well-represented in the training and validation datasets. (2) Size disparities, where even a few misclassifications in a smaller test dataset can significantly impact performance metrics. (3) The inherent randomness in neural networks, from weight initialization to training dynamics, can lead to varied outcomes, especially in limited dataset scenarios. Additional information on this topic can be found [here](https://stackoverflow.com/questions/60475162/why-is-my-neural-network-validation-accuracy-higher-than-my-training-accuracy-an).

In light of these considerations, the model's commendable validation performance juxtaposed with slightly diminished test results underscores the importance of iterative refinement. Ensuring consistent performance across diverse datasets is crucial for establishing the model's clinical trustworthiness. 

![conf matrix](https://github.com/ramses02/Metastasis_Diagnosis/blob/main/Images/confmatrix.png)

Based on the confusion matrix, the model exhibits a strong performance in classifying cases in a binary classification task. Out of a total of 26,215 cases, the model correctly identified 11,043 cases as belonging to one class ("Positive") and accurately recognized 11,235 cases as belonging to the other class ("Negative"). These correct classifications represent the True Positives and True Negatives, respectively.

However, it's worth noting that no model is without its imperfections. In this case, the model had a tendency to make some misclassifications. Specifically, it incorrectly classified 2,002 cases as belonging to the "Positive" class when they actually belonged to the "Negative" class, constituting the False Positives. Additionally, the model misjudged 1,935 actual "Positive" cases as "Negative," resulting in False Negatives.

Despite these misclassifications, the model demonstrates commendable overall accuracy, reflecting its ability to make correct predictions for a significant portion of the dataset. Furthermore, the model exhibits a high precision, indicating that when it predicts a case as "Positive," it is often reliable in its diagnosis. Additionally, the solid recall score signifies that the model is adept at identifying a substantial proportion of the actual "Positive" cases, highlighting its ability to effectively detect cases of interest.

In summary, this diagnostic tool, as indicated by the confusion matrix, showcases robust classification capabilities, with impressive accuracy, precision, and recall metrics, all of which are vital in medical applications, where the accurate identification of positive cases is of paramount importance for patient well-being.

## Conclusions and Recommendations

### Model Selection 

After rigorous evaluation of various models, it is evident that our selected model has demonstrated exceptional performance in classifying histopathologic images from the PatchCamelyon dataset. It exhibits high precision, effectively distinguishing between the presence and absence of metastatic tissue. Additionally, it maintains a low rate of false positives, a critical factor in medical diagnosis.

### Deployment Recommendations

The consistent excellence exhibited by the model positions it as a strong candidate for integration into clinical settings for the interpretation of lymph node histopathologic images. Healthcare professionals can benefit significantly from its precision, providing valuable insights during the diagnostic process. However, continuous monitoring post-deployment is crucial to ensure that the model maintains its precision and adapts to new data.

## Next Steps

### Data Expansion

To further enhance the robustness and generalization capabilities of the model, it is imperative to expand the dataset. Diversifying the dataset with additional samples and employing data augmentation techniques can expose the model to a broader range of features and variations, ultimately improving its performance.

### Hyperparameter Refinement

Exploring the intricate hyperparameters of the model can unlock even better performance metrics. Techniques such as comprehensive grid search or random search can help identify the optimal combination of hyperparameters, fine-tuning the model for increased accuracy and efficiency, with the help of state-of-the-art machines dedicated to such tasks.

### Transfer Learning Implementation

Considering the limitations of dataset size, integrating pretrained models, such as VGG16, ResNet, or MobileNet, might be beneficial. These models, trained on extensive datasets, can transfer their learned insights to our specific task, potentially enhancing performance even with limited data.

### Clinical Trials

Before full-scale deployment, the model should undergo rigorous testing with a diverse set of histopathologic images in real-world clinical conditions. This process will ensure the model's consistent reliability and validate its applicability in a clinical setting.

### Feedback Integration

Establishing a robust feedback mechanism where clinicians and pathologists can provide feedback and validate the model's predictions is invaluable. This continuous feedback loop will facilitate iterative improvements, refining the model's predictions over time and ensuring its alignment with evolving diagnostic needs in histopathology analysis.

We hope that this project serves as a valuable resource for advancing AI-driven cancer diagnosis and contributes to improving patient outcomes through early and accurate detection. Please feel free to explore the code, documentation, and findings provided in this repository.


## Repository Structure

- `Images/`: Contains the images of the metrics of the model.
- `Auxilliary_Notebook.ipynb/`: Contains the auxilliary notebook and the initial models and searches for hyperparameters.
- `Notebook.ipynb/`: Contains the main notebook and the main, comprehensive corpus of code.
- `README.md/`: Contains information about how to navigate this project.
