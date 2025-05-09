# SignSense: ASL Fingerspelling Classifier

## Team Members

 | Name | Email | Roles |
 |------|-------|--------------|
 |Aidan K.|amkavanagh@mavs.coloradomesa.edu|Hand Tracking Skeleton Model|
 |Elena S.|eeschmitt@mavs.coloradomesa.edu|Website/Flask|
 |Brandon K. |bekamplain@mavs.coloradomesa.edu |Model Training|

## Our Grade using Rubric

| Criteria | Score (1-5) | Justification |
| :-- | :--: | :-- |
| **1. Problem Understanding** |
| Clearly defined problem statement | 5 | ASL fingerspelling recognition remains a challenging task due to variability in hand shapes, signer styles, and environmental conditions such as lighting. |
| Understanding of the domain and context | 5 | Project aims to develop a machine learning-based classifier for ASL fingerspelling recognition using CNNs and image augmentation. The system will enable real-time recognition and feedback for ASL education and accessibility.|
| **2. Data Preprocessing** |
| Data cleaning and handling missing values | 5 | While missing values weren’t a concern, data cleaning was performed to reconcile structural inconsistencies when merging two large ASL datasets. |
| Feature engineering and selection | 5 | See https://github.com/bkamplai/PyML-FinalProj/blob/main/training/compare_models.md|
| Data normalization and scaling | 5 | Images were normalized to the [0, 1] range, and combining both datasets increased class sample counts from ~900 to over 9,000, improving model generalizability. |
| **3. Model Selection and Evaluation** |
| Selection of appropriate ML algorithms | 5 | Evaluated MLP, CNN, MobileNetV2, ResNet50, and EfficientNetB0. See https://github.com/bkamplai/PyML-FinalProj/blob/main/training/compare_models.md|
| Model training and tuning | 5 | Tuned hyperparameters (epochs, dropout, learning rate), and used transfer learning with frozen and fine-tuned MobileNetV2. See https://github.com/bkamplai/PyML-FinalProj/blob/main/training/train_mobilenet.py |
| Evaluation metrics and performance analysis | 5 | Used training/validation/test accuracy, loss curves, and confusion matrices. See https://github.com/bkamplai/PyML-FinalProj/tree/main/training/Screenshots |
| **4. Creativity and Innovation** |
| Novelty and originality of approach | 5 | We introduce a novel ASL recognition pipeline that combines multi-dataset training, confidence-aware filtering, and real-time webcam inference for robust, interactive deployment. |
| Exploration of advanced techniques (Deep Learning, e.g.) | 5 | Leveraged deep learning with transfer learning, dropout regularization, data augmentation, and model fine-tuning. See https://github.com/bkamplai/PyML-FinalProj/tree/main/training|
| **5. Presentation** |
| Quality of visualizations and insights | 5 | Our model comparison includes accuracy/loss plots and test predictions to illustrate performance trends and classifier effectiveness.|
| Ability to communicate results effectively | 5 | The presentation follows the START structure, includes a demo slide, and breaks down technical steps clearly.|
| **Final Score**| 60/60 |

## How to Run
1. Download git repo as a zip.
2. Traverse to app folder and run the app.py program using Python.
   ```bash
   cd app
   python3 app.py
   ```
3. Open http://127.0.0.1:5000/ in web browser.
4. Upload an image, a video, or click on link to start live webcam.

## Abstract
### Title: SignSense: ASL Fingerspelling Classifier
American Sign Language (ASL) is a vital means of communication for the Deaf and Hard of Hearing community. However, automatic recognition of ASL fingerspelling remains a challenging problem due to variations in hand shapes, lighting conditions, and signer differences. This project aims to develop a machine learning-based classifier that recognizes ASL fingerspelling letters from images or live webcam input. Using a Convolutional Neural Network (CNN), we will train a model on a diverse dataset of ASL hand signs, optimizing it for real-time performance. The project includes preprocessing techniques such as image augmentation to improve model robustness. A user-friendly web or desktop application will be developed to allow real-time recognition, providing feedback on the predicted letters. Our approach will compare multiple deep learning architectures, analyze classification accuracy, and explore potential applications in ASL education and accessibility. The final system will demonstrate how artificial intelligence can enhance ASL learning and communication.

### TLDR;

Project aims to develop a machine learning-based classifier for ASL fingerspelling recognition using CNNs and image augmentation. The system will enable real-time recognition and feedback for ASL education and accessibility.

### Model: asl_fingerspell_finetuned_combined.keras
### [Dataset](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet)
### [Other Dataset?](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset)
### Delete "space" and "del" folders and rename "nothing" to "Blank".
