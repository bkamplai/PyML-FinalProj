# SignSense: ASL Fingerspelling Classifier

## Team Members

 | Name | Email | Roles |
 |------|-------|--------------|
 |Aidan K.|amkavanagh@mavs.coloradomesa.edu|Hand Tracking Skeleton Model|
 |Elena S.|eeschmitt@mavs.coloradomesa.edu|Website/Flask|
 |Brandon K. |bekamplain@mavs.coloradomesa.edu |Model Training|

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
