# Welcome to my Personal Data Science Portfolio!

My name is Ross Jervin Lorenz B. Ruiz, but you may call me "Rjel". I am now a Senior Student (4th Year) of the BS Data Science Program in the University of Science and Technology of Southern Philippines (USTP-CDO). I like reading books, solving puzzles, play video games, musical instruments like guitars and drums, jogging, swimming, biking, and food trips.

Data Science Interests: Data Visualization, Geospatial Data Analysis, Introduction to Artificial Intelligence, Machine Learning.

Skillset: Python, RStudio, Java, C, HTML-CSS-JavaScript trio and their respective frameworks, PHP, MySQL.

This Portfolio will be added with more Notebooks, Lecture/Laboratory Tasks, and other lessons for Deep Learning as time passes by.

## Table of Contents
- Notebook 1: DS414 Elective 4 Portfolio Contents (September 21, 2026)
- Notebook 2: Ruiz, Ross Jervin Lorenz B. - Model Inference on Custom Dataset

Task: Binary Classification on Bird vs. Non-Bird Images.

Frameworks Used: PyTorch, Torchvision, OpenCV, Pandas, Matplotlib, Seaborn, Scikit-Learn.

This laboratory exercise evaluates the performance of five pre-trained torchvision Convolutional Neural Network (CNN) backbones on a binary image classification task (bird vs. non_bird). Instead of training models from scratch, this pipeline leverages zero-shot inference by mapping top-1 ImageNet-1k prediction class indices directly to target binary labels. The complete workflow follows a modular 5-step pipeline:

Raw (Manually Created) Dataset -> Custom PyTorch Dataset and Transformation -> Batched DataLoader -> Multi-Model Inference Engine -> Metrics and Visualization.

Step 1: Setup and Dataset Curation.
- Directory paths are parsed, subfolder targets are aligned (bird: 1, non_bird: 0), and a structured DataFrame index is built.

Step 2: PyTorch Pipeline Construction.
- BGR NumPy arrays are converted to PIL images, resized to 224 * 224, normalized via ImageNet channel statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), and mini-batches are constructed.

Step 3: Batch Inference & Multi-Model Evaluation.
- The models ResNet-18, ResNet-50, MobileNet_V3_Small, EfficientNet_B0, and DenseNet-121 are evaluated. Top-1 predictions falling within ImageNet bird index sets are converted to binary 1.

Step 4: Visualization.
- A grouped bar plot comparing macro metrics across model families is rendered.