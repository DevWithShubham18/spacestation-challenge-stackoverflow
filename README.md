# spacestation-challenge-stackoverflow
hackaura's 24 hour hackathon AI domain source code by team StackOverflow

# Space Station Safety Detector
An advanced object detection application built with YOLOv8 and Streamlit to identify vital safety equipment in real-time. This tool is designed to enhance safety protocols by quickly locating items like fire extinguishers, oxygen tanks, and first aid kits from an image or camera feed.

# Key FeaturesHigh-Performance Detection: 
Utilizes a custom-trained YOLOv8 model to accurately identify multiple classes of safety equipment. 
Multiple Input Sources: Analyze images by uploading a file, pasting a URL, or capturing a photo from a live webcam.
Interactive UI: A user-friendly interface with a dark/light mode toggle, a live clock, and an adjustable confidence slider.
Detailed Analysis Dashboard: Displays the original and annotated images, a summary table of detected objects, object count metrics, and a bar chart for visualization.
Downloadable Results: Save the annotated image with a single click.

# Tech Stack & Environment
Model: YOLOv8 (by Ultralytics)
Framework: Streamlit
Core Libraries: PyTorch, OpenCV, Pandas, Requests, Pillow
Environment: Python 3.9+
Deployment: Streamlit Community Cloud

# Running the App
Ensure you have the following installed on your system:
Python 3.9 or higher.GitGit LFS (Large File Storage)

Open your terminal and clone this GitHub repository

# Install all required Python packages
pip install -r requirements.txt

# Expected Outputs & Interpretation:
Once an image is processed, the application provides the following outputs in the right-hand column-

Annotated Image: A copy of your input image with bounding boxes drawn around all detected objects. Each box is color-coded and labeled with the object class and a confidence score.

Total Objects Detected: A metric showing the total count of objects found in the image.

Detection Summary Table: A table listing each detected object, its class, and its confidence score.
How to interpret: The Confidence Score (from 0.0 to 1.0) indicates how "sure" the model is about its prediction. A score of 0.85 means the model is 85% confident that the object is correctly identified and located. You can use the slider in the sidebar to filter out detections below a certain confidence level.

Object Counts Chart: A bar chart that visually summarizes the number of instances detected for each object class.

# Results
Replace path/to/your/dataset.yaml with the actual path to your data configuration file.This command will evaluate the model and print a results table showing the Precision, Recall, and mAP50 scores for each class, matching the metrics achieved at the end of the training process

# Bonus - Model Maintenance & Continuous Improvement (Falcon System)
A machine learning model's performance can degrade over time as real-world conditions change. The "Falcon" system is designed to be part of a continuous improvement loop to keep the model accurate and relevant.

Here is how Falcon can be used to maintain and update the model:

Step 1: Identify Model Weaknesses
The Streamlit application serves as the primary tool for identifying when the model needs an update. This happens in two main scenarios:

Existing Objects Change: If a new version of a fire extinguisher is introduced that looks different, the current model may fail to detect it. Users of the app would notice this failure.

New Objects are Introduced: If a new type of safety equipment is added to the environment like a "screwdriver", the model will be completely unaware of it and may misclassify it as something else, causing confusion.

Step 2: Collect New Data
The Falcon application could be enhanced with a "Flag Incorrect Detection" button. When a user sees a missed or wrong detection, they could press this button to automatically save the problematic image to a dedicated storage location for review. This creates a feedback loop where the app itself helps collect the exact data needed to improve it.

Step 3: Re-label and Augment
The newly collected images are then annotated by a human expert.

For changed objects, the existing labels are updated.

For new objects, a new class eg: "screwdriver" is added to the dataset, and the new images are labeled accordingly.

This new, small dataset is then combined with the original training data.

Step 4: Retrain the Model
The YOLOv8 model is then retrained on this updated, larger dataset. The process of "fine-tuning" from the previously trained weights allows the model to quickly learn the new patterns without forgetting what it already knows.

Step 5: Evaluate and Deploy
The newly trained model is evaluated on a test set. If its performance i.e. map50 score is better than the currently deployed model, the new best.pt file is pushed to the GitHub repository. Streamlit Community Cloud automatically detects the change and redeploys the application with the improved model, completing the cycle.

This MLOps loop ensures the Falcon system remains robust, adaptable, and continuously improves over time.



