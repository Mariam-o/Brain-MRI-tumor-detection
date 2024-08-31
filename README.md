1. Preprocessing: Cropped image edges to focus on key areas.
   
2. Data Augmentation: Applied random rotations, shearing, flips, brightness adjustments, and nearest pixel filling to enhance the dataset.

3. Data Splitting: Ensured equal distribution of 'yes' and 'no' cases across training, validation, and test sets.

4. Model Training:
   - Custom CNN achieved 84% accuracy.
   - Switching to VGG16 improved accuracy to 93%.

5. Evaluation & Deployment: Visualized training/validation metrics and deployed the model via Streamlit, enabling users to upload MRI scans for tumor prediction.
