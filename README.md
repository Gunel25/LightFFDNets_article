## About the Project  
Accurate and rapid facial forgery detection is essential in fields like artificial intelligence, image processing, and object detection. This project introduces LightFFDNet v1 and LightFFDNet v2, two lightweight convolutional neural network models designed to efficiently detect facial forgeries. Additionally, the performance of these models has been compared with 8 pretrained CNN architectures, showcasing their effectiveness and superiority in various aspects of forgery detection, such as accuracy, processing speed, and resource utilization.

# LightFFDNet
LightFFDNet: Lightweight CNNs for Efficient Facial Forgery Detection

The key contributions of this project are:  
1. **LightFFDNet v1**:  
   - Only two convolutional layers.  
   - High speed and competitive accuracy.  
   - Suitable for resource-constrained environments.  
2. **LightFFDNet v2**:  
   - Incorporates five convolutional layers.  
   - Delivers enhanced accuracy, especially on larger datasets.  
   - Maintains computational efficiency.  

The models were evaluated using the **Fake-Vs-Real-Faces (Hard)** and **140k Real and Fake Faces** datasets.  

---

## Comparison with Existing Models
The performance of LightFFDNet v1 and LightFFDNet v2 has been compared with 8 pretrained CNN architectures. The following CNN architectures were used for comparison:

1. **ResNet-50**
2. **ResNet-101**
3. **VGG-16**
4. **VGG-19**
5. **GoogleNet**
6. **MobileNetV2**
7. **Alexnet**
8. **DarkNet-53**

The results demonstrate that LightFFDNet models outperform these existing models in key metrics such as accuracy, speed, and resource efficiency.
For a detailed explanation of the comparison, including training/test splits and additional metrics, please refer to the [full paper](https://doi.org/10.48550/arXiv.2411.11826).







## Dataset Information  

This project utilizes two datasets, described below:  

### <span style="color:blue">1. **Fake-Vs-Real-Faces (Hard) Dataset**</span>  
- The dataset was downloaded as a whole and saved as a file on the local machine.
- The dataset was split into 70% train, 20% test, and 10% validation using the provided code.
- The splitting process is implemented in the provided code examples, which load the dataset from the local directory.
- [Download Fake-Vs-Real-Faces (Hard) Dataset](https://www.kaggle.com/datasets/hamzaboulahia/hardfakevsrealfaces)  

### <span style="color:green">2. **140k Real and Fake Faces Dataset**</span>  
- A subset of this dataset, equivalent in size to the Fake-Vs-Real-Faces (Hard) dataset, was used in this project.  
- This dataset was already pre-split into the following proportions on the local machine:
     70% train,
     20% test,
     10% validation
- No additional splitting was performed using the code; the dataset is loaded directly from the local directory.  
- [Download 140k Real and Fake Faces Dataset](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)  



---
# How to Use

## 1. Run the Codes from LightFFDNet v1 in MATLAB

To use the LightFFDNet v1 model for training and testing on your dataset, follow these steps:

### 1.1 Prepare Your Dataset:
- Ensure your dataset (e.g., "140k Real and Fake Faces Dataset" or "Fake-Vs-Real-Faces (Hard) Dataset") is organized with images in subfolders. Each subfolder should represent a class (e.g., "Real" and "Fake").
- Ensure your dataset is structured as follows:
  - **For the "140k Real and Fake Faces Dataset":**
    - The dataset should include the following folders:
      - **Train**: Contains training images categorized into subfolders (e.g., "Real" and "Fake").
      - **Validation**: Contains validation images, also categorized into subfolders.
      - **Test**: Contains test images with the same class-based subfolder structure.
  
  - **For the "Fake-Vs-Real-Faces (Hard) Dataset":**
    - Ensure the images are resized to `[224x224x3]` pixels to maintain consistency with the model’s input requirements.


### 1.2 Modify Dataset Path:
- Update the dataset path in the code to point to the location where your dataset is stored on your computer. Make sure to specify the paths for each dataset (train, validation, test) in the respective sections of the code.

### 1.3 Resize the Images:
- Ensure that all images are resized to the specified input size (224x224x3 pixels) to be compatible with the model. The code automatically resizes the images during preprocessing.

### 1.4 Create and Train the Model:
- Run the code to define the CNN architecture. The model will then be trained using the training dataset and validated on the validation set.
- Training will proceed with the configured settings, including the learning rate and number of epochs.

### 1.5 Test the Model:
- After training, the model's performance will be evaluated on the test dataset. The accuracy will be computed by comparing the predicted labels with the actual labels of the test set.

### 1.6 Monitor Training Progress:
- During the training process, you can monitor the model's progress through training plots. These plots will help you visualize the learning curve and assess the model's performance.

By following these steps, you can train and evaluate the **LightFFDNet v1** model using your dataset in MATLAB.











# How to Use

## 1. Run the codes from LightFFDNet in Matlab.
## 2. For training on custom dataset
1. Download the dataset as a file to your computer. The link is provided above. Specify the file path and prepare it for MATLAB.
2. If you have your own custom dataset with a different annotation system, you can train it using the LightFFDNet code in MATLAB. Modify the code as needed to load and use your dataset.


# Article
Access my article from the link below:
   
https://doi.org/10.48550/arXiv.2411.11826

# Citation
Please cite as:

    @INPROCEEDINGS{
      Jabbarli2024arXiv,
      author={Jabbarli, Gunel and Kurt, Murat},
      title = {LightFFDNets: Lightweight Convolutional Neural Networks for Rapid Facial Forgery Detection},
      journal = {arXiv preprint arXiv:2411.11826},
      pages = {arXiv:2411.11826},
      year = {2024},
      month = nov,
      eid = {arXiv:2411.11826},
      doi = {10.48550/arXiv.2411.11826},
      url = {https://doi.org/10.48550/arXiv.2411.11826}, 
      archivePrefix = {arXiv},
      eprint = {2411.11826}
}
