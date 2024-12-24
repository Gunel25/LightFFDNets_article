# About the Project  
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
## How to Use

### Run the Codes for LightFFDNet v1 or LightFFDNet v2 in MATLAB

- **LightFFDNet v1**: Designed for faster training with fewer layers.  
- **LightFFDNet v2**: Includes additional layers for improved feature extraction and higher accuracy.

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

---
## Model Overview and Implementation Details

### 1. Execution Environment
- **Required Software**: MATLAB R2021b or later.  
- **Necessary Toolbox**: MATLAB Deep Learning Toolbox.  
- **Dataset**: Dataset2 (divided into Train, Validation, and Test).  
- **Input Dimensions**: `[224, 224, 3]` pixels (RGB images).  


### 2. Key Algorithms and Principles of the Model
#### Layers
- **Input Layer**:
  - Dimensions: `[224 x 224 x 3]`.
  - Processes images in RGB format.
- **Convolutional Layers**:
  - **For LightFFDNet v1**:
    - First Layer: 32 filters of size `3x3` with 'same' padding, extracting image features.
    - Second Layer: 64 filters of size `3x3`, focusing on higher-level features.
  - **For LightFFDNet v2**:
    - First Layer: 32 filters of size `3x3` with 'same' padding, extracting image features.
    - Second Layer: 64 filters of size `3x3`, focusing on intermediate-level features.
    - Third Layer: 128 filters of size `3x3` for deeper feature extraction.
    - Fourth Layer: 256 filters of size `3x3` to capture complex patterns.
    - Fifth Layer: 512 filters of size `3x3` for detailed feature learning.
- **Activation Function**:
  - **ReLU (Rectified Linear Unit)**: Replaces negative values with zero, improving computational efficiency.
- **Pooling Layer**:
  - **Max Pooling (2x2 size)**: Reduces feature dimensions and minimizes noise.
- **Fully Connected Layer**:
  - Maps features to two classes (for classification purposes).
- **Softmax Layer**:
  - Outputs class probabilities.
- **Classification Layer**:
  - Selects the class with the highest probability.


### 3. Optimization Method
- **Optimizer**: Adam (Adaptive Moment Estimation).  
- **Learning Rate**: `0.0001` (fixed).  
- **Epochs**: Maximum of 10 iterations.  
- **Batch Size**: 16 (to optimize memory usage).  


### 4. Reproducibility of Experiments
- Image data is read using the `imageDatastore` object, which automatically loads and labels images stored in subfolders.  
- Validation and Testing: Accuracy during training and testing phases is ensured.

---
## Explanation: Using Flexible Code Structure for Transfer Learning Models
Run the Codes from "Models Transfer Learning" in MATLAB. This code is designed to work with various transfer learning models. The main idea is that certain sections of the code must be activated (i.e., uncommented) based on the chosen model. Key parts that vary according to the selected model are as follows:

### 1. Model Selection:
At the beginning of the code, there is a model selection section. Only activate the model you intend to use.
For example, to use MobileNetV2, activate the following line:
- net = mobilenetv2;
### 2. Modifying Layers:
Each model has unique names for its final layers. Activate the appropriate layer modification steps based on your chosen model:

For example, MobileNetV2:
- lgraph = replaceLayer(lgraph,'Logits',newLearnableLayer);
- lgraph = replaceLayer(lgraph,'ClassificationLayer_Logits',newClassLayer);

For other models, refer to the relevant layer names provided in the code and activate the corresponding layer modification sections.

### 3. Dataset Selection:
Each model has its own dataset loading and splitting procedure as shown in earlier sections. There is no need to modify the dataset loading part since the code has been generalized for all models.

### 4. Training and Testing:
Once the model is selected and adjusted, the remaining code (training options, model training, and testing) remains unchanged and works automatically.

This structure allows you to switch flexibly between different transfer learning models. Simply choose the model and activate the corresponding layer modification sections to adapt the code for your desired model.


---
# Article
Access my article from the link below:
   
https://doi.org/10.48550/arXiv.2411.11826

# Citation
Please cite as:

    @INPROCEEDINGS{
      JabbarliG2024arXiv,
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
