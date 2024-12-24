## About the Project  
Accurate and rapid facial forgery detection is essential in fields like artificial intelligence, image processing, and object detection. This project introduces LightFFDNet v1 and LightFFDNet v2, two lightweight convolutional neural network models designed to efficiently detect facial forgeries. Additionally, the performance of these models has been compared with 8 other existing models, showcasing their effectiveness and superiority in various aspects of forgery detection, such as accuracy, processing speed, and resource utilization.

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
For a detailed explanation of the comparison, including training/test splits and additional metrics, please refer to the full paper.
https://doi.org/10.48550/arXiv.2411.11826






## Dataset Information  

This project utilizes two datasets, described below:  

### <span style="color:blue">1. **Fake-Vs-Real-Faces (Hard) Dataset**</span>  
- The dataset was downloaded as a whole and split into **70% train**, **20% test**, and **10% validation** using code.  
- The splitting process is implemented in the provided code examples.  
- [Download Fake-Vs-Real-Faces (Hard) Dataset](https://www.kaggle.com/datasets/hamzaboulahia/hardfakevsrealfaces)  

### <span style="color:green">2. **140k Real and Fake Faces Dataset**</span>  
- Only a subset of the dataset, equivalent to the size of the **Fake-Vs-Real-Faces (Hard) Dataset**, was used in this project.  
- This dataset was already pre-split on the computer into the following proportions:  
  - **70% train**,  
  - **20% test**,  
  - **10% validation**.  
- No additional splitting was performed using the code.  
- [Download 140k Real and Fake Faces Dataset](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)  



---


Additionally, we compare our models with eight pretrained CNN architectures to assess their performance. The following CNN architectures were used for comparison:

1. **ResNet-50**
2. **ResNet-101**
3. **VGG-16**
4. **VGG-19**
5. **GoogleNet**
6. **MobileNetV2**
7. **Alexnet**
8. **DarkNet-53**
   
The results demonstrate that **LightFFDNet v1** and **LightFFDNet v2** perform competitively with these pretrained models while maintaining high computational efficiency.

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
