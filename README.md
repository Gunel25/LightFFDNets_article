## About the Project  
Accurate and rapid facial forgery detection is essential in fields like **artificial intelligence**, **image processing**, and **object detection**. This project introduces **LightFFDNet v1** and **LightFFDNet v2**, two lightweight convolutional neural network models designed to efficiently detect facial forgeries.  

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


## Datasets  
The following datasets were used for evaluation:  

1. **Fake-Vs-Real-Faces (Hard)**  
   - [Dataset Link](https://www.kaggle.com/datasets/hamzaboulahia/hardfakevsrealfaces)  

2. **140k Real and Fake Faces**  
   - [Dataset Link](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)  


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

## 1. Run the codes from SiStNet-inference.ipynb in Jupyter Notebook.
## 2. For training on custom dataset
1. This codes load the dataset in YOLO format. You can see an annotation example in './dataset/valid_example/labels' folder.
2. If you have your own custom dataset with different annotation system, then you can train it by running thecodes in SiStNet.ipynb in jupyter notebook. You can change the codes for your purposes to load the dataset.

# Article
Access my article from the link below:

    https://doi.org/10.48550/arXiv.2401.17972

# Citation
Please cite as:

    @INPROCEEDINGS{
      Azadvatan2024arXiv,
      author={Azadvatan, Yashar and Kurt, Murat},
      title = {MelNet: A Real-Time Deep Learning Algorithm for Object Detection},
      journal = {arXiv preprint arXiv:2401.17972},
      pages = {arXiv:2401.17972},
      year = {2024},
      month = jan,
      eid = {arXiv:2401.17972},
      doi = {10.48550/arXiv.2401.17972},
      url = {https://doi.org/10.48550/arXiv.2401.17972}, 
      archivePrefix = {arXiv},
      eprint = {2401.17972}
    }
