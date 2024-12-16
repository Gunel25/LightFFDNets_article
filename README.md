## About the Project  
Accurate and rapid facial forgery detection is essential in fields like **artificial intelligence**, **image processing**, and **object detection**. This project introduces **LightFFDNet v1** and **LightFFDNet v2**, two lightweight convolutional neural network models designed to efficiently detect facial forgeries.  

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

```markdown
## DOI for this Repository
This project is assigned a DOI to enable citation:
[DOI: 10.5281/zenodo.14499827](https://zenodo.org/records/14499827)
