# Writeup

The repository contains following files

- `model_training.ipynb`
  - It's a Jupyter notebook to train the segmentation netwrok
- `model_training.html`
  - HTML version of the notebook
- `model.h5`
  - Contains model architecture and weights of the network
- `weights.h5`
  - Contains only weights of the network

- `writeup.md`
  - Contains writeup for the project

- `images` directory
  - Contains images for this writeup

---

### Training

- Following is a graph showing training of the model for 20 epochs

> Training graphs is in cell 9 of the notebook

![training](https://github.com/Shilpaj1994/Follow-me/blob/master/images/training.png?raw=true)

- After the 20 epochs, the loss is still decreasing and the **mean-IoU is 49.60%**

> Mean IoU calculation is in cell 21 of the notebook

---

### Model Performance

- Left image: Input image from quad camera
- Center image: Ground truth
- Right image: Model output

> More image comparison is in the cell 13, 14 and 15 of the notebook

![result](https://github.com/Shilpaj1994/Follow-me/blob/master/images/result.png?raw=true)

---

## Model Architecture

- We want to detect where exactly the 'hero' is in the image. This makes it as a detection problem and not a classification problem.
- To detect the exact location of the 'hero' in an image, we need a network in which spatial information is carried till the output layer and hence we have used a Fully Convolutional Network (FCN) 
- In Fully Convolutional Network (FCN), dimensions of input and output are same

![Network Image](https://github.com/Shilpaj1994/Follow-me/blob/master/images/network.png?raw=true)

> Code for Model Architecture is in the cell 6

- The network includes following components
  - Encoder
  - Dilation Layer
  - 1x1 convolution
  - Decoder

- **Encoder Block**
  - It contain a pair of **Separable Convolutional Layer** and **Batch Normalization Layer**.
  - The model is going to be implemented on a quadcopter which does not have powerful hardware for processing the camera images.
  - For such embedded hardware, separable convolutions layer is used. It reduce the number of parameters significantly and thus increases the performance on inference side.
  - Batch Normalization Layer helps to train network faster. [Batch Normalization Benefits](https://towardsdatascience.com/batch-normalization-8a2e585775c9)

> Code for Encoder Block is in the cell 2 and 4

- **Dilation Convolution**

  ![](https://github.com/Shilpaj1994/Phase1_assignments/blob/master/Assignment%203/Files/dilated.gif?raw=true)
  - For segmentation networks, I wanted to expand the dimension covered by the kernel to improve the result of the network.
  - Increasing kernel size was another option but it would have slow downed the network on inference side as the number of parameters would have increased.
  - Hence, I have used a dilation convolution in the network

  > Code for Dilation Layer is in the cell 6

- **1x1 Convolution**

  ![](https://github.com/Shilpaj1994/Phase1_assignments/blob/master/Assignment%201/1x1convolution.png?raw=true)

  - 1x1 or pointwise convolution layer is used to combine to the information from all the preceding layers.
  - Flatten layer converts the 4D vector into 2D and thus there is a loss of spatial information
  - To avoid this loss of spatial information, 1x1 are used.

  > Code for 1x1 layer is in the cell 6

- **Decoder**

  - The decoder blocks contains the **Skip Connection** and a **Upsampling Layer**
  - The rich information in the initial layers is not passed to the later layers in the network. Skip connections are used to carry this information from initial layers to the later layers
  - Since, the model output dimensions are same as input dimensions upsampling is done.
  - Bilinear upsampling is done to reduce the number of parameters and optimize the performance of the network

  > Code for Decoder Block is in the cell 3 and 5

---

### Hyperparameters

> Hyperparameters are listed in the cell 8

```python
learning_rate = 0.01
batch_size = 32
num_epochs = 50
```

- Learning rate should be a small value. I started with 0.01 and it did train model
- I started with batch size of 64 but ran into error `OOM(Out Of Memory)` so reduced the size to 32
- The losses were fluctuating for initial 15-20 epochs so selected 50 epochs so that loss stabalizes

---

### Future Enhancements

- Increasing efficiency
- Trying out different segmentation architecture like UNet, SegNet, etc
- Hyperparameter tuning to train the model faster

---

### Limitations

This model is specific for the pedestrian detection. It has only following three classes

- Hero
- Pedestrians
- Everything else than above two

Since, the model is trained on only these 3 classes, it won't be useful to follow any other car, cat, dog, etc