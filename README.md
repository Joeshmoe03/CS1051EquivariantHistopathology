### **Requirements**

A conda environment with the following:
```
pytorch
seaborn
escnn
e2cnn
numpy
torchvision
torchmetrics
monai
albumentations
scikit-learn
pandas
matplotlib
```

### **Setup**

NOTE: Unless you have cluster access, I don't recommend training anything. 

1. Install all requirements in a python environment
2. Run setup.sh (I need to double check this)
3. activate your environment
4. run the v1_0.ipynb notebook to generate important remaining files and visualizations

5. To train, configure the appropriate .sh script for training (according to your cluster and resources)
6. Set up comet_ml account for training tracking and input your API key to the training script

### **Motivation**

Equivariance and invariance, a special case of equivariance, are useful properties when dealing with data containing symmetries. Classical convolution neural networks are considered translationally equivariant. That is, if the input image is translated, the feature maps are translated in the same manner, preserving the spatial relationships. This allows features of the image to be recognized in the same way, regardless of their position within the image. 

However, this does not generalize to convolution on rotations of the input. Additionally, while rotations in $SO(2)$ and translations in $\mathbb{R}^2$ are individually commutative (i.e. any permutation of the same set of rotations or set of translations will yield the same result), this is not true of the group of rotations and translations together, known as the Euclidean group $E(2)$ in $\mathbb{R}^2$. 

Many data types, medical imaging data among them, contain rotational (e.g. tumors are the same regardless of orientation in a WSI), reflectional (e.g. bilateral structures of brain or body), and translational symmetries. Firstly, equivariance would be useful in a medical imaging context to increase generalizability of models regardless of the orientation that data is presented. Secondly, I claim that equivariance implicitly deals with the need for certain kinds of data augmentations, improving overall training efficiency. 

Comet_ML: https://www.comet.com/joeshmoe03/deep-learning/view/new/panels#manage



