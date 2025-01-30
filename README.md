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

### **Motivation**

Equivariance and invariance, a special case of equivariance, are useful properties when dealing with data containing symmetries. Classical convolution neural networks are considered translationally equivariant. That is, if the input image is translated, the feature maps are translated in the same manner, preserving the spatial relationships. This allows features of the image to be recognized in the same way, regardless of their position within the image. 

However, this does not generalize to convolution on rotations of the input. Additionally, while rotations in $SO(2)$ and translations in $\mathbb{R}^2$ are individually commutative (i.e. any permutation of the same set of rotations or set of translations will yield the same result), this is not true of the group of rotations and translations together, known as the Euclidean group $E(2)$ in $\mathbb{R}^2$. 

Many data types, medical imaging data among them, contain rotational (e.g. tumors are the same regardless of orientation in a WSI), reflectional (e.g. bilateral structures of brain or body), and translational symmetries. Firstly, equivariance would be useful in a medical imaging context to increase generalizability of models regardless of the orientation that data is presented. Secondly, I claim that equivariance implicitly deals with the need for certain kinds of data augmentations, improving overall training efficiency. 

### **Equivariance**

Three channel example:

For scalar outputs ($J_s_j(x)$):

$J_s^1(x) = \int f_{11}(|x - x'|)I_1(x') \, dx' + \int f_{12}(|x - x'|)I_2(x') \, dx' + \int f_{13}(|x - x'|)I_3(x') \, dx'$

$J_s^2(x) = \int f_{21}(|x - x'|)I_1(x') \, dx' + \int f_{22}(|x - x'|)I_2(x') \, dx' + \int f_{23}(|x - x'|)I_3(x') \, dx'$

For vector outputs ($J_v_j(x)$):

$J_v^1(x) = \int g_{11}(|x - x'|)(x - x')I_1(x') \, dx' + \int g_{12}(|x - x'|)(x - x')I_2(x') \, dx' + \int g_{13}(|x - x'|)(x - x')I_3(x') \, dx'$

$J_v^2(x) = \int g_{21}(|x - x'|)(x - x')I_1(x') \, dx' + \int g_{22}(|x - x'|)(x - x')I_2(x') \, dx' + \int g_{23}(|x - x'|)(x - x')I_3(x') \, dx'$
