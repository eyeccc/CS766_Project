# CS766 Project

This project involves two parts of implementation: 

1. A Neural Algorithm of Artistic Style (http://arxiv.org/abs/1508.06576)

2. Style classification (based on several papers)

For full information, including reference, please visit: http://eyeccc.github.io/CS766_Project/ to check our reports.

## Part1. Generating stylish images 

Implementation for "A Neural Algorithm of Artistic Style"

### Dependency

Chainer v1.8.0 (http://chainer.org/), Numpy, Scipy.misc

### Convolutional Neural Network Model

VGG19 caffe-model (https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md)

### Usage

```
python nn_art.py -s PATH_TO_STYLE_IMG -c PATH_TO_CONTENT_IMG
```

=======
## Part2. Style Classification
1. feature.xlsx contains all the paintings' features and its labels.
2. readFile.py read the feature.xlsx file and is used to generate training and testing set.
3. Classify.py classify paintings based on artists.
4. TwowayClassifier.py classify paintings based on art movements.
5. Wrong_samples.txt contains the wrongly classified paintings, we used this for the debuging and tuning. 

### Dependency

scikit-learn, Numpy, openpyxl. To make things easier, you can download the Anaconda python package manager. 

### Usage

```
python Classify.py 
```


