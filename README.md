# DeepLabV3+ with DenseDDSSPP
# Automated Road Extraction from Satellite Imagery Integrating Dense Depthwise Dilated Separable Spatial Pyramid Pooling with DeepLabV3+

### [paper](https://arxiv.org/pdf/2410.14836)


### Getting started
**Work in Progress**
This repository currently contains:
- Core components for the DenseDDSSPP module.
- DeepLabV3+ model with DenseDDSSPP integration.


### [Datasets]
The data presented in this study are available in the public domain for research purposes
1.	The Massachusetts Road Dataset can be accessed at https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset
2.	The DeepGlobe Road Dataset is available at https://datasetninja.com/deepglobe-road-extraction


##Training
1. Install dependencies as mentioned in 'requirements.txt'
2. Download the dataset from the public repository:
   - Place all satellite images in data/images
   - Place all corresponding road extracted ground truth images in data/masks
3. Run the training script: `python train.py`
   This will:
   - Dynamically load and split the dataset into training, validation, and test sets.
   - Save the best model checkpoint in the outputs/directory.

##Testing
   Run: `python test.py`  
   This will:
  - Dynamically regenerate the test set using the same split logic.
  - Evaluate the trained model and display metrics like accuracy, IoU, and F1 score.

### Citation
If you use this code for your research, please cite our [paper](https://arxiv.org/pdf/2410.14836).


### Acknowledgments
Our code is developed based on [DeepLabV3+](https://github.com/tensorflow/models/tree/master/research/deeplab) and [Semantic-Segmentation-Architecture](https://github.com/nikhilroxtomar/Semantic-Segmentation-Architecture) using the TensorFlow framework. We appreciate the great contributions provided by DeepLabV3+.
