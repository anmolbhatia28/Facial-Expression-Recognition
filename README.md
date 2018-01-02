# Facial Expression Recognition Using Torch

## FER Data : https://drive.google.com/file/d/1bfKmoq4IregQnmQI2XCAcjJ8d25HcIHI/view?usp=sharing
## Preprocessed FER data : https://drive.google.com/file/d/1rycn9pRX3fkCTUx9W6CVFgxesa7sdwvS/view?usp=sharing

## Experiments conducted
1. modified VGG Net on FER data : Validation Accuracy of 44%
2. VGG Net on Preprocessed data: Validation Accuracy of 59%

NOTE: Details of the network are present in models/vgg.lua and model/vgg_modified.lua
To change the model from modified_vgg to vgg, change the model type to vgg or vggbn(VGG with batch normalization) in opts.lua 
The Preprocessing was done using Histogram Equilization

