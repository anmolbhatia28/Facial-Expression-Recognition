# Facial Expression Recognition Using Torch

## FER Data : 
## Preprocessed FER data : 

## Experiments conducted
1. modified VGG Net on FER data : Validation Accuracy of 44%
2. VGG Net on Preprocessed data: Validation Accuracy of 59%

NOTE: Details of the network are present in models/vgg.lua and model/vgg_modified.lua
To change the model from modified_vgg to vgg, change the model type to vgg or vggbn(VGG with batch normalization) in opts.lua 
The Preprocessing was done using Histogram Equilization

