require 'nn'
require 'cunn'
require 'optim'
require 'torch'
require 'image'
dofile('donkey.lua')
img = testHook({loadSize}, '../data_predict/0.jpg')
model = torch.load('../data_predict/model_1.t7')
if img:dim() == 3 then
  img = img:view(1, img:size(1), img:size(2), img:size(3))
end
predictions = model:forward(img:cuda())