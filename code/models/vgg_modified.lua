function createModel(nGPU)
  
   local features = nn.Sequential()
   

features:add(nn.SpatialConvolution(3,64,3,3,1,1,1,1))       
features:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))       
features:add(nn.SpatialMaxPooling(2,2,2,2))
features:add(nn.Dropout(0.5))
features:add(nn.SpatialConvolution(64,128,3,3,1,1,1,1))       
features:add(nn.SpatialConvolution(128,128,3,3,1,1,1,1))       
features:add(nn.SpatialMaxPooling(2,2,2,2))
features:add(nn.Dropout(0.5))
features:add(nn.SpatialConvolution(128,256,3,3,1,1,1,1))       
features:add(nn.SpatialConvolution(256,256,3,3,1,1,1,1))
features:add(nn.SpatialMaxPooling(2,2,2,2))
features:add(nn.Dropout(0.5))

features:add(nn.SpatialConvolution(256,512,3,3,1,1,1,1))       
features:add(nn.SpatialConvolution(512,512,3,3,1,1,1,1))
features:add(nn.SpatialMaxPooling(2,2,2,2))
features:add(nn.Dropout(0.5))

--newly added to speed up the program



features:cuda()
   
local classifier = nn.Sequential()
   classifier:add(nn.View(512*3*3))
   classifier:add(nn.Linear(512*3*3, 2048))
   
   classifier:add(nn.Linear(2048, 1024))
   classifier:add(nn.Linear(1024, 7))
   classifier:add(nn.LogSoftMax())

   classifier:cuda()

    local model = nn.Sequential():add(features):add(classifier)
   model.imageSize = 64
   model.imageCrop = 60

   return model
end