require 'image'
require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'
require 'cudnn'
--dofile('donkey.lua')

model = torch.load('/input/model_.t7')
--net = cudnn.convert(model, nn)
--torch.save(paths.concat('/output/', 'model_.t7'), model)

--print(model.modules[1].modules[1]:size())
--print(model.modules[1].modules[1].modules[1].weight)
--print(model.modules[1].modules[1].modules[1].weight:size())
--image.save("/output/filters.png",image.toDisplayTensor(model.modules[1].modules[1].modules[1].weight ))

--for i=1,5 do
    --print('layer no.' ..i)
    --for j=1,i do
        --if j==1 then
           -- if i==1 then
            --    n  = model.modules[1].modules[1].modules[i]
           -- else
           --     n=model.modules[1].modules[1].modules[j].output
           -- end
     --   elseif j==i then
        --    n  = n.modules[1].modules[i].output
--else
          --  n  = n.modules[1].modules[j]
      --  end
  --  end
   
    
    print(model.modules)
    print('----------')
    print(model)
    img = image.load('./data_predict/30.jpg')
    n1  = model.modules[1].modules[1].modules[1]
    res = n1:forward(img:cuda() )
    image.save("/output/layer1.png",image.toDisplayTensor(res))
    n2 = model.modules[1].modules[1].modules[2]
    res = n2:forward(res)
    image.save("/output/layer2.png",image.toDisplayTensor(res))
    n3 = model.modules[1].modules[1].modules[3]
    res = n3:forward(res)
    image.save("/output/layer3.png",image.toDisplayTensor(res))
    n4 = model.modules[1].modules[1].modules[4]
    res = n4:forward(res)
    image.save("/output/layer4.png",image.toDisplayTensor(res))
    n5 = model.modules[1].modules[1].modules[5]
    res = n5:forward(res)
    image.save("/output/layer5.png",image.toDisplayTensor(res))
    n6 = model.modules[1].modules[1].modules[6]
    res = n6:forward(res)
    image.save("/output/layer6.png",image.toDisplayTensor(res))
     n7  = model.modules[1].modules[1].modules[7]
    res = n7:forward(res)
    image.save("/output/layer7.png",image.toDisplayTensor(res))
    n8 = model.modules[1].modules[1].modules[8]
    res = n8:forward(res)
    image.save("/output/layer8.png",image.toDisplayTensor(res))
    n9 = model.modules[1].modules[1].modules[9]
    res = n9:forward(res)
    image.save("/output/layer9.png",image.toDisplayTensor(res))
    n10 = model.modules[1].modules[1].modules[10]
    res = n10:forward(res)
    image.save("/output/layer10.png",image.toDisplayTensor(res))
    n11 = model.modules[1].modules[1].modules[11]
    res = n11:forward(res)
    image.save("/output/layer11.png",image.toDisplayTensor(res))
    n12 = model.modules[1].modules[1].modules[12]
    res = n12:forward(res)
    image.save("/output/layer12.png",image.toDisplayTensor(res))
    n13 = model.modules[1].modules[1].modules[13]
    res = n13:forward(res)
    image.save("/output/layer13.png",image.toDisplayTensor(res))
    print('---------------------------------------------------')
    print(model.modules[1].modules[1].modules[13].output:size())
    print('---------------------------------------------------')
    print(model.modules[1].modules[1]:size())
    
    
    
    n14 = model.modules[2].modules[1]
    res = n14:forward(res)
    image.save("/output/layer2_1.png",image.toDisplayTensor(res))
    
--print(model.modules[1].modules[i+1]:size())
