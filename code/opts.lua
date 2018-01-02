--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Torch-7 Imagenet Training script')
    cmd:text()
    cmd:text('Options:')
    ------------ General options --------------------

    cmd:option('-cache', './imagenet/checkpoint/', 'subdirectory in which to save/log experiments')
    cmd:option('-data', './imagenet/imagenet_raw_images/256', 'Home of ImageNet dataset')
    cmd:option('-manualSeed',         2, 'Manually set RNG seed')
    cmd:option('-GPU',                1, 'Default preferred GPU')
    cmd:option('-nGPU',               1, 'Number of GPUs to use by default')
    cmd:option('-backend',     'cudnn', 'Options: cudnn | nn')
    cmd:option('-cudnnAutotune',     0, 'Enable the cudnn auto tune feature Options: 1 | 0')
    ------------- Data options ------------------------
    cmd:option('-nDonkeys',        2, 'number of donkeys to initialize (data loading threads)')
    cmd:option('-imageSize',         64,    'Smallest side of the resized image')
    cmd:option('-cropSize',          60,    'Height and Width of image crop to be used as input layer')
    cmd:option('-nClasses',        7, 'number of classes in the dataset')
    ------------- Training options --------------------
    cmd:option('-nEpochs',         80,    'Number of total epochs to run')
    cmd:option('-epochSize',      220, 'Number of batches per epoch')
    --earlier 158
    cmd:option('-epochNumber',     1,     'Manual epoch number (useful on restarts)')
    cmd:option('-batchSize',       128,   'mini-batch size (1 = pure stochastic)')
    ---------- Optimization options ----------------------
    cmd:option('-LR',    0.0, 'learning rate; if set, overrides default LR/WD recipe')
    cmd:option('-momentum',        0.9,  'momentum')
    cmd:option('-weightDecay',     5e-4, 'weight decay')
    ---------- Model options ----------------------------------
    cmd:option('-netType',     'vgg_modified', 'Options: alexnet | overfeat | alexnetowtbn | vgg | googlenet')
    -----------additional options -----------------------------
    cmd:option('-depth',        18,       'ResNet depth: 18 | 34 | 50 | 101 | ...', 'number')
    cmd:option('-shortcutType', '',       'Options: A | B | C')
    cmd:option('-shareGradInput',  'false', 'Share gradInput tensors to reduce memory usage')
    cmd:option('-optnet',          'false', 'Use optnet to reduce memory usage')
    cmd:option('-dataset',    'imagenet', 'Options: imagenet | cifar10 | cifar100')
    cmd:option('-cudnn',      'deterministic',  'Options: fastest | default | deterministic')

   ------------------------------------------------------------
    cmd:option('-retrain',     'none', 'provide path to model to retrain with')
    cmd:option('-optimState',  'none', 'provide path to an optimState to reload from')
    cmd:option('-wInit',       'xavier', 'Options kaiming | xavier ')    
    ---------- DropResNet options ----------------------------------
    cmd:option('-deathRate', 0.5, 'in stochastic layer resnet death rate of layers (0.5)')
    cmd:option('-deathMode', 'uniform', 'uniform or lin_decay (uniform)')
    
    cmd:text()

    local opt = cmd:parse(arg or {})
    -- add commandline specified options
    opt.save = paths.concat(opt.cache,
                            cmd:string(opt.netType, opt,
                                       {netType=true, retrain=true, optimState=true, cache=true, data=true}))
    -- add date/time
    opt.save = paths.concat(opt.save, '' .. os.date():gsub(' ',''))
    return opt
    
    
end

return M
