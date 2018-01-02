





--na ho paayi






--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

local batchNumber
local top1_center, loss
local timer = torch.Timer()



local function computeScore(output, target, nCrops)
  
    if nCrops > 1 then
      -- Sum over crops
      output = output:view(output:size(1) / nCrops, nCrops, output:size(2))
         --:exp()
         :sum(2):squeeze(2)
   end
   -- Coputes the top1 and top5 error rate
   local batchSize = output:size(1)

   _ , predictions = output:float():sort(2, true) -- descending
    local correct = predictions:eq(
      target:long():view(batchSize, 1):expandAs(predictions))
      local top1 = 1.0 * batchSize - (correct:narrow(2, 1, 1):sum())
      return top1,batchSize
   -- Top-1 score
   
end








function test()
   print('==> doing epoch on validation data:')
   print("==> online epoch # " .. epoch)

   batchNumber = 0
   cutorch.synchronize()
   timer:reset()

   -- set the dropouts to evaluate mode
   model:evaluate()

   top1_center = 0
   loss = 0
   for i = 1, math.ceil(nTest/opt.batchSize) do -- nTest is set in 1_data.lua
      local indexStart = (i-1) * opt.batchSize + 1
      local indexEnd = math.min(nTest, indexStart + opt.batchSize - 1)
      donkeys:addjob(
         -- work to be done by donkey thread
         function()
            local inputs, labels = testLoader:get(indexStart, indexEnd)
            return inputs, labels
         end,
         -- callback that is run in the main thread once the work is done
         testBatch
      )
   end

   donkeys:synchronize()
   cutorch.synchronize()

   top1_center = top1_center * 100 / nTest
   loss = loss / nTest -- because loss is calculated per batch
   testLogger:add{
      ['% top1 accuracy (test set) (center crop)'] = top1_center,
      ['avg loss (test set)'] = loss
   }
   print(string.format('Epoch: [%d][TESTING SUMMARY] Total Time(s): %.2f \t'
                          .. 'average loss (per batch): %.2f \t '
                          .. 'accuracy [Center](%%):\t top-1 %.2f\t ',
                       epoch, timer:time().real, loss, top1_center))

   print('\n')
   file = io.open("/output/test_results.txt", "a")
    file_loss = io.open("/output/test_loss_results.txt","a")
    -- appends a word test to the last line of the file
    file:write(top1_center.."\n")
    file_loss:write(loss.."\n")
    -- closes the open file
    file:close()
    file_loss:close()


end -- of test()
-----------------------------------------------------------------------------
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

function testBatch(inputsCPU, labelsCPU)
   batchNumber = batchNumber + opt.batchSize

   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   labels:resize(labelsCPU:size()):copy(labelsCPU)

   local outputs = model:forward(inputs)
   local err = criterion:forward(outputs, labels)
   cutorch.synchronize()
   local pred = outputs:float()

   loss = loss + err * outputs:size(1)
   
   
   --local _, pred_sorted = pred:sort(2, true)
   top1,batchSize = computeScore(outputs, labels, 1)
   top1_center = top1_center + top1
   
   if batchNumber % 1024 == 0 then
      print(('Epoch: Testing [%d][%d/%d]'):format(epoch, batchNumber, nTest))
   end
end
