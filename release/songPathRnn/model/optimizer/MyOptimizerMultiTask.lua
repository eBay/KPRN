--********************************************************************
--Source: https://github.com/rajarshd/ChainsofReasoning
--See Chains of Reasoning over Entities, Relations, and Text using Recurrent Neural Networks
--https://arxiv.org/abs/1607.01426
--**********************************************************************
require 'FeatureEmbedding'
require 'os'

local MyOptimizerMultiTask = torch.class('MyOptimizerMultiTask')


--NOTE: various bits of this code were inspired by fbnn Optim.lua 3/5/2015
function MyOptimizerMultiTask:__init(model, modules_to_update, typeNet, criterion, typeCriterion, trainingOptions, optInfo, rnnType, colEncoder, entityTypeLookupTable)
    assert(trainingOptions)
    assert(optInfo)
    self.structured = structured or false
    self.model = model
    self.typeNet = typeNet
    self.typeCriterion = typeCriterion
    self.origModel = model
    self.optState = optInfo.optState   
    self.optConfig = optInfo.optConfig
    self.optimMethod = optInfo.optimMethod
    self.regularization = optInfo.regularization
    self.trainingOptions = trainingOptions 
    self.totalError = torch.Tensor(1):zero()
    self.checkForConvergence = optInfo.converged ~= nil
    self.startIteration = optInfo.startIteration
    self.optInfo = optInfo
    self.minibatchsize = trainingOptions.minibatchsize
    self.rnnType = rnnType
    self.entityTypePadToken = optInfo.entityTypePadToken
    self.relationPadToken = optInfo.relationPadToken
    self.entityPadToken = optInfo.entityPadToken
    self.gradClipNorm = optInfo.gradClipNorm
    self.gradientStepCounter = optInfo.gradientStepCounter
    self.useGradClip = optInfo.useGradClip
    self.l2 = optInfo.l2
    self.typeL2 = optInfo.typeL2
    self.regularize = optInfo.regularize
    self.createExptDir = optInfo.createExptDir
    self.typeRegularize = optInfo.typeRegularize
    self.typeGradClip = optInfo.typeGradClip
    self.typeGradClipNorm = optInfo.typeGradClipNorm
    self.totalTypeError = torch.Tensor(1):zero()
    self.typeOptConfig = optInfo.typeOptConfig
    self.typeOptState = optInfo.typeOptState
    self.typeOptimMethod = optInfo.typeOptimMethod
    self.saveFileNameTypes = optInfo.saveFileNameTypes
    local parameters
    local gradParameters
    -- so to share params across the net you have to do make them a combined network
    -- https://groups.google.com/forum/#!topic/torch7/_AW4T98-t0s
    local combined_net = nn.Sequential():add(self.model):add(self.typeNet)
    parameters, gradParameters = combined_net:getParameters()
    self.parameters = parameters
    self.gradParameters = gradParameters
   
    self.cuda = optInfo.cuda
     if(optInfo.useCuda) then
        self.totalError:cuda()
        self.totalTypeError:cuda()
    end

     self.criterion = criterion
    for hookIdx = 1,#self.trainingOptions.epochHooks do
        local hook = self.trainingOptions.epochHooks[hookIdx]
        if( hook.epochHookFreq == 1) then
            hook.hook(0)
        end
    end

end
function MyOptimizerMultiTask:toCuda(x) return self.useCuda and x:cuda() or x:double() end

function MyOptimizerMultiTask:zeroPadTokens()

    local entityTypeLookUpTable = FeatureEmbedding:getEntityTypeLookUpTable()
    if entityTypeLookUpTable ~= nil then
        local params, gradParams = entityTypeLookUpTable:parameters()
        params[1][self.entityTypePadToken]:zero()
    end

    local relationLookUpTable = FeatureEmbedding:getRelationLookUpTable()
    if relationLookUpTable ~= nil then
        local params, gradParams = relationLookUpTable:parameters()
        params[1][self.relationPadToken]:zero()
    end

    local entityLookUpTable = FeatureEmbedding:getEntityLookUpTable()
    if entityLookUpTable ~= nil then
        local params, gradParams = entityLookUpTable:parameters()
        params[1][self.entityPadToken]:zero()
    end
end

function MyOptimizerMultiTask:train(trainBatcher, typeBatcher)
    
    self.model:training() -- turn on the training flag; especially imp when using dropout
    self.typeNet:training()
    local prevTime = sys.clock()
    local numProcessed = 0
    --count the total number of batches once. This is for displpaying the progress bar; helps to track time
    local totalBatches = 0 
    local totalTypeBatches = 0
    print('Making a pass of the relations data to count the batches')
    while(true) 
    do
        local minibatch_targets,minibatch_inputs,num, classId = trainBatcher:getBatch()
        if minibatch_targets == nil then
            break --end of a batch
        end
        totalBatches = totalBatches + 1
    end
    print('Total num batches for relation extraction tasks'..totalBatches)
    print('Making a pass of the types data to count the batches')
    while(true) 
    do
        local ret = typeBatcher:getBatch()
        if ret == nil then break  end
        totalTypeBatches = totalTypeBatches + 1
    end
    print('Total number of batches ')
    trainBatcher:reset()
    typeBatcher:reset()
    local p = 0
    if totalBatches > totalTypeBatches then
        p = totalTypeBatches / totalBatches
    else
        p = totalBatches / totalTypeBatches
    end
    print('Total batches for relation prediction '..totalBatches)
    print('Total batches for type prediction '..totalTypeBatches)
    print('p = '..p)
    local numEpochsRelationPred = self.trainingOptions.numEpochs
    local numEpochsTypePred = self.trainingOptions.typeNumEpochs
    local numEpochs = math.min(numEpochsRelationPred, numEpochsTypePred)
    print('Number of iterations to train both '..numEpochs)
    local i = 1
    while i <= numEpochs do
        local  countBatches = 1
        local countTypeBatches = 1
        self.totalError:zero()
        self.totalTypeError:zero()
        while(true) do
            rand = torch.bernoulli(p)
            if rand == 1 then
                --do training for type prediction
                if countTypeBatches <= totalTypeBatches then
                    -- print('countTypeBatches '..countTypeBatches)
                    local ret = typeBatcher:getBatch()
                    pos, types, neg = self:toCuda(ret[1]), self:toCuda(ret[2]), self:toCuda(ret[3])
                    local data = {pos, types, neg}
                    local labels = self:toCuda(torch.ones(pos:size())) -- its a dummy label, since in BPR criterion there isnt any label
                    local batch = {data = data,labels = labels}
                    self:trainTypeBatch(batch)
                    countTypeBatches = countTypeBatches + 1
                    -- xlua.progress(countTypeBatches, totalTypeBatches)
                else --type prediction training is over
                    if countBatches <= totalBatches then
                        --but relation prediction is still left
                        p = 0 --make p =0, so it doesnt come here anymore
                    else
                        --both are over; time to break out of the loop
                        break
                    end
                end
            else
                -- do training for relation prediction
                if countBatches <= totalBatches then
                    local minibatch_targets,minibatch_inputs,num, classId = trainBatcher:getBatch()
                    self.model = nn.Sequential():add(self.origModel):add(nn.Select(2,classId)):cuda()
                    if minibatch_targets == nil then
                        break --end of a batch
                    end
                    if(minibatch_targets) then
                        numProcessed = numProcessed + minibatch_targets:nElement() --this reports the number of 'training examples.' If doing sequence tagging, it's the number of total timesteps, not the number of sequences. 
                    else
                        --in some cases, the targets are actually part of the inputs with some weird table structure. Need to account for this.
                        numProcessed = numProcessed + self.minibatchsize
                    end
                    self:trainBatch(minibatch_inputs,minibatch_targets)
                    countBatches = countBatches + 1
                    xlua.progress(countBatches, totalBatches)
                else
                    if countTypeBatches <= totalTypeBatches then
                        p = 1
                    else
                        --both are over; time to break out of the loop
                        break
                    end
                end
            end
        end
        local avgError = self.totalError[1]/totalBatches
        local avgErrorTypes = self.totalTypeError[1]/totalTypeBatches
        local currTime = sys.clock()
        local ElapsedTime = currTime - prevTime
        prevTime = currTime
        print(string.format('\nIter: %d\navg loss in epoch = %f\navg loss in epoch (type_prediction) = %f\ntotal elapsed = %f\ntime per batch = %f',i,avgError, avgErrorTypes, ElapsedTime,ElapsedTime/totalBatches))
        for hookIdx = 1,#self.trainingOptions.epochHooks do
            local hook = self.trainingOptions.epochHooks[hookIdx]
            if( i % hook.epochHookFreq == 0) then
                hook.hook(i)
            end
        end
        if(self.createExptDir) then
            if i%1 == 0 then
                local file = self.saveFileNameTypes..i
                print('Saving to '..file)
                torch.save(file,self.typeNet)
            end
        end
        trainBatcher:reset()
        typeBatcher:reset()
        i = i + 1
    end
    if  numEpochsTypePred > numEpochs then
        print('Now training only for type prediction')
        self:trainTypePrediction(typeBatcher, i)
    else
        print('Now training only for relation prediction')
        self:trainRelationPrediction(trainBatcher, i)
    end
end

function MyOptimizerMultiTask:trainTypePrediction(typeBatcher, startIter)
    
    self.typeNet:training()
    local prevTime = sys.clock()
    local numProcessed = 0
    --count the total number of batches once. This is for displpaying the progress bar; helps to track time
    local totalBatches = 0
    print('Making a pass of the data to count the batches')
    while(true) 
    do
        local ret = typeBatcher:getBatch()
        if ret == nil then break  end
        totalBatches = totalBatches + 1
    end
    print('Total num batches '..totalBatches)
    local i = startIter
    while i <= self.trainingOptions.typeNumEpochs do
        typeBatcher:reset()
        self.totalTypeError:zero()
        local batch_counter = 0
        while(true) do
            local ret = typeBatcher:getBatch()
            if ret == nil then break end
            pos, types, neg = self:toCuda(ret[1]), self:toCuda(ret[2]), self:toCuda(ret[3])
            local data = {pos, types, neg}
            local labels = self:toCuda(torch.ones(pos:size())) -- its a dummy label, since in BPR criterion there isnt any label
            local batch = {data = data,labels = labels}
            self:trainTypeBatch(batch)
            batch_counter = batch_counter + 1
            xlua.progress(batch_counter, totalBatches)
        end
        local avgError = self.totalTypeError[1]/batch_counter
        local currTime = sys.clock()
        local ElapsedTime = currTime - prevTime
        local rate = numProcessed/ElapsedTime
        numProcessed = 0
        prevTime = currTime
        print(string.format('\nIter: %d\navg loss in epoch = %f\ntotal elapsed = %f\ntime per batch = %f',i,avgError, ElapsedTime,ElapsedTime/batch_counter))
        print(string.format('examples/sec = %f',rate))
        if (self.createExptDir) then
            if i%1 == 0 then
                local file = self.saveFileNameTypes..i
                print('Saving to '..file)
                torch.save(file,self.typeNet)
            end
        end
        i = i + 1
    end
end

function MyOptimizerMultiTask:trainRelationPrediction(trainBatcher, startIter)
    
    self.model:training() -- turn on the training flag; especially imp when using dropout
    local prevTime = sys.clock()
    local numProcessed = 0
    --count the total number of batches once. This is for displpaying the progress bar; helps to track time
    local totalBatches = 0 -- I computed this manually. This is for all the dataset combined
    print('Making a pass of the data to count the batches')
    while(true) 
    do
        local minibatch_targets,minibatch_inputs,num, classId = trainBatcher:getBatch()
        if minibatch_targets == nil then
            break --end of a batch
        end
        totalBatches = totalBatches + 1
    end
    print('Total num batches '..totalBatches)
    trainBatcher:reset()
    local i = startIter
    while i <= self.trainingOptions.numEpochs do
        self.totalError:zero()
        num_data = 0
        batch_counter = 0
        local gradientStepCounter = 0
        count = 0
        while(true) do
            local minibatch_targets,minibatch_inputs,num, classId = trainBatcher:getBatch()
            self.model = nn.Sequential():add(self.origModel):add(nn.Select(2,classId)):cuda()
            if minibatch_targets == nil then
                break --end of a batch
            end
            batch_counter = batch_counter + 1
            if(minibatch_targets) then
                numProcessed = numProcessed + minibatch_targets:nElement() --this reports the number of 'training examples.' If doing sequence tagging, it's the number of total timesteps, not the number of sequences. 
            else
                --in some cases, the targets are actually part of the inputs with some weird table structure. Need to account for this.
                numProcessed = numProcessed + self.minibatchsize
            end
            self:trainBatch(minibatch_inputs,minibatch_targets)
            gradientStepCounter = gradientStepCounter + 1
            if(gradientStepCounter % self.gradientStepCounter == 0) then
                local avgError = self.totalError[1]/gradientStepCounter
                print(string.format('Printing after %d gradient steps\navg loss in epoch = %f\n',self.gradientStepCounter, avgError))     
            end
            count = count + 1
            xlua.progress(count, totalBatches)
        end
        local avgError = self.totalError[1]/batch_counter
        local currTime = sys.clock()
        local ElapsedTime = currTime - prevTime
        local rate = numProcessed/ElapsedTime
        numProcessed = 0
        prevTime = currTime
        print(string.format('\nIter: %d\navg loss in epoch = %f\ntotal elapsed = %f\ntime per batch = %f',i,avgError, ElapsedTime,ElapsedTime/batch_counter))
        --print(string.format('cur learning rate = %f',self.optConfig.learningRate))
        print(string.format('examples/sec = %f',rate))
        if(not self.createExptDir) then
            print('WARNING! - createExptDir is NOT set!')
        end
        self:postEpoch()
        for hookIdx = 1,#self.trainingOptions.epochHooks do
            local hook = self.trainingOptions.epochHooks[hookIdx]
            if( i % hook.epochHookFreq == 0) then
                hook.hook(i)
            end
       end
        trainBatcher:reset()
        i = i + 1
    end
end

function MyOptimizerMultiTask:trainBatch(inputs, targets)
    assert(inputs)
    assert(targets)
    --print(targets)
    self:zeroPadTokens()
    local parameters = self.parameters
    local gradParameters = self.gradParameters
    local function fEval(x)
        if parameters ~= x then parameters:copy(x) end
        self.model:zeroGradParameters()
        local output = self.model:forward(inputs)
        local df_do = nil
        local err = nil        
        err = self.criterion:forward(output, targets)
        df_do = self.criterion:backward(output, targets)
        self.model:backward(inputs, df_do)
        if self.regularize == 1 then
            if self.useGradClip then
                local norm = gradParameters:norm()
                if(norm > self.gradClipNorm) then
                  gradParameters:mul(self.gradClipNorm/norm)
                end
                --Also do L2 after clipping
                gradParameters:add(self.l2, parameters)
            else
                gradParameters:add(self.l2, parameters)
            end
        end
        self.totalError[1] = self.totalError[1] + err
        return err, gradParameters
    end
    self.optimMethod(fEval, parameters, self.optConfig, self.optState)
    self:zeroPadTokens()
    return err
end

function MyOptimizerMultiTask:trainTypeBatch(batch)
    -- body
    assert(batch)
    -- local parameters = self.typeParams
    -- local gradParameters = self.typeGradParams
    local parameters = self.parameters
    local gradParameters = self.gradParameters
    local function fEval(x)
        if parameters ~= x then parameters:copy(x) end
        self.typeNet:zeroGradParameters()
        local data = batch.data
        local labels = batch.labels
        local  pred = self.typeNet:forward(data)
        local err = self.typeCriterion:forward(pred, labels)
        local df_do = self.typeCriterion:backward(pred, labels)
        self.typeNet:backward(data, df_do)
        if self.typeRegularize == 1 then
            if self.typeGradClip == 1 then
                local norm = gradParameters:norm()
                if norm > self.typeGradClipNorm then
                    gradParameters:mul(self.typeGradClipNorm/norm)
                end
                --Also do L2 after clipping
                gradParameters:add(self.typeL2, parameters)
            else
                gradParameters:add(self.typeL2, parameters)
            end
        end
        self.totalTypeError[1] = self.totalTypeError[1] + err
        return err, gradParameters
    end
    self.typeOptimMethod(fEval, parameters, self.typeOptConfig, self.typeOptState)
    return err
end