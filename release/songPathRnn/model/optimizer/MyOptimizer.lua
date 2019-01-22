--********************************************************************
--Source: https://github.com/rajarshd/ChainsofReasoning
--See Chains of Reasoning over Entities, Relations, and Text using Recurrent Neural Networks
--https://arxiv.org/abs/1607.01426
--**********************************************************************
require 'FeatureEmbedding'
require 'os'

local MyOptimizer = torch.class('MyOptimizer')


--NOTE: various bits of this code were inspired by fbnn Optim.lua 3/5/2015
function MyOptimizer:__init(model,modules_to_update,criterion, trainingOptions,optInfo,rnnType)
     assert(trainingOptions)
     assert(optInfo)
     self.structured = structured or false
     self.model = model
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
     self.regularize = optInfo.regularize
     self.createExptDir = optInfo.createExptDir
     -- self.recurrence = optInfo.recurrence
    local parameters
    local gradParameters
    parameters, gradParameters = modules_to_update:getParameters()
    self.parameters = parameters
    self.gradParameters = gradParameters

    self.l2s = {}
    self.params = {}
    self.grads = {}
    for i = 1,#self.regularization.params do
            local params,grad = self.regularization.params[i]:parameters()
            local l2 = self.regularization.l2[i]
            table.insert(self.params,params)
            table.insert(self.grads,grad)
            table.insert(self.l2s,l2)
    end
    self.numRegularizers = #self.l2s


    self.cuda = optInfo.cuda
     if(optInfo.useCuda) then
        self.totalError:cuda()
    end

     self.criterion = criterion
    for hookIdx = 1,#self.trainingOptions.epochHooks do
        local hook = self.trainingOptions.epochHooks[hookIdx]
        if( hook.epochHookFreq == 1) then
            hook.hook(0)
        end
    end

end

function MyOptimizer:zeroPadTokens()

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

function MyOptimizer:train(trainBatcher)
    
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
    local i = self.startIteration
    while i <= self.trainingOptions.numEpochs and (not self.checkForConvergence or not self.optInfo.converged) do
        self.totalError:zero()
        num_data = 0
        batch_counter = 0
        local gradientStepCounter = 0
        count = 0
        while(true) do
            local minibatch_targets,minibatch_inputs,num, classId = trainBatcher:getBatch()
            if self.cuda then
                self.model = nn.Sequential():add(self.origModel):add(nn.Select(2,classId)):cuda()
            else
--                print("opitimizer:",classId)
                self.model = nn.Sequential():add(self.origModel):add(nn.Select(2,classId))
            end
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

function MyOptimizer:postEpoch()
    --this is to be overriden by children of MyOptimizer
end

-- function zeroPadTokenEmbeddings()

function MyOptimizer:trainBatch(inputs, targets)
    assert(inputs)
    assert(targets)
    --print(targets)
    self:zeroPadTokens()
    local parameters = self.parameters
    local gradParameters = self.gradParameters
    local function fEval(x)
        if parameters ~= x then parameters:copy(x) end
        self.model:zeroGradParameters()
--        print("before forward")
--        print(inputs)
        local output = self.model:forward(inputs)
--        print("after forward")
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
                -- for i = 1,self.numRegularizers do
                --     local l2 = self.l2s[i]
                --     for j = 1,#self.params[i] do
                --         self.grads[i][j]:add(l2,self.params[i][j])
                --     end
                -- end
                --  gradParameters:clamp(-5,5)
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