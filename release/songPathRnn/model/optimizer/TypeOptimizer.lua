--********************************************************************
--Source: https://github.com/rajarshd/ChainsofReasoning
--See Chains of Reasoning over Entities, Relations, and Text using Recurrent Neural Networks
--https://arxiv.org/abs/1607.01426
--**********************************************************************
local TypeOptimizer = torch.class('TypeOptimizer')


function TypeOptimizer:__init(type_net, type_criterion, type_opt_config)

	self.typeNet = type_net
	self.typeCriterion = type_criterion
	self.typeRegularize = type_opt_config.typeRegularize
	self.typeGradClip = type_opt_config.typeGradClip
	self.typeParameters, self.typeGradParameters = type_net:getParameters()
	self.typeGradClipNorm = type_opt_config.typeGradClipNorm
	self.typeL2 = type_opt_config.typeL2
	self.numEpochs = type_opt_config.numEpochs
	self.useCuda = type_opt_config.useCuda
	self.totalTypeError = torch.Tensor(1):zero()
	self.optConfig = type_opt_config.optConfig
	self.optState = type_opt_config.optState
	self.typeOptimMethod = type_opt_config.optimMethod
	self.saveFileName = type_opt_config.saveFileName
end

function TypeOptimizer:toCuda(x) return self.useCuda and x:cuda() or x:double() end

function TypeOptimizer:train(typeBatcher)
	
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
    local i = 1
    while i <= self.numEpochs do
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
        	self:trainBatch(batch)
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
        if i%1 == 0 then
	        local file = self.saveFileName.."-"..i
	        print('Saving to '..file)
	        torch.save(file,self.typeNet:clone():float())
    	end
    	i = i + 1
    end
end

function TypeOptimizer:trainBatch(batch)
	-- body
	assert(batch)
	local parameters = self.typeParameters
    local gradParameters = self.typeGradParameters
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
	self.typeOptimMethod(fEval, parameters, self.optConfig, self.optState)
	return err
end
