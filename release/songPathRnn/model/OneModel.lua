
--********************************************************************
--Source: https://github.com/rajarshd/ChainsofReasoning
--See Chains of Reasoning over Entities, Relations, and Text using Recurrent Neural Networks
--https://arxiv.org/abs/1607.01426
--**********************************************************************
package.path = package.path ..';../model/batcher/?.lua'
package.path = package.path ..';../model/net/?.lua'
package.path = package.path ..';../model/model/?.lua'
package.path = package.path ..';../model/module/?.lua'
package.path = package.path ..';../model/optimizer/?.lua'
package.path = package.path ..';../model/criterion/?.lua'
require 'torch'
require 'nn'
require 'optim'
require 'rnn'
--Dependencies from this package
require 'MyOptimizer'
require 'OptimizerCallback'
require 'BatcherFileList'
require 'FeatureEmbedding'
require 'MapReduce'
require 'TopK'
require 'Print'
require 'LogSumExp'

cmd = torch.CmdLine()
-- cmd:option('-trainList','','torch format train file list')
cmd:option('-dataDir','','path to the folder containing the dataset')
cmd:option('-minibatch',32,'minibatch size')
cmd:option('-testTimeMinibatch',32,'minibatch size')
cmd:option('-numRowsToGPU',1,'Num of rows to be loaded to GPU from each input file in the file list')
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:option('-lazyCuda',0,'put limited number of batches to gpu. 0 is false')

cmd:option('-relationVocabSize',51503,'vocabulary size of relations')
cmd:option('-entityTypeVocabSize',2267,'vocabulary size of entity types')
cmd:option('-entityVocabSize',1540261,'vocabulary size of entities')
cmd:option('-relationEmbeddingDim',50,'embedding dim for relations')
cmd:option('-entityTypeEmbeddingDim',50,'embedding dim for entity types')
cmd:option('-entityEmbeddingDim',50,'embedding dim for entity types')
cmd:option('-numFeatureTemplates',-1,'Number of feature templates')
cmd:option('-numEntityTypes',-1,'Number of entity types')

cmd:option('-learningRate',0.001,'init learning rate')
cmd:option('-learningRateDecay',0,'learning rate decay')
cmd:option('-tokenFeatures',1,'whether to embed features')
cmd:option('-evaluationFrequency',10,'how often to evaluation on test data')
cmd:option('-model',"",'where to save the model. If not specified, does not save')
cmd:option('-exptDir',"",'Output directory')
cmd:option('-initModel',"",'model checkpoint to initialize from')
cmd:option('-paramInit',0.1,'paramInit')
cmd:option('-startIteration',1,'Iteration number of starting iteration. Defaults to 1, but for example if you preload model-15, you want the next iteration to be 16')
cmd:option('-saveFrequency',50,'how often to save a model checkpoint')

cmd:option('-embeddingL2',0.0001,'extra l2 regularization term on the embedding weights')
cmd:option('-l2',0.0001,'l2 regularization term on all weights')

cmd:option('-architecture',"rnn",'only support rnn')
cmd:option('-numEpochs',300,'Number of epochs to run. Please note the way the framework is set up, it doesnt necessarily mean a pass over the data. Set batchesPerEpoch accordingly')
cmd:option('-batchesPerEpoch',500,'Number of minibatches in an epoch.')

--RNN-specific options
cmd:option('-rnnType',"rnn",'lstm or rnn')
cmd:option('-rnnDepth',1,'rnn depth')
cmd:option('-rnnHidSize',50,'rnn hidsize')
cmd:option('-useAdam',0,'use adam')
cmd:option('-epsilon',0.00000001,'epsilon for adam')
cmd:option('-useGradClip',1,'use gradClip')
cmd:option('-gradClipNorm',5,'gradClipNorm')
cmd:option('-gradientStepCounter',100,'gradientStepCounter')

--option for running just the beaseline model with relations and ignoring the entity types
cmd:option('-includeEntityTypes',1,'include entity types')
cmd:option('-includeEntity',-1,'include entity')

--consider top-k instead of maxpooling
cmd:option('-topK',0,'consider top K paths')
cmd:option('-K',5,'number of paths to consider')
cmd:option('-package_path','','package_path')
cmd:option('-createExptDir',1,'createExptDir')
cmd:option('-useReLU',1,'ReLU is 1; Sigmoid otherwise')
cmd:option('-rnnInitialization',1,'Initialize RNN')
cmd:option('-regularize',1,'Regularize?')
cmd:option('-numLayers',1,'num layers')
cmd:option('-useDropout',0,'to use dropout or not')
cmd:option('-dropout',0,'dropout rate')

local params = cmd:parse(arg)
local tokenFeatures = params.tokenFeatures
local debugMode = false
local isBinaryClassification = true
local numRowsToGPU = params.numRowsToGPU
local lazyCuda = params.lazyCuda == 1
local useCuda = params.gpuid ~= -1
local minibatch = params.minibatch
local dataDir = params.dataDir
local testList = params.testList
local relationVocabSize = params.relationVocabSize
local relationEmbeddingDim = params.relationEmbeddingDim
local  entityTypeVocabSize = params.entityTypeVocabSize
local entityTypeEmbeddingDim = params.entityTypeEmbeddingDim
local  entityVocabSize = params.entityVocabSize
local entityEmbeddingDim = params.entityEmbeddingDim
local numFeatureTemplates = params.numFeatureTemplates
local  numEntityTypes = params.numEntityTypes
assert(numEntityTypes <= numFeatureTemplates)
local rnnHidSize = params.rnnHidSize
local useAdam = params.useAdam == 1	
local  includeEntityTypes = params.includeEntityTypes == 1
local  includeEntity = params.includeEntity == 1
local isTopK = params.topK == 1
local k = params.K
local useGradClip = params.useGradClip == 1
local seed = 12345
local createExptDir = params.createExptDir == 1
local useReLU = params.useReLU == 1
local rnnInitialization = params.rnnInitialization == 1
local labelDimension = 46
local numLayers = params.numLayers
local useDropout = params.useDropout
local dropout = params.dropout
--torch.manualSeed(seed)

local exptDir = nil
local configFileName = nil
local configFile = nil
if createExptDir then
	exptDir = params.exptDir
	configFileName = exptDir..'/config.txt'
	configFile = io.open(configFileName,'w')
	configFile:write('gpu_id\t'..params.gpuid..'\n')
	configFile:write('model\t'..params.rnnType..'\n')
	configFile:write('rnnHidSize\t'..params.rnnHidSize..'\n')
	configFile:write('relationEmbeddingDim\t'..params.relationEmbeddingDim..'\n')
	configFile:write('entityTypeEmbeddingDim\t'..params.entityTypeEmbeddingDim..'\n')
	configFile:write('Learning Rate\t'..params.learningRate..'\n')
	configFile:write('Learning Rate Decay\t'..params.learningRateDecay..'\n')
	configFile:write('L2\t'..params.l2..'\n')
	configFile:write('Minibatch\t'..params.minibatch..'\n')
	configFile:write('numEpochs\t'..params.numEpochs..'\n')
	configFile:write('numFeatureTemplates\t'..params.numFeatureTemplates..'\n')
	configFile:write('isTopK\t'..params.topK..'\n')
	configFile:write('K\t'..params.K..'\n')
	configFile:write('epsilon\t'..params.epsilon..'\n')
	configFile:write('useGradClip\t'..params.useGradClip..'\n')
	configFile:write('gradClipNorm\t'..params.gradClipNorm..'\n')
	configFile:write('gradientStepCounter\t'..params.gradientStepCounter..'\n')
	configFile:write('paramInit\t'..params.paramInit..'\n')
	configFile:write('useReLU\t'..params.useReLU..'\n')
	if (includeEntityTypes) then
		configFile:write('includeEntityTypes\t'..'true'..'\n')	
	else
		configFile:write('includeEntityTypes\t'..'false'..'\n')	
	end
	if (includeEntity) then
		configFile:write('includeEntity\t'..'true'..'\n')	
	else
		configFile:write('includeEntity\t'..'false'..'\n')
	end
	configFile:write('numEntityTypes\t'..params.numEntityTypes..'\n')
	if params.initModel ~= "" then
		configFile:write('initModel\t'..params.initModel..'\n')
	end
	configFile:write('rnnInitialization\t'..params.rnnInitialization..'\n')
	configFile:write('Regularize\t'..params.regularize..'\n')
	configFile:write('numLayers\t'..params.numLayers..'\n')
	configFile:write('useDropout\t'..params.useDropout..'\n')
	configFile:write('Dropout\t'..params.dropout..'\n')
end

local gpuid = params.gpuid
if(useCuda or lazyCuda) then
    print('USING GPU')
    require 'cutorch'
    require('cunn')
    cutorch.setDevice(gpuid + 1) 
    cutorch.manualSeed(seed)
end

tokenprocessor = function (x) return x end
labelprocessor = function (x) return x end

if(tokenLabels or tokenFeatures)then
	preprocess = function(a,b,c) 
		return labelprocessor(a),tokenprocessor(b),c 
	end
end
local shuffle = true
local maxBatches = 100
local loadModel = params.initModel ~= ""
local predictor_net = nil
local embeddingLayer = nil 
local reducer = nil
local training_net = nil
local criterion = nil
local totalInputEmbeddingDim = 0
local input2hidden = nil
local hidden2hidden = nil
local input2hiddens = {}
local hidden2hiddens = {} -- table to store the params so that I can initialize them apppropriately.

-----Define the Architecture-----
if(not loadModel) then
	predictor_net = nn.Sequential()
	-- now create the embedding layer
	if includeEntityTypes and includeEntity then
		embeddingLayer = FeatureEmbedding:getEmbeddingNetworkBothEntitiesAndTypes(relationVocabSize, relationEmbeddingDim, entityVocabSize, entityEmbeddingDim, entityTypeVocabSize, entityTypeEmbeddingDim, numFeatureTemplates,numEntityTypes)
		totalInputEmbeddingDim = relationEmbeddingDim + entityTypeEmbeddingDim + entityEmbeddingDim
	elseif includeEntityTypes then
		embeddingLayer = FeatureEmbedding:getEmbeddingNetwork(relationVocabSize, relationEmbeddingDim, entityTypeVocabSize, entityTypeEmbeddingDim, numFeatureTemplates, numEntityTypes)
		totalInputEmbeddingDim = relationEmbeddingDim + entityTypeEmbeddingDim
	elseif includeEntity then
		embeddingLayer = FeatureEmbedding:getEmbeddingNetworkOnlyEntities(relationVocabSize, relationEmbeddingDim, entityVocabSize, entityEmbeddingDim, numFeatureTemplates)
		 totalInputEmbeddingDim = relationEmbeddingDim + entityEmbeddingDim
	else
		embeddingLayer = FeatureEmbedding:getRelationEmbeddingNetwork(relationVocabSize, relationEmbeddingDim)
		totalInputEmbeddingDim = relationEmbeddingDim
	end
	assert(embeddingLayer ~= nil)
	assert(totalInputEmbeddingDim ~= 0)

	predictor_net:add(nn.SplitTable(3)):add(embeddingLayer)
	local nonLinear = nil
	if useReLU then
		nonLinear = function() return nn.ReLU() end
	else
		nonLinear = function() return nn.Tanh() end
	end

	input2hidden = function() return nn.Linear(totalInputEmbeddingDim, rnnHidSize) end
	hidden2hidden = function() return nn.Linear(rnnHidSize, rnnHidSize) end

	
	if(params.rnnType == "lstm") then 
				rnn = function() return nn.FastLSTM(totalInputEmbeddingDim, rnnHidSize) end --todo: add depth
	elseif (params.rnnType == "gru") then
		rnn = function() return nn.GRU(totalInputEmbeddingDim, rnnHidSize) end 
	else
		rnn = function()
			-- recurrent module
			local  input2hidden = input2hidden()
			local hidden2hidden = hidden2hidden()
			table.insert(input2hiddens, input2hidden)
			table.insert(hidden2hiddens, hidden2hidden)
			if params.useDropout == 0 then
				local rm = nn.Sequential()
				   :add(nn.ParallelTable()
				      :add(input2hidden)
				      :add(hidden2hidden))
				   :add(nn.CAddTable())
				   :add(nonLinear())
				rm = nn.MaskZero(rm,1) --to take care of padding
				return nn.Recurrence(rm, rnnHidSize, 1)
			else
				local i2hNet = nn.Sequential():add(nn.Dropout(params.dropout)):add(input2hidden)
				-- local h2hNet = nn.Sequential():add(nn.Dropout()):add(hidden2hidden)
				local h2hNet = nn.Sequential():add(hidden2hidden)
				local par = nn.ParallelTable()
							:add(i2hNet)
							:add(h2hNet)
				local rm = nn.Sequential():add(par):add(nn.CAddTable()):add(nonLinear())
				rm = nn.MaskZero(rm,1) --to take care of padding
				return nn.Recurrence(rm, rnnHidSize, 1)
			end
		end
	end
	predictor_net:add(nn.SplitTable(2))
	
	for l=1,params.numLayers do
		rnn_layer = nn.Sequencer(rnn())
		predictor_net:add(rnn_layer)
	end
	predictor_net:add(nn.SelectTable(-1)) --select the last state
	predictor_net:add(nn.Linear(rnnHidSize,labelDimension))
	print(predictor_net)
else
	print('initializing model from '..params.initModel)
	local checkpoint = torch.load(params.initModel)
	predictor_net = checkpoint.predictor_net
	embeddingLayer = checkpoint.embeddingLayer
end

if params.topK == 1 then
	print('Reducer is topK')
	reducer = nn.Sequential():add(nn.TopK(k,2)):add(nn.Mean(2))
elseif params.topK == 2 then
	print('Reducer is LogSumExp')
	reducer = nn.Sequential():add(nn.LogSumExp(2)):add(nn.Squeeze(2))
else
	print('Reducer is max pool')
	reducer = nn.Max(2)
end
training_net = nn.Sequential():add(nn.MapReduce(predictor_net,reducer)):add(nn.Sigmoid())

-- test code here

-- end of test code
local classId = 1
criterion = nn.BCECriterion()
-- criterion = nn.MaskZeroCriterion(criterion,1)
if(useCuda or lazyCuda) then
	criterion:cuda()
	training_net:cuda()
end
if (not loadModel) then
	for k,param in ipairs(training_net:parameters()) do
	      param:uniform(-1*params.paramInit, params.paramInit)
	end
	if rnnInitialization then
		--initialize the recurrent matrix to identity and bias to zero
		for _,input2hidden in pairs(input2hiddens) do
			local params, gradParams = input2hidden:parameters()
			params[1]:copy(torch.eye(totalInputEmbeddingDim, rnnHidSize))
			params[2]:copy(torch.zeros(rnnHidSize))
		end
		for _,hidden2hidden in pairs(hidden2hiddens) do
			local params, gradParams = hidden2hidden:parameters()
			params[1]:copy(torch.eye(rnnHidSize))
			params[2]:copy(torch.zeros(rnnHidSize))
		end
	end
end

print("use cuda:",useCuda)
local trainBatcher = BatcherFileList(dataDir, minibatch, shuffle, maxBatches, useCuda, 'train.list')


--------Initialize Optimizer-------
local regularization = {
    l2 = {},
	params = {}
}

local embeddingL2 = params.embeddingL2
table.insert(regularization.l2,params.l2)
table.insert(regularization.params,embeddingLayer)


local momentum = 1.0
local dampening = 0.95
local beta1 = 0.9
local beta2 = 0.999
local epsilon = params.epsilon
local optConfig = {}
local optimMethod = nil
if(useAdam) then
	optimMethod = optim.adam
	print('Using Adam!')
	if createExptDir then	
		configFile:write('epsilon\t'..epsilon..'\n')
		configFile:write('beta1\t'..beta1..'\n')
		configFile:write('beta2\t'..beta2..'\n')
	end
	optConfig = {learningRate = params.learningRate,beta1 = beta1,beta2 = beta2,epsilon = epsilon}
else
	print('Using adagrad!')
	optimMethod = optim.adagrad
	optConfig = {learningRate = params.learningRate, learningRateDecay=params.learningRateDecay}
	
end
if createExptDir then	
	configFile:close()
end
optInfo = {
	optimMethod = optimMethod,
	optConfig = optConfig,
    optState = {},  
    regularization = regularization,
    cuda = useCuda,
    learningRate = params.learningRate,
    converged = false,
    startIteration = params.startIteration,
    entityTypePadToken = entityTypeVocabSize,
    relationPadToken = relationVocabSize,
    entityPadToken = entityVocabSize,
    gradClipNorm = params.gradClipNorm,
    gradientStepCounter = params.gradientStepCounter,
    useGradClip = useGradClip,
    l2 = params.l2,
    createExptDir = createExptDir,
    regularize = params.regularize
    -- recurrence = recur
}
--------Callbacks-------
callbacks = {}
local evaluator = nil
-- evaluator = BinaryEvaluation(testBatcher,training_net,exptDir)
-- local evaluationCallback = OptimizerCallback(params.evaluationFrequency,function(i) evaluator:evaluate(i) end,'evaluation')
-- table.insert(callbacks,evaluationCallback)

if(params.model  ~= "") then
	local saver = function(i) 
		local file = params.model.."-".."latest"
		print('saving to '..file)
		local toSave = {
			embeddingLayer = embeddingLayer,
			predictor_net = predictor_net,
		}
		torch.save(file,toSave)
	end
	if createExptDir then
		local savingCallback = OptimizerCallback(params.saveFrequency,saver,'saving')
		table.insert(callbacks,savingCallback)	
	else
		print('WARNING! - createExptDir is NOT set!')
	end
end
--------Training Options-------
local trainingOptions = {
    numEpochs = params.numEpochs,
    epochHooks = callbacks,
    minibatchsize = params.minibatch,
}
-----------------------------------	
params.learningRate = params.pretrainLearningRate
optimizer = MyOptimizer(training_net,training_net,criterion,trainingOptions,optInfo,params.rnnType) 
optimizer:train(trainBatcher)
