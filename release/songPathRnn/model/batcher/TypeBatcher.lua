--********************************************************************
--Source: https://github.com/rajarshd/ChainsofReasoning
--See Chains of Reasoning over Entities, Relations, and Text using Recurrent Neural Networks
--https://arxiv.org/abs/1607.01426
--**********************************************************************
local TypeBatcher = torch.class('TypeBatcher')

function TypeBatcher:__init(fileName, batchSize, shuffle, genNeg, vocabSize)
	print('Loading data from '..fileName)
	local loadedData = torch.load(fileName)
	self.entities = loadedData.entities
	self.types = loadedData.types
	self.batchSize = batchSize
	self.doShuffle = shuffle or false
	self.curStart = 1
	self.dataSize = self.entities:size(1)
	self.genNeg = genNeg
	self.vocabSize = vocabSize
	self:shuffle()
end

function TypeBatcher:shuffle()
	if self.doShuffle then
		local inds = torch.randperm(self.types:size(1)):long()
		self.entities = self.entities:index(1,inds)
		self.types = self.types:index(1,inds)
	end
end

function TypeBatcher:genNegExamples(posEntities)
	
	if posEntities == nil then return nil end
	local negCount = posEntities:size(1) --number of positive examples
	local negBatch = torch.rand(negCount):mul(self.vocabSize):floor():add(1):view(posEntities:size())
	return negBatch	
end

function TypeBatcher:get_batch(batcher, vocabSize)
    local pos_entities, types = batcher:getBatch()
    if pos_entities == nil then return nil end
    local neg_entities = gen_neg(pos_entities, vocabSize)
    print(neg_entities:size())
    return {pos_entities, types, neg_entities}
end

function TypeBatcher:getBatch()
	local startIndex = self.curStart
	if startIndex > self.dataSize then return nil end
	local endIndex = math.min(startIndex+self.batchSize-1, self.dataSize)
	local currBatchSize = endIndex - startIndex + 1
	local batchEntities = self.entities:narrow(1, startIndex, currBatchSize)
	local batchTypes = self.types:narrow(1, startIndex, currBatchSize)
	local batchNegEntities = nil
	if self.genNeg then batchNegEntities = self:genNegExamples(batchEntities) end
	self.curStart = endIndex + 1
	
	return {batchEntities, batchTypes, batchNegEntities}
end

function TypeBatcher:reset()
	self.curStart = 1
	if self.doShuffle then self:shuffle() end
end