--********************************************************************
--Source: https://github.com/rajarshd/ChainsofReasoning
--See Chains of Reasoning over Entities, Relations, and Text using Recurrent Neural Networks
--https://arxiv.org/abs/1607.01426
--**********************************************************************
-- package.path = package.path ..';/home/rajarshi/EMNLP/LSTM-KBC/model/?.lua'
require "ConcatTableNoGrad"
-- require "LookupTableWithGrad"
-- require "SplitTableNoGrad"
-- require "Sum_nc"

local FeatureEmbedding = torch.class('FeatureEmbedding')

function FeatureEmbedding:getEntityTypeLookUpTable()
	return self.entityTypeLookUpTable
end

function FeatureEmbedding:getRelationLookUpTable()
	return self.relationLookUpTable
end

function FeatureEmbedding:getEntityLookUpTable()
	return self.entityLookUpTable
end

function FeatureEmbedding:getRelationEmbeddingNetwork(relationVocabSize, relationEmbeddingDim)

	local network = nn.Sequential()
	local relationLookUpTable = nn.LookupTable(relationVocabSize,relationEmbeddingDim)
	self.relationLookUpTable = relationLookUpTable
	return network:add(nn.SelectTable(-1)):add(relationLookUpTable)
	-- return network:add(nn.LookupTable(relationVocabSize,relationEmbeddingDim))

end

function FeatureEmbedding:getEntityTypeEmbeddingNetwork(entityTypeVocabSize, entityTypeEmbeddingDim, numFeatureTemplates, numEntityTypes, isEntityPresent)

	local network = nn.Sequential()
	--create the parallel table network
	local par = nn.ParallelTable()
	local lookupTable1 = nn.LookupTable(entityTypeVocabSize, entityTypeEmbeddingDim)
	self.entityTypeLookUpTable = lookupTable1
	par:add(lookupTable1)
	
	for i = 1,numEntityTypes-1 -- -1 because we have already added for the first feature and one of tne feature is relation which will be added later
	do 
		local  lookupTable = nn.LookupTable(entityTypeVocabSize, entityTypeEmbeddingDim)
		lookupTable:share(lookupTable1,'weight', 'gradWeight', 'bias', 'gradBias')
		par:add(lookupTable)
	end
	local startIndex = numFeatureTemplates - numEntityTypes - 1 -- -1 because the last and the second last are entity and relation
	-- if isEntityPresent then
	-- 	startIndex = startIndex - 1 --because the second last position is for entities
	-- end
	return network:add(nn.NarrowTable(startIndex,numEntityTypes)):add(par):add(nn.CAddTable())--:add(nn.MulConstant(1/numEntityTypes))
end


function FeatureEmbedding:getEntityTypeEmbeddingNetworkFeedFwd(typeLookupTable, entityTypeVocabSize, entityTypeEmbeddingDim, numFeatureTemplates, numEntityTypes, isEntityPresent)

	local network = nn.Sequential()
	--create the parallel table network
	local par = nn.ParallelTable()
	-- local lookupTable1 = nn.LookupTable(entityTypeVocabSize, entityTypeEmbeddingDim)
	local lookupTable1 = typeLookupTable
	-- self.entityTypeLookUpTable = lookupTable1
	par:add(lookupTable1)
	
	for i = 1,numEntityTypes-1 -- -1 because we have already added for the first feature and one of tne feature is relation which will be added later
	do 
		local  lookupTable = nn.LookupTable(entityTypeVocabSize, entityTypeEmbeddingDim)
		lookupTable:share(lookupTable1,'weight', 'gradWeight', 'bias', 'gradBias')
		par:add(lookupTable)
	end
	local startIndex = numFeatureTemplates - numEntityTypes - 1 -- -1 because the last and the second last are entity and relation
	-- if isEntityPresent then
	-- 	startIndex = startIndex - 1 --because the second last position is for entities
	-- end
	return network:add(nn.SplitTableNoGrad(3)):add(nn.NarrowTable(startIndex,numEntityTypes)):add(par):add(nn.CAddTable())--:add(nn.MulConstant(1/numEntityTypes))
end


function FeatureEmbedding:getEntityEmbeddingNetwork(entityVocabSize, entityEmbeddingDim, numFeatureTemplates)

	local network = nn.Sequential()
	entityLookUpTable = nn.LookupTable(entityVocabSize,entityEmbeddingDim)
	self.entityLookUpTable = entityLookUpTable
	return network:add(nn.SelectTable(numFeatureTemplates-1)):add(entityLookUpTable)
end

function FeatureEmbedding:getEmbeddingNetwork(relationVocabSize, relationEmbeddingDim, entityTypeVocabSize, entityTypeEmbeddingDim, numFeatureTemplates, numEntityTypes)

	local  cat = nn.ConcatTableNoGrad()
	local relationEmbeddingNetwork = self:getRelationEmbeddingNetwork(relationVocabSize, relationEmbeddingDim)
	local entityTypeEmbeddingNetwork = self:getEntityTypeEmbeddingNetwork(entityTypeVocabSize, entityTypeEmbeddingDim, numFeatureTemplates, numEntityTypes, false)

	cat:add(entityTypeEmbeddingNetwork):add(relationEmbeddingNetwork)
	return nn.Sequential():add(cat):add(nn.JoinTable(3))
	-- return relationEmbeddingNetwork
end

function FeatureEmbedding:getEmbeddingNetworkOnlyEntities(relationVocabSize, relationEmbeddingDim, entityVocabSize, entityEmbeddingDim, numFeatureTemplates)

	local  cat = nn.ConcatTableNoGrad()
	local relationEmbeddingNetwork = self:getRelationEmbeddingNetwork(relationVocabSize, relationEmbeddingDim)
	local entityEmbeddingNetwork = self:getEntityEmbeddingNetwork(entityVocabSize, entityEmbeddingDim, numFeatureTemplates)
	cat:add(entityEmbeddingNetwork):add(relationEmbeddingNetwork)
	return nn.Sequential():add(cat):add(nn.JoinTable(3))
	-- return relationEmbeddingNetwork
end

function FeatureEmbedding:getEmbeddingNetworkBothEntitiesAndTypes(relationVocabSize, relationEmbeddingDim, entityVocabSize, entityEmbeddingDim, entityTypeVocabSize, entityTypeEmbeddingDim, numFeatureTemplates,numEntityTypes)

	local  cat = nn.ConcatTableNoGrad()
	local relationEmbeddingNetwork = self:getRelationEmbeddingNetwork(relationVocabSize, relationEmbeddingDim)
	local entityEmbeddingNetwork = self:getEntityEmbeddingNetwork(entityVocabSize, entityEmbeddingDim, numFeatureTemplates)
	local entityTypeEmbeddingNetwork = self:getEntityTypeEmbeddingNetwork(entityTypeVocabSize, entityTypeEmbeddingDim, numFeatureTemplates, numEntityTypes, true)
	cat:add(entityTypeEmbeddingNetwork):add(entityEmbeddingNetwork):add(relationEmbeddingNetwork)
	return nn.Sequential():add(cat):add(nn.JoinTable(3))
	-- return relationEmbeddingNetwork
end


---------New faster implementation---------

function FeatureEmbedding:getEmbeddingNetworkFast(relationVocabSize, relationEmbeddingDim, entityTypeVocabSize, entityTypeEmbeddingDim, numFeatureTemplates)
	local  cat = nn.Concat(3)
	local relationEmbeddingNetwork = self:getRelationEmbeddingNetworkFast(relationVocabSize, relationEmbeddingDim, numFeatureTemplates)
	local entityTypeEmbeddingNetwork = self:getEntityTypeEmbeddingNetworkFast(entityTypeVocabSize, entityTypeEmbeddingDim, numFeatureTemplates)
	cat:add(entityTypeEmbeddingNetwork):add(relationEmbeddingNetwork)
	return nn.Sequential():add(cat)
end


function FeatureEmbedding:getEntityTypeEmbeddingNetworkFast(entityTypeVocabSize, entityTypeEmbeddingDim, numFeatureTemplates)
	local network = nn.Sequential()
	local lookupTable = nn.LookupTableWithGrad(entityTypeVocabSize, entityTypeEmbeddingDim)
	local mapper = nn.Sequential():add(lookupTable):add(nn.Copy(false,false,true))
	local mr1 = nn.MapReduce(mapper,nn.Sum_nc(3))
	return network:add(nn.Narrow(3,1,numFeatureTemplates-1)):add(mr1)
end

function FeatureEmbedding:getRelationEmbeddingNetworkFast(relationVocabSize, relationEmbeddingDim, numFeatureTemplates)
	local network = nn.Sequential()
	return network:add(nn.Select(3,numFeatureTemplates)):add(nn.LookupTableWithGrad(relationVocabSize,relationEmbeddingDim)):add(nn.Copy(false,false,true))
end