--********************************************************************
--Source: https://github.com/rajarshd/ChainsofReasoning
--See Chains of Reasoning over Entities, Relations, and Text using Recurrent Neural Networks
--https://arxiv.org/abs/1607.01426
--**********************************************************************
package.path = package.path ..';../?.lua'

require 'TypeBatcher'

local input_file = '/iesl/canvas/rajarshi/emnlp_entity_types_data/train/train.torch'

local batchSize = 128
local shuffle = true
local vocabSize = 10000
local genNeg = true


local typeBatcher = TypeBatcher(input_file, batchSize, shuffle, genNeg, vocabSize)

-----see if getBatch() works--------------
print('####################Test 1 - Check if getBatch works####################')
local count = 0
local ret = typeBatcher:getBatch()
local pos, types, neg = ret[1], ret[2], ret[3]
print(pos:size())
print(types:size())
assert(pos ~= nil and types ~= nil)
print('Success!')
count = count + 1

-- ------call getBatch till it returns nil and reset and then go over it again and check if counts are same--------
print('####################Test 2 - Check if reset works ####################')
while(true)
do
	local  ret = typeBatcher:getBatch()
	if ret == nil then
		break
	end
	count = count + 1
end
print('Total number of batches are '..count)

typeBatcher:reset()
local count1 = 0
while(true)
do
	local  ret = typeBatcher:getBatch()
	if ret == nil then
		break
	end
	count1 = count1 + 1
end

assert(count1 == count)

print('Success!')


--Check if genNeg flag works-----
print('####################Check if genNeg flag works####################')
local genNeg = true

typeBatcher = TypeBatcher(input_file, batchSize, shuffle, genNeg, vocabSize)
local ret = typeBatcher:getBatch()
local pos, types, neg = ret[1], ret[2], ret[3]
assert(neg ~= nil)
assert(pos:size(1) == neg:size(1))
local max = torch.max(neg)
assert(max <= vocabSize)

local genNeg = false
typeBatcher = TypeBatcher(input_file, batchSize, shuffle, genNeg, vocabSize)
local ret = typeBatcher:getBatch()
local pos, types, neg = ret[1], ret[2], ret[3]
assert(neg == nil)
print('Success!')


