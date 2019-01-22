--********************************************************************
--Source: https://github.com/rajarshd/ChainsofReasoning
--See Chains of Reasoning over Entities, Relations, and Text using Recurrent Neural Networks
--https://arxiv.org/abs/1607.01426
--**********************************************************************
--test cases for batcher.lua
package.path = package.path ..';../?.lua'
require 'Batcher'

local fileName = '/iesl/local/rajarshi/data_arvind_original/_music_artist_genre/train/train.txt.25.torch'
local  batchSize = 32
local shuffle = true
local  batcher = Batcher(fileName, batchSize, shuffle)

local test_counter = 0
-----see if getBatch() works--------------
local  labels, data = batcher:getBatch()
print('Labels size')
print(labels:size())
print('data size')
print(data:size())
assert(labels ~= nil and data ~= nil)
print(string.format('test %d passed!',test_counter))
test_counter = test_counter + 1

--------see if getClassId() works-----------

local classId = batcher:getClassId()
print('classId\t'..classId)
assert(classId ~= nil)
print(string.format('test %d passed!',test_counter))
test_counter = test_counter + 1


------call getBatch till it returns nil--------
local count = 1 --starting from 1, because already counted 1 in the previous call to getBacther()
while(true)
do
	local  labels, data = batcher:getBatch()
	if labels == nil then
		break
	end
	-- print(labels:size())
	-- print(data:size())
	count = count + 1
end
assert(count == 6) --count is the number of batches
print(string.format('test %d passed!',test_counter))
test_counter = test_counter + 1


--------------test reset-----------------
batcher:reset()

local count1 = 0
while(true)
do
	local  labels, data = batcher:getBatch()
	if labels == nil then
		break
	end
	-- print(labels:size())
	-- print(data:size())
	count1 = count1 + 1
end
assert(count1 == 6)
print(string.format('test %d passed!',test_counter))
test_counter = test_counter + 1


