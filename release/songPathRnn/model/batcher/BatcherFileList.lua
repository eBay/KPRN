--********************************************************************
--Source: https://github.com/rajarshd/ChainsofReasoning
--See Chains of Reasoning over Entities, Relations, and Text using Recurrent Neural Networks
--https://arxiv.org/abs/1607.01426
--**********************************************************************
require 'Batcher'
require 'os'
local BatcherFileList = torch.class('BatcherFileList')

function BatcherFileList:__init(dataDir, batchSize, shuffle, maxBatches, useCuda, filelist)
	local fileList = dataDir..'/'..filelist
	self.doShuffle = shuffle
	self.batchSize = batchSize
	self.useCuda = useCuda
	self.gpuLabelsTable = {} -- tables to hold preallocated gpu tensors
	self.gpuDataTable = {}
	self.gpuClassIdTable = {}
	self.emptyBatcherIndex = {} -- table to hold index of batchers which are already empty. When all of them are empty we have to call reset
	self.numEmptyBatchers = 0
	if useCuda then require 'cunn' end
	self.batchers = {}
	self.numBatchers = 0
	print(string.format('reading file list from %s',fileList))
	local file_counter = 0
	for file in io.lines(fileList) do
		-- concatenate with the path to the data directory
		local batcher = Batcher(dataDir..'/'..file, batchSize, self.doShuffle)
		table.insert(self.batchers, batcher)
		self.numBatchers = self.numBatchers + 1
		file_counter = file_counter + 1
		if file_counter % 1000 == 0 then
			print(file_counter..' files read!')
		end
	end
	print((string.format(' Done reading file list from %s',fileList)))
--	self.maxBatches = math.min(maxBatches, self.numBatchers)
	self.maxBatches = self.numBatchers
	self.startIndex = 1
	self.endIndex = self.maxBatches
	self.index = nil
	if self.doShuffle then
		self.index = torch.randperm(self.numBatchers)
	else
		self.index = torch.Tensor(self.numBatchers)
		for i=1, self.numBatchers do
			self.index[i] = i
		end
	end
	self.currentIndex = 1
	if useCuda then self:preallocateTensorToGPU() end
end

function BatcherFileList:preallocateTensorToGPU()
	
	-- for k,v in pairs(self.gpuLabelsTable) do --getting the tensors out of the gpu manually probably is more space efficient?
	-- 	self.gpuLabelsTable[k]:double()
	-- 	self.gpuDataTable[k]:double()
	-- end
	self.gpuLabelsTable = {}
	self.gpuDataTable = {}
	self.gpuClassIdTable = {}
	for i=self.startIndex, self.endIndex do
		local batchIndex = self.index[i]
		local batcher = self.batchers[batchIndex]
		local labelDimension, numPaths, numTokensInPath, numFeatureTemplates = batcher:getSizes()
		local classId = batcher:getClassId()
		--preallocate batchSize X numPaths X numTokensInPath X numFeatureTemplates tensors; the first dimension batchSize is the max size of the first dimension; so tensor:resize operation wont create any havoc.
		local labelTensor = torch.CudaTensor(self.batchSize, labelDimension)
		local dataTensor = torch.CudaTensor(self.batchSize, numPaths, numTokensInPath, numFeatureTemplates)
		self.gpuLabelsTable[batchIndex] = labelTensor
		self.gpuDataTable[batchIndex] = dataTensor
		self.gpuClassIdTable[batchIndex] = classId
	end
	self:populateGPUTensor() --populate it here for the first time
end

--populate the gpu tensors with the next batches from each batcher. If the batcher has been used up, put it in the map.
function BatcherFileList:populateGPUTensor()
	--populate tensors with data
	for i=self.startIndex, self.endIndex do
		local batchIndex = self.index[i]
		local batcher = self.batchers[batchIndex]
		local labels, data = batcher:getBatch()
		if labels ~= nil then
			local labelTensor = self.gpuLabelsTable[batchIndex]
			local dataTensor = self.gpuDataTable[batchIndex]
			labelTensor:resize(labels:size()):copy(labels)
			dataTensor:resize(data:size()):copy(data)
		else
			if self.emptyBatcherIndex[batchIndex] == nil then
				self.numEmptyBatchers = self.numEmptyBatchers + 1
				self.emptyBatcherIndex[batchIndex] = 0 --adding to the table with a dummy value.
			end
		end
	end
end

--to be called when we have gone over every example of training set
function BatcherFileList:reset()
	self.currentIndex = 1
	self.emptyBatcherIndex = {}
	self.numEmptyBatchers = 0
	self.startIndex = 1
	self.endIndex = self.maxBatches
	if self.doShuffle then
		self.index = torch.randperm(self.numBatchers)
	else
		self.index = torch.Tensor(self.numBatchers)
		for i=1, self.numBatchers do
			self.index[i] = i
		end
	end
	--finally reset all batchers
	for i=1, self.numBatchers do self.batchers[i]:reset() end
	if self.useCuda then self:preallocateTensorToGPU() end
end

--gets all batches present in batchers[startIndex] to batchers[endIndex]
function BatcherFileList:getBatchInternal()
	if self.useCuda then
		local numBatchesInGPU = self.endIndex - self.startIndex + 1
		while self.numEmptyBatchers < numBatchesInGPU do
			for i = self.currentIndex, self.endIndex do
				local batchIndex = self.index[i]
				if self.emptyBatcherIndex[batchIndex] == nil then -- the batcher isnt empty
					self.currentIndex = i + 1
					return self.gpuLabelsTable[batchIndex], self.gpuDataTable[batchIndex], self.gpuClassIdTable[batchIndex]
				end
			end
			self:populateGPUTensor()
			self.currentIndex = self.startIndex
		end
	else
		if self.currentIndex >= self.numBatchers then self.currentIndex = 1 end
		for i = self.currentIndex, self.numBatchers do
			local batchIndex = self.index[i]
			local batcher = self.batchers[batchIndex]
			local labels, data = batcher:getBatch()
			if labels ~= nil then
				self.currentIndex = i
--				print("batcher:getClassId()",batcher:getClassId())
				return labels, data, batcher:getClassId()
			end
		end
	end
	return nil -- end of epoch
end

function BatcherFileList:getBatch()
	if self.useCuda then
		while self.startIndex <= self.numBatchers do
			local labels, data, classId = self:getBatchInternal()
			if labels == nil then 
				--all batches in batchers[startIndex] to batchers[endIndex] have been returned
				self.startIndex = self.endIndex + 1
				self.endIndex = math.min(self.startIndex + self.maxBatches - 1, self.numBatchers)
				self.emptyBatcherIndex = {}
				self.numEmptyBatchers = 0
				self.currentIndex = self.startIndex
				if self.startIndex <= self.numBatchers then
					self:preallocateTensorToGPU()
				end
			else
				return labels, data, labels:size(1), classId
			end
			
		end
		return nil
	else
		while self.startIndex <= self.numBatchers do
			local labels, data, classId = self:getBatchInternal()
--			print("BatcherFileList:", classId)
			if labels == nil then
				--all batches in batchers[startIndex] to batchers[endIndex] have been returned
				self.startIndex = self.endIndex + 1
				self.endIndex = math.min(self.startIndex + self.maxBatches - 1, self.numBatchers)
				self.emptyBatcherIndex = {}
				self.numEmptyBatchers = 0
				self.currentIndex = self.startIndex
--				if self.startIndex <= self.numBatchers then
--					self:preallocateTensorToGPU()
--				end
			else
				return labels, data, labels:size(1), classId
			end

		end
		return nil
	end
end