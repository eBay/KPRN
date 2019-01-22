--********************************************************************
--Source: https://github.com/rajarshd/ChainsofReasoning
--See Chains of Reasoning over Entities, Relations, and Text using Recurrent Neural Networks
--https://arxiv.org/abs/1607.01426
--**********************************************************************
require 'nn';
require '../MyBCECriterion';

batch = 32
labelDim = 6

data = torch.randn(batch, labelDim):clamp(0,1) -- this should be the output of my network I think
for i=1,batch do
	data[i][5] = 1
end
labels = torch.zeros(batch,1) -- labels are all zeros; so loss should be high.
criterion = nn.MyBCECriterion(5) -- 5 is the id of the target label
local err = criterion:forward(data, labels)
print(err)

local grad = criterion:backward(data, labels)

print(grad:size())
print(grad)


