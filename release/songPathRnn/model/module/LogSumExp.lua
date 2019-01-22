--********************************************************************
--Source: https://github.com/rajarshd/ChainsofReasoning
--See Chains of Reasoning over Entities, Relations, and Text using Recurrent Neural Networks
--https://arxiv.org/abs/1607.01426
--**********************************************************************
local LogSumExp, parent = torch.class('nn.LogSumExp', 'nn.Module')


function LogSumExp:__init(dim)
	self.dim = dim
end

function LogSumExp:updateOutput(input)
	self.dim = self.dim or input:dim() --default is the last dim
	if self.maxes == nil then
		self.maxes = torch.max(input, self.dim) --max scores along the dim
		self.score_minus_max = torch.add(input, -1, self.maxes:expandAs(input))
		self.exp_score_minus_max = torch.exp(self.score_minus_max)
		self.sum_exp_score_minus_max = torch.sum(self.exp_score_minus_max, self.dim)
		self.output = torch.log(self.sum_exp_score_minus_max)
	else
		torch.max(self.maxes,input, self.dim)
		self.score_minus_max:add(input, -1, self.maxes:expandAs(input))
		self.exp_score_minus_max:exp(self.score_minus_max)
		self.sum_exp_score_minus_max:sum(self.exp_score_minus_max, self.dim)
		self.output:log(self.sum_exp_score_minus_max)
	end
	self.output:add(self.maxes)
	return self.output
end

function LogSumExp:updateGradInput(input, gradOutput)
	self.gradInput = input:clone()
	self.gradInput:cdiv(self.exp_score_minus_max, self.sum_exp_score_minus_max:expandAs(input))
	self.gradInput:cmul(gradOutput:expandAs(input))
	return self.gradInput
end