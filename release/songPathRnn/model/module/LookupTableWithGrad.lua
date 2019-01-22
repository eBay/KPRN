--********************************************************************
--Source: https://github.com/rajarshd/ChainsofReasoning
--See Chains of Reasoning over Entities, Relations, and Text using Recurrent Neural Networks
--https://arxiv.org/abs/1607.01426
--**********************************************************************
local LookupTableWithGrad, parent = torch.class('nn.LookupTableWithGrad', 'nn.LookupTable')

function LookupTableWithGrad:__init(nIndex, nOutput)
   parent.__init(self,nIndex,nOutput)
end

function LookupTableWithGrad:updateGradInput(input,gradInput)
	self.gradInput:resizeAs(input)
	self.gradInput:zero()
	return self.gradInput
end