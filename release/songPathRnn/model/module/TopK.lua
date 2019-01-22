--********************************************************************
--Source: https://github.com/rajarshd/ChainsofReasoning
--See Chains of Reasoning over Entities, Relations, and Text using Recurrent Neural Networks
--https://arxiv.org/abs/1607.01426
--**********************************************************************
--
-- This was implemented by Pat.
-- Date: 2/17/16
--
local TopK, parent = torch.class('nn.TopK', 'nn.Max')

function TopK:__init(K, dimension, nInputDims)
    parent.__init(self, dimension, nInputDims)
    self.K = K
end

function TopK:updateOutput(input)
    self:_lazyInit()
    local dimension = self:_getPositiveDimension(input)
    local k = math.min (self.K, input:size(dimension))
    torch.topk(self._output, self._indices, input, k, dimension, true)
    self.output = self._output
    return self.output
end

function TopK:_lazyInit()
--    parent:_lazyInit()
    self._output = self._output or self.output.new()
    self._indices = self._indices or
        (torch.type(self.output) == 'torch.CudaTensor' and torch.CudaTensor() or torch.LongTensor())
    self.gradInput = self._output.new()
end

function TopK:updateGradInput(input, gradOutput)
    self:_lazyInit()
    local dimension = self:_getPositiveDimension(input)
    self.gradInput:resizeAs(input):zero():scatter(dimension, self._indices, gradOutput)
    return self.gradInput
end