--********************************************************************
--Source: https://github.com/rajarshd/ChainsofReasoning
--See Chains of Reasoning over Entities, Relations, and Text using Recurrent Neural Networks
--https://arxiv.org/abs/1607.01426
--**********************************************************************
Sum_nc, _ = torch.class('nn.Sum_nc', 'nn.Sum')
-- function Sum_nc:updateGradInput(input, gradOutput)
--     local size = input:size()
--     size[self.dimension] = 1
--     -- modified code:
--     if gradOutput:isContiguous() then
--         gradOutput = gradOutput:view(size) -- doesn't work with non-contiguous tensors
--     else
--         gradOutput = gradOutput:resize(size) -- slower because of memory reallocation and changes gradOutput
--         -- gradOutput = gradOutput:clone():resize(size) -- doesn't change gradOutput; safer and even slower
--     end
--     --
--     self.gradInput:resizeAs(input)
--     self.gradInput:copy(gradOutput:expandAs(input))
--     return self.gradInput
-- end 

function Sum_nc:updateGradInput(input, gradOutput)
    local dimension = self:_getPositiveDimension(input)
    -- zero-strides dont work with MKL/BLAS, so
    -- dont set self.gradInput to zero-stride tensor.
    -- Instead, do a deepcopy
    local size = input:size()
    size[dimension] = 1
     if gradOutput:isContiguous() then
        gradOutput = gradOutput:view(size) -- doesn't work with non-contiguous tensors
    else
        gradOutput = gradOutput:resize(size) -- slower because of memory reallocation and changes gradOutput
        -- gradOutput = gradOutput:clone():resize(size) -- doesn't change gradOutput; safer and even slower\
    end
    self.gradInput:resizeAs(input)
    self.gradInput:copy(gradOutput:expandAs(input))
    if self.sizeAverage then
        self.gradInput:div(input:size(dimension))
    end
    return self.gradInput
end