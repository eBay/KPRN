--********************************************************************
--Source: https://github.com/rajarshd/ChainsofReasoning
--See Chains of Reasoning over Entities, Relations, and Text using Recurrent Neural Networks
--https://arxiv.org/abs/1607.01426
--**********************************************************************
local SplitTableNoGrad, parent = torch.class('nn.SplitTableNoGrad', 'nn.Module')

function SplitTableNoGrad:__init(dimension, nInputDims)
   parent.__init(self)
   self.dimension = dimension
   self.nInputDims = nInputDims
end

function SplitTableNoGrad:_getPositiveDimension(input)
   local dimension = self.dimension
   if dimension < 0 then
      dimension = input:dim() + dimension + 1
   elseif self.nInputDims and input:dim()==(self.nInputDims+1) then
      dimension = dimension + 1
   end
   return dimension
end

function SplitTableNoGrad:updateOutput(input)
   local dimension = self:_getPositiveDimension(input)
   local slices = input:size(dimension)

   local currentOutput= {}
   for i=1,slices do
      currentOutput[#currentOutput+1] = input:select(dimension,i)
   end
   self.output = currentOutput
   return self.output
end 

function SplitTableNoGrad:updateGradInput(input, gradOutput)
   -- local dimension = self:_getPositiveDimension(input)
   -- local slices = input:size(dimension)
   -- self.gradInput:resizeAs(input)

   -- for i=1,slices do 
   --    local currentGradInput = gradOutput[i];        
   --    self.gradInput:select(dimension,i):copy(currentGradInput)
   -- end
   return self.gradInput
end