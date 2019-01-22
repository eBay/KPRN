--********************************************************************
--Source: https://github.com/rajarshd/ChainsofReasoning
--See Chains of Reasoning over Entities, Relations, and Text using Recurrent Neural Networks
--https://arxiv.org/abs/1607.01426

-- See also: https://github.com/torch/nn/blob/master/ConcatTable.lua
-- https://github.com/torch/nn/blob/master/COPYRIGHT.txt
--**********************************************************************
local ConcatTableNoGrad, parent = torch.class('nn.ConcatTableNoGrad', 'nn.Container')

function ConcatTableNoGrad:__init()
   parent.__init(self)
   self.modules = {}
   self.output = {}
end

function ConcatTableNoGrad:updateOutput(input)
   for i=1,#self.modules do
      self.output[i] = self.modules[i]:updateOutput(input)
   end
   return self.output
end

local function retable(t1, t2, f)
   for k, v in ipairs(t2) do
      if (torch.type(v) == "table") then
         t1[k] = retable(t1[k] or {}, t2[k], f)
      else
         f(t1, k, v)
      end
   end
   for i=#t2+1, #t1 do
      t1[i] = nil
   end
   return t1
end

function ConcatTableNoGrad:updateGradInput(input, gradOutput)
   -- print("gradOutput size ")
   -- print(gradOutput)
   local isTable = torch.type(input) == 'table'
   local wasTable = torch.type(self.gradInput) == 'table'
   if isTable then
      for i,module in ipairs(self.modules) do
         local currentGradInput = module:updateGradInput(input, gradOutput[i])
         if torch.type(currentGradInput) ~= 'table' then
            error"currentGradInput is not a table!"
         end
         if #input ~= #currentGradInput then
            error("table size mismatch: "..#input.." ~= "..#currentGradInput)
         end
         if i == 1 then
            self.gradInput = wasTable and self.gradInput or {}
            retable(self.gradInput, currentGradInput,
               function(t, k, v)
                  t[k] = t[k] or v:clone()
                  t[k]:resizeAs(v)
                  t[k]:copy(v)
               end
            )
         else
            retable(self.gradInput, currentGradInput,
               function(t, k, v)
                  if t[k] then
                     t[k]:resizeAs(v)
                     t[k]:add(v)
                  else
                     t[k] = v:clone()
                  end
               end
            )
         end
      end
   else
      self.gradInput = (not wasTable) and self.gradInput or input:clone()
      for i,module in ipairs(self.modules) do
         local currentGradInput = module:updateGradInput(input, gradOutput[i])
         if i == 1 then
            self.gradInput:resizeAs(currentGradInput):copy(currentGradInput)
         else
            self.gradInput:add(currentGradInput)
         end
      end
   end
   return self.gradInput
end

function ConcatTableNoGrad:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   for i,module in ipairs(self.modules) do
      module:accGradParameters(input, gradOutput[i], scale)
   end
end

function ConcatTableNoGrad:accUpdateGradParameters(input, gradOutput, lr)
   for i,module in ipairs(self.modules) do
      module:accUpdateGradParameters(input, gradOutput[i], lr)
   end
end

function ConcatTableNoGrad:zeroGradParameters()
   for _,module in ipairs(self.modules) do
      module:zeroGradParameters()
   end
end

function ConcatTableNoGrad:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = '  |`-> '
   local ext = '  |    '
   local extlast = '       '
   local last = '   ... -> '
   local str = torch.type(self)
   str = str .. ' {' .. line .. tab .. 'input'
   for i=1,#self.modules do
      if i == self.modules then
         str = str .. line .. tab .. next .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab .. extlast)
      else
         str = str .. line .. tab .. next .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab .. ext)
      end
   end
   str = str .. line .. tab .. last .. 'output'
   str = str .. line .. '}'
   return str
end
