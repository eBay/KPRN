--********************************************************************
--Source: https://github.com/rajarshd/ChainsofReasoning
--See Chains of Reasoning over Entities, Relations, and Text using Recurrent Neural Networks
--https://arxiv.org/abs/1607.01426
--**********************************************************************
--
--From Pat's repo (https://github.com/patverga/torch-relation-extraction/blob/master/src/nn-modules/BPRLoss.lua)
--

local BPRLoss, parent = torch.class('nn.BPRLoss', 'nn.Criterion')

function BPRLoss:__init()
    parent.__init(self)
    self.output = nil
    self.epsilon = .0001
end

function BPRLoss:updateOutput(input, y)
    local theta = input[1] - input[2]
    self.output = self.output and self.output:resizeAs(theta) or theta:clone()
    self.output = self.output:fill(1):cdiv(torch.exp(-theta):add(1))
    -- add epsilon so that no log(0)
    self.output:add(self.epsilon)
    local err = torch.log(self.output):mean() * -1.0
    return err
end

function BPRLoss:updateGradInput(input, y)
    local step = self.output:mul(-1):add(1)
    self.gradInput = { -step, step }
    return self.gradInput
end