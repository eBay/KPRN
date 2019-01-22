--********************************************************************
--Source: https://github.com/rajarshd/ChainsofReasoning
--See Chains of Reasoning over Entities, Relations, and Text using Recurrent Neural Networks
--https://arxiv.org/abs/1607.01426
--**********************************************************************
--Most of the code has been (shamelessly) taken from Pat's repo (https://github.com/patverga/torch-relation-extraction)

local TypeNetwork = torch.class('TypeNetwork')

function TypeNetwork:__init(useCuda)
    -- body
    self.useCuda = useCuda or false
end


function TypeNetwork:to_cuda(x) return self.useCuda and x:cuda() or x:double() end

--[[ a function that takes the the output of {pos_row_encoder, col_encoder, neg_row_encoder}
    and returns {pos score, neg score} ]]--
-- layers to compute the dot prduct of the positive and negative samples
function TypeNetwork:build_scorer()
    local pos_score = nn.Sequential()
        :add(nn.NarrowTable(1, 2)):add(nn.CMulTable()):add(nn.Sum(2))
    local neg_score = nn.Sequential()
        :add(nn.NarrowTable(2, 2)):add(nn.CMulTable()):add(nn.Sum(2))
    local score_table = nn.ConcatTable()
        :add(pos_score):add(neg_score)
    return score_table
end

function TypeNetwork:build_network(pos_row_encoder, col_encoder)
	local neg_row_encoder = pos_row_encoder:clone()
    local loading_par_table = nn.ParallelTable()
    loading_par_table:add(pos_row_encoder)
    loading_par_table:add(col_encoder)
    loading_par_table:add(neg_row_encoder)

    -- add the parallel dot products together into one sequential network
    local net = nn.Sequential()
        :add(loading_par_table)
        :add(self:build_scorer())

    -- need to do param sharing after tocuda
    pos_row_encoder:share(neg_row_encoder, 'weight', 'bias', 'gradWeight', 'gradBias')
    return net
end