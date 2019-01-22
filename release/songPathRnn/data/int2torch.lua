package.path = package.path ..';../util/?.lua'
require 'torch'
require 'Util'
require 'os'

cmd = torch.CmdLine()
cmd:option('-input','','input file')
cmd:option('-output','','out')
cmd:option('-tokenLabels',0,'whether the labels are at the token level (alternative: a single label for the whole sequence)')
cmd:option('-tokenFeatures',1,'whether each token has features (alternative: just a single int index into vocab)')
cmd:option('-addOne',0,'whether to add 1 to every input (torch is 1-indexed and the preprocessing may be 0-indexed)')


local params = cmd:parse(arg)
local expectedLen = params.len
local outFile = params.output
local useTokenLabels = params.tokenLabels == 1
local useTokenFeats = params.tokenFeatures == 1

local multiPaths = true

local intLabels = {}
local intInputs = {}

shift = params.addOne
assert(shift == 0 or shift == 1)
for line in io.lines(params.input) do
	local fields = Util:splitByDelim(line,"\t",false)
	local labelString = fields[1]
	local inputString = fields[2]
	local labels = nil
	if(useTokenLabels) then 
		labels = Util:splitByDelim(labelString," ",true)
	else
		labels = tonumber(labelString)	
	end
	local all_paths = {}
	if(multiPaths) then 		
		local inputs = Util:splitByDelim(inputString,";")		
		for i = 1,#inputs do 			
			tokens_in_a_path = Util:splitByDelim(inputs[i]," ")
			path_table = {}
			if(useTokenFeats) then				
				for i = 1,#tokens_in_a_path do
					token_feats = Util:splitByDelim(tokens_in_a_path[i],",")
					table.insert(path_table,token_feats)
				end
				table.insert(all_paths,path_table)
			else
				table.insert(all_paths,tokens_in_a_path)
			end			
		end
	end
	table.insert(intLabels,labels)
	table.insert(intInputs,all_paths)
end

local labels = Util:table2tensor(intLabels)
local data = Util:table2tensor(intInputs) --internally, this asserts that every input sentence is of the same length and there are the same # of features per token
if(shift == 1) then
--	labels:add(1)
	data:add(1)
end

local out = {
	labels = labels,
	data = data
}

torch.save(outFile,out)
