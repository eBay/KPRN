require 'torch'

cmd = torch.CmdLine()
cmd:option('-input','','input file')
cmd:option('-classLabel',-1,'input file')


local params = cmd:parse(arg)
local input_file = params.input
local classLabel = params.classLabel


t = torch.load(input_file)
--print("classLabel", classLabel)
t['classId'] = classLabel
--print(t)
torch.save(input_file,t) --saving with the same name