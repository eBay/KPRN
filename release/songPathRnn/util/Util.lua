
#********************************************************************
#Source: https://github.com/rajarshd/ChainsofReasoning
#See Chains of Reasoning over Entities, Relations, and Text using Recurrent Neural Networks
#https://arxiv.org/abs/1607.01426
#**********************************************************************
-- Shamelessly copying script by David Belanger.

require 'os'
local Util = torch.class('Util')

function Util:splitByDelim(str,delim,convertFromString)
	local convertFromString = convertFromString or false

	local function convert(input)
		if(convertFromString) then  return tonumber(input)  else return input end
	end

    local t = {}
    local pattern = '([^'..delim..']+)'
    for word in string.gmatch(str, pattern) do
     	table.insert(t,convert(word))
    end
    return t
end

function Util:printRow(t)
	assert(t:dim() == 1)
	local num = t:size(1)
	for i = 1,num do
		io.write(t[i].." ")
	end
	io.write('\n')
end

function Util:printMatMatlab(t)
	assert(t:dim() == 2)
	io.write('[')
	for i = 1,t:size(1) do
		for j = 1,(t:size(2)-1) do
			io.write(t[i][j]..",")
		end
		io.write(t[i][t:size(2)])
		if(i < t:size(1) ) then
			io.write(';')
		end
	end
	io.write(']\n')
end

function Util:printRow(t)
	assert(t:dim() == 1)
	local num = t:size(1)
	for i = 1,num do
		io.write(t[i].." ")
	end
	io.write('\n')
end

function Util:printMat(t)
	assert(t:dim() == 2)
	for i = 1,t:size(1) do
		Util:printRow(t[i])
	end
end


function Util:loadMap(file)
	print(string.format('reading from %s',file))
	local map = {}
	for s in io.lines(file) do
		table.insert(map,s)
	end
	return map
end

function Util:loadReverseMap(file)
	print(string.format('reading from %s',file))
	local map = {}
	local cnt = 1
	for s in io.lines(file) do
		map[s] = cnt
		cnt = cnt+1
	end
	return map
end

function Util:CopyTable(table)
	copy = {}
	for j,x in pairs(table) do copy[j] = x end
	return copy
end

function Util:deepcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[Util:deepcopy(orig_key)] = Util:deepcopy(orig_value)
        end
        setmetatable(copy, Util:deepcopy(getmetatable(orig)))
    elseif torch.isTensor(orig) then
    	copy = orig:clone()
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end

function Util:assertNan(x,msg)
	if(torch.isTensor(x))then
		assert(x:eq(x):all(),msg)
	else
		assert( x == x, msg)
	end
end

--This assumes that the inputs are regularly sized. It accepts inputs of dimension 1,2, or 3
--TODO: it's possible that there's a more efficient way to do this using something in torch
function Util:table2tensor(tab)
	
	local function fourDtable2tensor(tab)
		local s1 = #tab
		local s2 = #tab[1]
		local s3 = #tab[1][1]
		local s4 = #tab[1][1][1]

		local tensor = torch.Tensor(s1,s2,s3,s4)
		for i = 1,s1 do
			assert(#tab[i] == s2,"input tensor is expected to have the same number of elements in each dim. issue in dim 2.")
			for j = 1,s2 do 
				assert(#tab[i][j] == s3,"input tensor is expected to have the same number of elements in each dim. isssue in dim 3.")
				for k = 1,s3 do
					assert(#tab[i][j][k] == s4,"input tensor is expected to have the same number of elements in each dim. isssue in dim 4.")
					for l = 1,s4 do
						tensor[i][j][k][l] = tab[i][j][k][l]
					end
				end
			end
		end
		return tensor
	end

	local function threeDtable2tensor(tab)
		local s1 = #tab
		local s2 = #tab[1]
		local s3 = #tab[1][1]

		local tensor = torch.Tensor(s1,s2,s3)
		for i = 1,s1 do
			assert(#tab[i] == s2,"input tensor is expected to have the same number of elements in each dim. issue in dim 2.")
			for j = 1,s2 do 
				assert(#tab[i][j] == s3,"input tensor is expected to have the same number of elements in each dim. isssue in dim 3.")
				for k = 1,s3 do 	
					tensor[i][j][k] = tab[i][j][k]
				end
			end
		end
		return tensor
	end

	local function twoDtable2tensor(tab)
		local s1 = #tab
		local s2 = #tab[1]
		local tensor = torch.Tensor(s1,s2)
		for i = 1,s1 do
			assert(#tab[s1] == s2,"input tensor is expected to have the same number of elements in each row")
			for j = 1,s2 do 
				tensor[i][j] = tab[i][j]
			end
		end
		return tensor
	end

	local function oneDtable2tensor(tab)
		local s1 = #tab
		local tensor = torch.Tensor(s1)
		for i = 1,s1 do
			tensor[i] = tab[i]
		end
		return tensor
	end

	local function isTable(elem)
		return type(elem) == "table"
	end

	if(isTable(tab[1])) then
		if(isTable(tab[1][1])) then
			if(isTable(tab[1][1][1])) then
				return fourDtable2tensor(tab)
			else
				return threeDtable2tensor(tab)
			end
		else
			return twoDtable2tensor(tab)
		end
	else
		return oneDtable2tensor(tab)
	end
end

function Util:mapLookup(ints,map)
	local out = {}
	for s in io.lines(ints:size(2)) do
		table.insert(out,s)
	end
	return map
end

--TODO: could this be improved by allocating ti11 on as a cuda tensor at the beginning?
function Util:sparse2dense(tl,labelDim,useCuda,shift) --the second arg is for the common use case that we pass it zero-indexed values
	local ti11
	local shift = shift or false

	if(useCuda) then
		ti11 = torch.CudaTensor(tl:size(1),tl:size(2),labelDim)
	else
		ti11 = torch.Tensor(tl:size(1),tl:size(2),labelDim)
	end
	ti11:zero()
	for i = 1,tl:size(1) do
		for j = 1,tl:size(2) do
			local v = tl[i][j]
			if(shift) then v = v+1 end
			ti11[i][j][v] = 1
		end
	end
	return ti11 
end

--this is copied from http://ericjmritz.name/2014/02/26/lua-is_array/
function Util:isArray(t)
    local i = 0
    for _ in pairs(t) do
        i = i + 1
        if t[i] == nil then return false end
    end
    return true
end


