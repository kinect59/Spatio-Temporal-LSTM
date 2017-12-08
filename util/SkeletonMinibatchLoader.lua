
function read_csv_to_tensor(filepath, rowcount, colcount)
   local csvFile = io.open(filepath, 'r')  
   --local header = csvFile:read()

   local rdata = torch.FloatTensor(rowcount, colcount)
   local i = 0  
   for line in csvFile:lines('*l') do
      i = i + 1
	  if (i % 10000 == 0) then print('read_csv_to_tensor ' .. tostring(i) .. '/' .. tostring(rowcount)) end
      local l = line:split(',')
      for key, val in ipairs(l) do
         rdata[i][key] = val
      end
   end
   csvFile:close()
   return rdata
end

-- Modified from https://github.com/oxford-cs-ml-2015/practical6
-- the modification included support for train/val/test splits
local SkeletonMinibatchLoader = {}
SkeletonMinibatchLoader.__index = SkeletonMinibatchLoader

local myuniformrandom
local unifidx
local unifsz

function SkeletonMinibatchLoader.create(data_dir, batch_size, seq_length, crosssub)
    local self = {}
    setmetatable(self, SkeletonMinibatchLoader)

    local skl_tfile          = path.join(data_dir, 'skl.csv')
    local desc_tfile         = path.join(data_dir, 'descs.csv')
    local skl_file           = path.join(data_dir, 'skl.t7')
    local desc_file          = path.join(data_dir, 'descs.t7')
	local xs_split_subj_file = path.join(data_dir, 'training_testing_subjects.csv')

	self.vocab_size = 150
	self.output_size = 60
	
	local run_prepro = false
    if not path.exists(skl_file) then
        print('skl.t7 does not exist. Running data conversion...!')
        run_prepro = true
    else 
    	-- check if the input file was modified since last time we ran the prepro. if so, we have to rerun the preprocessing
        local skl_tattr = lfs.attributes(skl_tfile)
        local skl_attr = lfs.attributes(skl_file)
        if skl_tattr.modification > skl_attr.modification then
            print('skl.t7 detected as stale. Re-running conversion...')
            run_prepro = true
        end
    end
    if run_prepro then
        print('one-time conversion: preprocessing SKL text file ' .. skl_tfile .. '...')
        self.sklraw = read_csv_to_tensor(skl_tfile, 4763874, self.vocab_size) -- Pay attention to the parameters!!!!!!!
		torch.save(skl_file, self.sklraw)
	else
		print('loading SKL data files...')
		self.sklraw = torch.load(skl_file)
    end
	
	run_prepro = false
    if not path.exists(desc_file) then
        print('descs.t7 does not exist. Running data conversion...!')
        run_prepro = true
    else
        -- check if the input file was modified since last time we ran the prepro. if so, we have to rerun the preprocessing
        local desc_tattr = lfs.attributes(desc_tfile)
        local desc_attr = lfs.attributes(desc_file)
        if desc_tattr.modification > desc_attr.modification then
            print('desc.t7 detected as stale. Re-running conversion...')
            run_prepro = true
        end
    end
    if run_prepro then
        print('one-time conversion: preprocessing DESC text file ' .. desc_tfile .. '...')
		self.descsraw = read_csv_to_tensor(desc_tfile, 7, 56578) -- Pay attention to the parameters!!!!!!!
		torch.save(desc_file, self.descsraw)
	else
		print('loading DESC data files...')
		self.descsraw = torch.load(desc_file)
	end	   
	
	self.xs_split_subjs = read_csv_to_tensor(xs_split_subj_file, 40, 1) -- Pay attention to the parameters!!!!!!!
	self.sample_count = self.descsraw:size(2)
	
	-- self.batches is a table of tensors
    print('reshaping tensor...')
    self.batch_size = batch_size
    self.seq_length = seq_length
	self.crosssub   = crosssub

	local ntrain = 0
	local nval   = 0 
	local ntest  = 0
	
	self.samples_testflag = torch.ByteTensor(self.sample_count)
	
	for i = 1, self.sample_count do
		local isfortest = false
		if self.crosssub == 1 then
			if self.xs_split_subjs[self.descsraw[3][i]][1] == 0 then
				isfortest = true
			end
		else
			if self.descsraw[2][i] == 1 then
				isfortest = true
			end
		end
		
		if isfortest then
			self.samples_testflag[i] = 1
			nval = nval + 1
		else
			self.samples_testflag[i] = 0
			ntrain = ntrain + 1
		end
	end
	
	self.ns_train = ntrain
	self.nb_train = ntrain / self.batch_size -- Note: nb_train is float number
	self.ns_val   = nval
	self.nb_val   = nval   / self.batch_size -- Note: nb_val   is float number
	self.ns_test  = 0
	self.nb_test  = 0

	-- to be assigned on the beggining of each epoch
	self.x_train_allbatches = torch.Tensor(torch.ceil(self.nb_train)*self.batch_size, seq_length, self.vocab_size):zero()
	self.y_train_allbatches = torch.Tensor(torch.ceil(self.nb_train)*self.batch_size):zero()
	self.x_val_allbatches   = torch.Tensor(torch.ceil(self.nb_val)*self.batch_size, seq_length, self.vocab_size):zero()
	self.y_val_allbatches   = torch.Tensor(torch.ceil(self.nb_val)*self.batch_size):zero()
	
	self.batch_count = {self.nb_train, self.nb_val, self.nb_test} 
	
	print('Reading done')
	print('Training data: '   .. tostring(ntrain) .. ' samples, ' .. tostring(self.nb_train) .. ' batches')
	print('Validation data: ' .. tostring(nval)   .. ' samples, ' .. tostring(self.nb_val)   .. ' batches')
	
	unifsz = 123456
	myuniformrandom = torch.Tensor(unifsz)
	unifidx = 1
	for i = 1, unifsz do
		myuniformrandom[i] = torch.uniform(0, 1)
	end

	self.epochno = 0
	self:next_epoch()
	
	return self
end

--[[function SkeletonMinibatchLoader:reset_batch_pointer(split_index, batch_index)
    batch_index = batch_index or 0
    self.batch_ix[split_index] = batch_index
end
]]

function SkeletonMinibatchLoader:next_epoch()

	local x_val_allbatches = self.x_val_allbatches
	local y_val_allbatches = self.y_val_allbatches
	local x_train_allbatches = self.x_train_allbatches
	local y_train_allbatches = self.y_train_allbatches
	local sklraw = self.sklraw
	local descsraw = self.descsraw
	local seq_length = self.seq_length
	
	local tttimer = torch.tic()
	
	self.epochno = self.epochno + 1
	unifidx = torch.round(torch.uniform(1, unifsz - 1))
	
	print('Data preparation for epoch ' .. tostring(self.epochno))
	
	x_train_allbatches:zero()
	y_train_allbatches:zero()
	
	local idx_randperm_tr = torch.randperm(self.ns_train)
	local idx_randperm_val 
	if self.epochno == 1 then
		idx_randperm_val = torch.randperm(self.ns_val)
	end
	
	local thissample
	local step
	local isfortest
	
	local tcnt = 0
	local vcnt = 0
	
	local randidx = torch.LongTensor(self.seq_length)
	
	local sumtest = 0
	
	for i = 1, self.sample_count do		
		isfortest = (self.samples_testflag[i] == 1)
		if (not isfortest) or (self.epochno == 1) then
			if isfortest then
				vcnt = vcnt + 1
			else
				tcnt = tcnt + 1
			end
			
			thissample = sklraw[{{descsraw[6][i], descsraw[7][i]}, {}}]

			step = thissample:size(1) / seq_length
			
			for s = 1, seq_length do
				--randidx[s] = torch.round(torch.uniform(1+(s-1)*step,s*step))
				randidx[s] = torch.round((s-1)*step + 1 + myuniformrandom[unifidx]*(step-1))
				unifidx = unifidx + 1
				if unifidx > unifsz then
					unifidx = 1
				end
			end
			
			if isfortest then
				x_val_allbatches[idx_randperm_val[vcnt]] = thissample:index(1, randidx)
				y_val_allbatches[idx_randperm_val[vcnt]] = descsraw[5][i]
			else
				x_train_allbatches[idx_randperm_tr[tcnt]] = thissample:index(1, randidx)
				y_train_allbatches[idx_randperm_tr[tcnt]] = descsraw[5][i]
			end
		end
	end
end

function SkeletonMinibatchLoader:retrieve_batch(split_index, batch_number)
	local needs_dummy_padding = false
	if batch_number > torch.ceil(self.batch_count[split_index]) then
		print('Error! The requested batch number is out of range!')
		print('Debug info:')
		print(split_index)
		print(batch_number)
		print(self.batch_count[split_index])
		os.exit(777)
	elseif batch_number > self.batch_count[split_index] then
		print('Warning! The requested batch contains dummy values!')
	end
	
	local startindex = 1 + (batch_number - 1) * self.batch_size
	local endindex = batch_number * self.batch_size
		
	if split_index == 1 then -- training
		self.x_batches = self.x_train_allbatches[{{startindex, endindex}, {}, {}}];
		self.y_batches = self.y_train_allbatches[{{startindex, endindex}}];
	else
		self.x_batches = self.x_val_allbatches[{{startindex, endindex}, {}}];
		self.y_batches = self.y_val_allbatches[{{startindex, endindex}}];
	end
    return self.x_batches, self.y_batches -- valid_samples_count
end

return SkeletonMinibatchLoader

