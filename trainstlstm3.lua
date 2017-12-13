
--[[

This file trains a character-level multi-layer RNN on text data

Code is based on implementation in 
https://github.com/oxford-cs-ml-2015/practical6
but modified to have multi-layer support, GPU support, as well as
many other common model/optimization bells and whistles.
The practical6 code is in turn based on 
https://github.com/wojciechz/learning_to_execute
which is turn based on other stuff in Torch, etc... (long lineage)

The code is built on [char-rnn](https://github.com/karpathy/char-rnn).
Thanks for the open source from Andrej Karpathy.
]]--

package.loaded.SkeletonMinibatchLoader = nil

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'csvigo'
require 'lfs'

require 'util.misc'

local SkeletonMinibatchLoader = require 'util.SkeletonMinibatchLoader'
local model_utils = require 'util.model_utils'
local LSTM = require 'model.LSTM'
local GRU  = require 'model.GRU'
local RNN  = require 'model.RNN'
local STLSTM  = require 'model.STLSTM'
local STLSTM2 = require 'model.STLSTM2'
local STLSTM3 = require 'model.STLSTM3'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a character-level language model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','data','data directory')
-- model params
cmd:option('-rnn_size', 128, 'size of LSTM internal state')
cmd:option('-num_layers', 2, 'number of layers in the LSTM') --
cmd:option('-model', 'stlstm3', 'lstm, gru or rnn')
-- optimization
cmd:option('-learning_rate', 2e-3, 'learning rate')  -- 
cmd:option('-learning_rate_decay', 0.998, 'learning rate decay')
cmd:option('-learning_rate_decay_after', 50, 'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate', 0.95, 'decay rate for rmsprop')
cmd:option('-dropout', 0.5, 'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-seq_length', 6, 'number of timesteps to unroll for')   -- 
cmd:option('-batch_size', 100, 'number of sequences to train on in parallel') -- 
cmd:option('-max_epochs', 10000, 'number of full passes through the training data')
cmd:option('-grad_clip', 8, 'clip gradients at this value') 
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
-- bookkeeping
cmd:option('-seed', 123, 'torch manual random number generator seed')
cmd:option('-print_every', 1, 'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every', 5, 'every how many epochs should we evaluate on validation data?')
cmd:option('-checkpoint_dir', 'cv_stlstm3R128L2D0.5S6T3B100G-0.5CS_J1ConnectLastBiTree_NoRelatPos_', 'output directory where checkpoints get written')
cmd:option('-savefile', 'stlstm3', 'filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-accurate_gpu_timing', 0, 'set this flag to 1 to get precise timings when using GPU. Might make code bit slower but reports accurate timings.')
-- GPU/CPU
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU') 
cmd:option('-opencl', 0, 'use OpenCL (instead of CUDA)')

-- Evaluation Criteria
cmd:option('-crosssub', 1, 'cross-subject = 1 or cross-view = 0')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

local TIME_SLIDE = 3

-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- initialize clnn/cltorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 1 then
    local ok, cunn = pcall(require, 'clnn')
    local ok2, cutorch = pcall(require, 'cltorch')
    if not ok then print('package clnn not found!') end
    if not ok2 then print('package cltorch not found!') end
    if ok and ok2 then
        print('using OpenCL on GPU ' .. opt.gpuid .. '...')
        cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        torch.manualSeed(opt.seed)
    else
        print('If cltorch and clnn are installed, your OpenCL driver may be improperly configured.')
        print('Check your OpenCL driver installation, check output of clinfo command, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- create the data loader class
local loader = SkeletonMinibatchLoader.create(opt.data_dir, opt.batch_size, TIME_SLIDE*opt.seq_length, opt.crosssub)

local vocab_size = loader.vocab_size  -- the size of the input feature vector
print('vocab size: ' .. vocab_size)
-- make sure output directory exists
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

-- define the model: prototypes for one timestep, then clone them in time
local do_random_init = true
if string.len(opt.init_from) > 0 then
    print('loading a model from checkpoint ' .. opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    protos = checkpoint.protos
    print('overwriting rnn_size=' .. checkpoint.opt.rnn_size .. ', num_layers=' .. checkpoint.opt.num_layers .. ' based on the checkpoint.')
    opt.rnn_size = checkpoint.opt.rnn_size
    opt.num_layers = checkpoint.opt.num_layers
	opt.startingiter = checkpoint.i
    do_random_init = false
else
    print('creating an ' .. opt.model .. ' with ' .. opt.num_layers .. ' layers')
    protos = {}
    if opt.model == 'lstm' then
        protos.rnn = LSTM.lstm(vocab_size, loader.output_size, opt.rnn_size, opt.num_layers, opt.dropout)
    elseif opt.model == 'gru' then
        protos.rnn = GRU.gru(vocab_size, loader.output_size, opt.rnn_size, opt.num_layers, opt.dropout)
    elseif opt.model == 'rnn' then
        protos.rnn = RNN.rnn(vocab_size, loader.output_size, opt.rnn_size, opt.num_layers, opt.dropout)
    elseif opt.model == 'stlstm' then
        protos.rnn = STLSTM.stlstm(TIME_SLIDE*3*2, loader.output_size, opt.rnn_size, opt.num_layers, opt.dropout) 
    elseif opt.model == 'stlstm2' then
        protos.rnn = STLSTM2.stlstm2(TIME_SLIDE*3*2, loader.output_size, opt.rnn_size, opt.num_layers, opt.dropout) 
    elseif opt.model == 'stlstm3' then
        protos.rnn = STLSTM3.stlstm3(TIME_SLIDE*3*2, loader.output_size, opt.rnn_size, opt.num_layers, opt.dropout) 
    end
    protos.criterion = nn.ClassNLLCriterion()
end

-- the initial state of the cell/hidden states
init_state = {}
for L = 1, opt.num_layers do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
    if opt.gpuid >=0 and opt.opencl == 0 then h_init = h_init:cuda() end
    if opt.gpuid >=0 and opt.opencl == 1 then h_init = h_init:cl() end
    table.insert(init_state, h_init:clone())
    if opt.model == 'lstm' or opt.model == 'stlstm' or opt.model == 'stlstm2' or opt.model == 'stlstm3' then
        table.insert(init_state, h_init:clone())
    end
end

-- ship the model to the GPU if desired
if opt.gpuid >= 0 and opt.opencl == 0 then
    for k,v in pairs(protos) do v:cuda() end
end
if opt.gpuid >= 0 and opt.opencl == 1 then
    for k,v in pairs(protos) do v:cl() end
end

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn)

-- initialization
if do_random_init then
    params:uniform(-0.08, 0.08) -- small uniform numbers
end
-- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
if opt.model == 'stlstm' or opt.model == 'stlstm2' or opt.model == 'stlstm3' then
    for layer_idx = 1, opt.num_layers do
        for _, node in ipairs(protos.rnn.forwardnodes) do
            if node.data.annotations.name == "i2h_" .. layer_idx then
                print('setting forget gate biases to 1 in STLSTM layer ' .. layer_idx)
                -- the gates are, in order, i,f_j,f_t,o,g, so f is the 2nd and 3rd blocks of weights
                node.data.module.bias[{{1*opt.rnn_size+1, 2*opt.rnn_size}}]:fill(1.0)
                node.data.module.bias[{{2*opt.rnn_size+1, 3*opt.rnn_size}}]:fill(1.0)
            end
        end
    end
end

print('number of parameters in the model: ' .. params:nElement())

--------------------------------------------------------------------------------------------------------------------------------

-- Bi-Tree Model (Choose 16 joints)
local order_joint    = { 2, 21,  4, 21,  9, 10, 24, 10,  9, 21,  5,  6, 22,  6,  5, 21,  2,  1, 17, 18, 19, 18, 17,  1, 13, 14, 15, 14, 13,  1,  2 }
local order_joint_pos= { 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 }
local prev_joint     = { 2,  2, 21,  4, 21,  9, 10, 24, 10,  9, 21,  5,  6, 22,  6,  5, 21,  2,  1, 17, 18, 19, 18, 17,  1, 13, 14, 15, 14, 13,  1 }
local prev_joint_pos = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30 }

--------------------------------------------------------------------------------------------------------------------------------
local JOINT_NUM = #order_joint -- steps through the joints
--------------------------------------------------------------------------------------------------------------------------------
local USE_RELATIVE_POSITION = false -- use the relative position of the previous joint
--------------------------------------------------------------------------------------------------------------------------------

-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name, proto in pairs(protos) do  
    print('cloning ' .. name)
    clones[name] = model_utils.clone_many_times(proto, JOINT_NUM*opt.seq_length, not proto.parameters) -- the 3rd parameter is not used!
end

-- preprocessing helper function
function prepro(x, y)  
    --x = x:transpose(1,2):contiguous() -- swap the axes for faster indexing
    -- y = y:transpose(1,2):contiguous() -- no need for y
    if opt.gpuid >= 0 and opt.opencl == 0 then -- ship the input arrays to GPU
        --x = x:float():cuda() -- have to convert to float because integers can't be cuda()'d
        y = y:float():cuda()
    end
    if opt.gpuid >= 0 and opt.opencl == 1 then -- ship the input arrays to GPU
        --x = x:cl()
        y = y:cl()
    end
    return x, y
end

local tr_predict
local tr_predict_W
local tr_gt_lables

local val_predict
local val_predict_W
local val_gt_lables

local val_input_gate
local val_forget_gate


if opt.gpuid >=0 then  
	tr_predict    = torch.CudaTensor(torch.floor(loader.nb_train)*loader.batch_size, loader.output_size):zero()
	tr_gt_lables  = torch.CudaTensor(torch.floor(loader.nb_train)*loader.batch_size):zero()
	val_predict   = torch.CudaTensor(loader.ns_val, loader.output_size):zero()
	val_gt_lables = torch.CudaTensor(loader.ns_val):zero()
	
	tr_predict_W  = torch.CudaTensor(torch.floor(loader.nb_train)*loader.batch_size, loader.output_size):zero()
	val_predict_W = torch.CudaTensor(loader.ns_val, loader.output_size):zero()
	
	val_input_gate_L1  = torch.CudaTensor(loader.ns_val, opt.seq_length * JOINT_NUM):zero()
	val_forget_gate_L1 = torch.CudaTensor(loader.ns_val, opt.seq_length * JOINT_NUM):zero()
else
	tr_predict    = torch.Tensor(torch.floor(loader.nb_train)*loader.batch_size, loader.output_size):zero()
	tr_gt_lables  = torch.Tensor(torch.floor(loader.nb_train)*loader.batch_size):zero()
	val_predict   = torch.Tensor(loader.ns_val, loader.output_size):zero()
	val_gt_lables = torch.Tensor(loader.ns_val):zero()	
	
	tr_predict_W  = torch.Tensor(torch.floor(loader.nb_train)*loader.batch_size, loader.output_size):zero()
	val_predict_W = torch.Tensor(loader.ns_val, loader.output_size):zero()
	
	val_input_gate_L1  = torch.Tensor(loader.ns_val, opt.seq_length * JOINT_NUM):zero()
	val_forget_gate_L1 = torch.Tensor(loader.ns_val, opt.seq_length * JOINT_NUM):zero()
end

-- evaluate the loss over an entire split
function eval_split(split_index, max_batches) -- Note: Currently, this function can only be used for validation set
    print('evaluating loss over split index ' .. split_index)
    local n = torch.ceil(loader.batch_count[split_index])
	local last_batch_has_dummy = n > loader.batch_count[split_index]
	local valid_sample_count_last_batch = loader.ns_val % loader.batch_size
    if max_batches ~= nil then n = math.min(max_batches, n) end

    -- loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
    local loss = 0
    local rnn_state = {[0] = init_state}
    for i = 1, n do -- iterate over batches in the split
        collectgarbage()
        -- fetch a batch
        local x, y = loader:retrieve_batch(split_index, i)
        x, y = prepro(x, y)
		-- forward pass
		local predscores = (opt.gpuid >= 0) and 
							torch.CudaTensor(loader.batch_size, loader.output_size):zero() or
							torch.Tensor(    loader.batch_size, loader.output_size):zero()
		local predscores_W = (opt.gpuid >= 0) and 
							torch.CudaTensor(loader.batch_size, loader.output_size):zero() or
							torch.Tensor(    loader.batch_size, loader.output_size):zero()
							
		local input_gate_norm = (opt.gpuid >= 0) and 
							torch.CudaTensor(loader.batch_size, opt.seq_length * JOINT_NUM):zero() or
							torch.Tensor(    loader.batch_size, opt.seq_length * JOINT_NUM):zero()
							
		local forget_gate_norm = (opt.gpuid >= 0) and 
							torch.CudaTensor(loader.batch_size, opt.seq_length * JOINT_NUM):zero() or
							torch.Tensor(    loader.batch_size, opt.seq_length * JOINT_NUM):zero()
						
		if i == n and last_batch_has_dummy then
			y = y[{{1, valid_sample_count_last_batch}}]
			predscores = predscores[{{1, valid_sample_count_last_batch}, {}}]
			predscores_W = predscores_W[{{1, valid_sample_count_last_batch}, {}}]
			
			input_gate_norm  = input_gate_norm[ {{1, valid_sample_count_last_batch}, {}}]
			forget_gate_norm = forget_gate_norm[{{1, valid_sample_count_last_batch}, {}}]
		end
        
        local CurrRnnIndx = 0
        for t = 1, opt.seq_length do 
            for j = 1, JOINT_NUM do
                CurrRnnIndx = CurrRnnIndx + 1
                clones.rnn[CurrRnnIndx]:evaluate() -- for dropout proper functioning

                local CurrJointIndex = order_joint[j]
                local PrevJointIndex = prev_joint[j]
                local PrevJointPosition = prev_joint_pos[j]

                --local preRnnIndexJ = (j > 1) and (CurrRnnIndx - 1) or 0
                local preRnnIndexJ = (j > 1) and (CurrRnnIndx - j + PrevJointPosition) or 0
                local preRnnIndexT = (t > 1) and (CurrRnnIndx - JOINT_NUM) or 0 
                if (t > 1) and (j == 1) then preRnnIndexJ = CurrRnnIndx - 1 end

                local inputX_person1 = x[{{}, {(t-1)*TIME_SLIDE+1, t*TIME_SLIDE}, {   CurrJointIndex*3-2,    CurrJointIndex*3}}]:contiguous():view(loader.batch_size, TIME_SLIDE*3)
                local inputX_person2 = x[{{}, {(t-1)*TIME_SLIDE+1, t*TIME_SLIDE}, {75+CurrJointIndex*3-2, 75+CurrJointIndex*3}}]:contiguous():view(loader.batch_size, TIME_SLIDE*3)
               
                if (USE_RELATIVE_POSITION) then
                    local inputX_person1_pre = x[{{}, {(t-1)*TIME_SLIDE+1, t*TIME_SLIDE}, {   PrevJointIndex*3-2,    PrevJointIndex*3}}]:contiguous():view(loader.batch_size, TIME_SLIDE*3)
                    local inputX_person2_pre = x[{{}, {(t-1)*TIME_SLIDE+1, t*TIME_SLIDE}, {75+PrevJointIndex*3-2, 75+PrevJointIndex*3}}]:contiguous():view(loader.batch_size, TIME_SLIDE*3)
                    inputX_person1 = inputX_person1 - inputX_person1_pre
                    inputX_person2 = inputX_person2 - inputX_person2_pre
                end

                local inputX = torch.cat({inputX_person1, inputX_person2}, 2)
                inputX = (opt.gpuid >=0) and inputX:float():cuda() or inputX

                local tempInput = {}
                table.insert(tempInput, inputX)
                if preRnnIndexJ == 0 then random_list(init_state) end
                for k, v in ipairs(rnn_state[preRnnIndexJ]) do table.insert(tempInput, v) end
                if preRnnIndexT == 0 then random_list(init_state) end
                for k, v in ipairs(rnn_state[preRnnIndexT]) do table.insert(tempInput, v) end                           
                local lst = clones.rnn[CurrRnnIndx]:forward(tempInput)

                rnn_state[CurrRnnIndx] = {}  
                for index = 1, #init_state do table.insert(rnn_state[CurrRnnIndx], lst[index]) end  
                prediction = lst[#lst] 
                if i == n and last_batch_has_dummy then
                    prediction = prediction[{{1, valid_sample_count_last_batch}, {}}]
                end
                predscores = predscores + prediction
                predscores_W = predscores_W + prediction*CurrRnnIndx
				
                loss = loss + clones.criterion[CurrRnnIndx]:forward(prediction, y) --
            end
        end
        
		local sam_idx_1 = (i - 1) * loader.batch_size + 1
		local sam_idx_2 = i * loader.batch_size
		if i == n and last_batch_has_dummy then
			sam_idx_2 = (i - 1) * loader.batch_size + valid_sample_count_last_batch
		end
        
		val_predict[{{sam_idx_1, sam_idx_2}, {}}] = predscores
		val_predict_W[{{sam_idx_1, sam_idx_2}, {}}] = predscores_W
		val_gt_lables[{{sam_idx_1, sam_idx_2}}] = y
		
		-- val_input_gate_L1[{{sam_idx_1, sam_idx_2}, {}, {}}] = 
		-- val_input_gate_L1[{{sam_idx_1, sam_idx_2}, {}, {}}] = 
		
        -- carry over lstm state
        --rnn_state[0] = rnn_state[#rnn_state] -- 
        print(i .. '/' .. n .. '...')
    end
        
    loss = loss / (opt.seq_length * JOINT_NUM) / loader.batch_count[split_index]
    return loss
end

-- do fwd/bwd and return loss, grad_params
local init_state_global = clone_list(init_state)
local current_training_batch_number = 0

function feval(x)
    if x ~= params then
        params:copy(x) -- update parameters
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    local x, y = loader:retrieve_batch(1, current_training_batch_number)
    x, y = prepro(x, y)
    ------------------- forward pass -------------------
    local rnn_state = {[0] = init_state_global}
    local predictions = {}           -- softmax outputs
    local loss = 0
    local predscores = (opt.gpuid >=0) and 
						torch.CudaTensor(loader.batch_size, loader.output_size):zero() or
						torch.Tensor(    loader.batch_size, loader.output_size):zero()
	local predscores_W = (opt.gpuid >=0) and 
						torch.CudaTensor(loader.batch_size, loader.output_size):zero() or
						torch.Tensor(    loader.batch_size, loader.output_size):zero()

    local CurrRnnIndx = 0
    collectgarbage()
    for t = 1, opt.seq_length do       
        for j = 1, JOINT_NUM do
            CurrRnnIndx = CurrRnnIndx + 1
            clones.rnn[CurrRnnIndx]:training() -- make sure we are in correct mode (this is cheap, sets flag)

            local CurrJointIndex = order_joint[j]
            local PrevJointIndex = prev_joint[j]
            local PrevJointPosition = prev_joint_pos[j]

            --local preRnnIndexJ = (j > 1) and (CurrRnnIndx - 1) or 0
            local preRnnIndexJ = (j > 1) and (CurrRnnIndx - j + PrevJointPosition) or 0
            local preRnnIndexT = (t > 1) and (CurrRnnIndx - JOINT_NUM) or 0 
            if (t > 1) and (j == 1) then preRnnIndexJ = CurrRnnIndx - 1 end
 
            local inputX_person1 = x[{{}, {(t-1)*TIME_SLIDE+1, t*TIME_SLIDE}, {   CurrJointIndex*3-2,    CurrJointIndex*3}}]:contiguous():view(loader.batch_size, TIME_SLIDE*3)
            local inputX_person2 = x[{{}, {(t-1)*TIME_SLIDE+1, t*TIME_SLIDE}, {75+CurrJointIndex*3-2, 75+CurrJointIndex*3}}]:contiguous():view(loader.batch_size, TIME_SLIDE*3)
            
            if (USE_RELATIVE_POSITION) then
                local inputX_person1_pre = x[{{}, {(t-1)*TIME_SLIDE+1, t*TIME_SLIDE}, {   PrevJointIndex*3-2,    PrevJointIndex*3}}]:contiguous():view(loader.batch_size, TIME_SLIDE*3)
                local inputX_person2_pre = x[{{}, {(t-1)*TIME_SLIDE+1, t*TIME_SLIDE}, {75+PrevJointIndex*3-2, 75+PrevJointIndex*3}}]:contiguous():view(loader.batch_size, TIME_SLIDE*3)
                inputX_person1 = inputX_person1 - inputX_person1_pre
                inputX_person2 = inputX_person2 - inputX_person2_pre
            end

            local inputX = torch.cat({inputX_person1, inputX_person2}, 2)
            inputX = (opt.gpuid >=0 ) and inputX:float():cuda() or inputX     

            local tempInput = {}
            table.insert(tempInput, inputX)
            if preRnnIndexJ == 0 then random_list(init_state_global) end
            for k, v in ipairs(rnn_state[preRnnIndexJ]) do table.insert(tempInput, v) end
            if preRnnIndexT == 0 then random_list(init_state_global) end
            for k, v in ipairs(rnn_state[preRnnIndexT]) do table.insert(tempInput, v) end                           
            local lst = clones.rnn[CurrRnnIndx]:forward(tempInput)

            
            rnn_state[CurrRnnIndx] = {}  
            for index = 1, #init_state do table.insert(rnn_state[CurrRnnIndx], lst[index]) end  
            predictions[CurrRnnIndx] = lst[#lst] 
           
            predscores = predscores + predictions[CurrRnnIndx]
            predscores_W = predscores_W + predictions[CurrRnnIndx]*CurrRnnIndx
            loss = loss + clones.criterion[CurrRnnIndx]:forward(predictions[CurrRnnIndx], y)
        end
    end
    loss = loss / (opt.seq_length * JOINT_NUM)

	local sam_idx_1 = (current_training_batch_number - 1) * loader.batch_size + 1
	local sam_idx_2 = current_training_batch_number * loader.batch_size
	tr_predict[{{sam_idx_1, sam_idx_2}, {}}] = predscores
	tr_predict_W[{{sam_idx_1, sam_idx_2}, {}}] = predscores_W
	tr_gt_lables[{{sam_idx_1, sam_idx_2}}] = y

    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local drnn_state = {}
    for i = 0, (opt.seq_length * JOINT_NUM) do
        drnn_state[i] = clone_list(init_state, true) -- true also zeros the clones
    end

    local CurrRnnIndx = opt.seq_length * JOINT_NUM
    collectgarbage()
    for t = opt.seq_length, 1, -1 do
        for j = JOINT_NUM, 1, -1 do 
            -- backprop through loss, and softmax/linear
            local doutput_t = clones.criterion[CurrRnnIndx]:backward(predictions[CurrRnnIndx], y) 
            table.insert(drnn_state[CurrRnnIndx], doutput_t) -- drnn includes two part: 1) from t + 1, 2) from criterion  
            assert (#(drnn_state[CurrRnnIndx]) == (#init_state)+1)
            
            local CurrJointIndex = order_joint[j]
            local PrevJointIndex = prev_joint[j]
            local PrevJointPosition = prev_joint_pos[j]

            --local preRnnIndexJ = (j > 1) and (CurrRnnIndx - 1) or 0
            local preRnnIndexJ = (j > 1) and (CurrRnnIndx - j + PrevJointPosition) or 0
            local preRnnIndexT = (t > 1) and (CurrRnnIndx - JOINT_NUM) or 0 
            if (t > 1) and (j == 1) then preRnnIndexJ = CurrRnnIndx - 1 end

            local inputX_person1 = x[{{}, {(t-1)*TIME_SLIDE+1, t*TIME_SLIDE}, {   CurrJointIndex*3-2,    CurrJointIndex*3}}]:contiguous():view(loader.batch_size, TIME_SLIDE*3)
            local inputX_person2 = x[{{}, {(t-1)*TIME_SLIDE+1, t*TIME_SLIDE}, {75+CurrJointIndex*3-2, 75+CurrJointIndex*3}}]:contiguous():view(loader.batch_size, TIME_SLIDE*3)
            
            if (USE_RELATIVE_POSITION) then
                local inputX_person1_pre = x[{{}, {(t-1)*TIME_SLIDE+1, t*TIME_SLIDE}, {   PrevJointIndex*3-2,    PrevJointIndex*3}}]:contiguous():view(loader.batch_size, TIME_SLIDE*3)
                local inputX_person2_pre = x[{{}, {(t-1)*TIME_SLIDE+1, t*TIME_SLIDE}, {75+PrevJointIndex*3-2, 75+PrevJointIndex*3}}]:contiguous():view(loader.batch_size, TIME_SLIDE*3)
                inputX_person1 = inputX_person1 - inputX_person1_pre
                inputX_person2 = inputX_person2 - inputX_person2_pre
            end

            local inputX = torch.cat({inputX_person1, inputX_person2}, 2)
            inputX = (opt.gpuid >=0) and inputX:float():cuda() or inputX  

            local tempInput = {}
            table.insert(tempInput, inputX)
            if preRnnIndexJ == 0 then random_list(init_state_global) end
            for k, v in ipairs(rnn_state[preRnnIndexJ]) do table.insert(tempInput, v) end
            if preRnnIndexT == 0 then random_list(init_state_global) end
            for k, v in ipairs(rnn_state[preRnnIndexT]) do table.insert(tempInput, v) end                           
            local dlst = clones.rnn[CurrRnnIndx]:backward(tempInput, drnn_state[CurrRnnIndx])

            for index = 1, #init_state do
                drnn_state[preRnnIndexJ][index] = drnn_state[preRnnIndexJ][index] + dlst[index+1]   
                drnn_state[preRnnIndexT][index] = drnn_state[preRnnIndexT][index] + dlst[index+1+(#init_state)]     
            end

            CurrRnnIndx = CurrRnnIndx - 1 
        end 
    end

    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
	
	-- init_state_global = rnn_state[#rnn_state] --  
    --grad_params:div(opt.seq_length * JOINT_NUM) -- this line should be here but since we use rmsprop it would have no effect. Removing for efficiency
    -- clip gradient element-wise
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    return loss, grad_params
end

function evaluate_classification()
    local current_errorrate = {}
	
	local confuseMatrix3 = torch.zeros(loader.output_size, loader.output_size)
	local confuseMatrix4 = torch.zeros(loader.output_size, loader.output_size)
	
    for splidx = 1, 4 do
		local scores
		local gtruth
		if splidx == 1 then
			scores = tr_predict
			gtruth = tr_gt_lables
		elseif splidx == 2 then
			scores = val_predict
			gtruth = val_gt_lables
		elseif splidx == 3 then
			scores = tr_predict_W
			gtruth = tr_gt_lables
		elseif splidx == 4 then
			scores = val_predict_W
			gtruth = val_gt_lables
		end
		
		local errorrates = torch.zeros(10)
		local sortedscores = torch.sort(scores, 2, true)
		for i = 1, scores:size(1) do
			local gtprob = scores[i][gtruth[i]]
			for j = 1, 10 do
				if gtprob < sortedscores[i][j] then
					errorrates[j] = errorrates[j] + 1
				end
			end
		end
		errorrates = errorrates * 100 / scores:size(1)
		current_errorrate[splidx] = errorrates
		
		local maxscores, maxpositions = torch.max(scores, 2)
		if (splidx == 3) then 
			for i = 1, scores:size(1) do
				confuseMatrix3[gtruth[i]][maxpositions[i][1]] = confuseMatrix3[gtruth[i]][maxpositions[i][1]] + 1
			end
		elseif (splidx == 4) then 
			for i = 1, scores:size(1) do
				confuseMatrix4[gtruth[i]][maxpositions[i][1]] = confuseMatrix4[gtruth[i]][maxpositions[i][1]] + 1
			end
		end
		
	end	
	return current_errorrate,confuseMatrix3,confuseMatrix4
end

-- start optimization here
train_losses = {}
val_losses = {}
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local iterations = opt.max_epochs * torch.floor(loader.nb_train)
--local iterations_per_epoch = loader.nb_train
local loss0 = nil
local lastepoch

for i = (opt.startingiter or 1), (iterations + 1) do
	local epoch = i / torch.floor(loader.nb_train)
	local lastepoch = (i - 1) / torch.floor(loader.nb_train)
	
	if torch.floor(epoch) > torch.floor(lastepoch) then -- it's a new epoch!
		if epoch % opt.eval_val_every == 0 then			
			-- evaluate loss on validation data
			local val_loss = eval_split(2) -- 2 = validation
			val_losses[i] = val_loss
			
			print('Validation loss: ' .. tostring(val_loss))
			local errorrates,confuseMatrix3,confuseMatrix4 = evaluate_classification()
			print('Error-rates (train)'); print(errorrates[1])
			print('Error-rates (validation)'); print(errorrates[2])
			print('Error-rates (train_W)'); print(errorrates[3])
			print('Error-rates (validation_W)'); print(errorrates[4])

			local savefile = string.format('%s/lm_%s_epoch%06d_%2.4f_%2.4f_%2.4f_%2.4f_%2.4f.t7', 
				opt.checkpoint_dir, opt.savefile, epoch, val_loss, errorrates[1][1], errorrates[2][1], errorrates[3][1], errorrates[4][1])
			print('saving checkpoint to ' .. savefile)
			local checkpoint = {}
			checkpoint.protos = protos
			checkpoint.opt = opt
			checkpoint.train_losses = train_losses
			checkpoint.val_loss = val_loss
			checkpoint.val_losses = val_losses
			checkpoint.i = i + 1
			checkpoint.epoch = epoch
			--checkpoint.vocab = loader.vocab_mapping
			torch.save(savefile, checkpoint)
			
			local confuseMatrix3_1 = {}
			for i = 1, confuseMatrix3:size(1) do
				confuseMatrix3_1[i] = {}
				for j = 1, confuseMatrix3:size(2) do
					confuseMatrix3_1[i][j] = confuseMatrix3[i][j]
				end
			end
			--csvigo.save{path = savefile..'_train.csv', data = confuseMatrix3_1}
				
			local confuseMatrix4_1 = {}
			for i = 1, confuseMatrix4:size(1) do
				confuseMatrix4_1[i] = {}
				for j = 1, confuseMatrix4:size(2) do
					confuseMatrix4_1[i][j] = confuseMatrix4[i][j]
				end
			end
			csvigo.save{path = savefile..'_val.csv', data = confuseMatrix4_1}
		end	
	
		current_training_batch_number = 0
		loader:next_epoch()
		
		tr_predict:zero()
		tr_predict_W:zero()
		tr_gt_lables:zero()

		val_predict:zero()
		val_predict_W:zero()
		val_gt_lables:zero()		
	end
	
	if i <= iterations then
		local timer = torch.Timer()
		current_training_batch_number = current_training_batch_number + 1
		local _, loss = optim.rmsprop(feval, params, optim_state)
		if opt.accurate_gpu_timing == 1 and opt.gpuid >= 0 then
			--[[
			Note on timing: The reported time can be off because the GPU is invoked async. If one
			wants to have exactly accurate timings one must call cutorch.synchronize() right here.
			I will avoid doing so by default because this can incur computational overhead.
			--]]
			cutorch.synchronize()
		end
		local time = timer:time().real
		
		local train_loss = loss[1] -- the loss is inside a list, pop it
		train_losses[i] = train_loss

		-- exponential learning rate decay
		if i % torch.floor(loader.nb_train) == 0 and opt.learning_rate_decay < 1 then
			if epoch >= opt.learning_rate_decay_after then
				local decay_factor = opt.learning_rate_decay
				optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
                print('current learning rate ' .. optim_state.learningRate)
			end
		end

		if i % opt.print_every == 0 then
			print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.4fs", i, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
		end
	   
		if i % 10 == 0 then collectgarbage() end

		-- handle early stopping if things are going really bad
		if loss[1] ~= loss[1] then
			print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
			break -- halt
		end
		if loss0 == nil then loss0 = loss[1] end
		if loss[1] > loss0 * 3 then
			print('loss is exploding, aborting.')
			break -- halt
		end
	end
end


