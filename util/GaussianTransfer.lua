local GaussianTransfer, parent = torch.class('nn.GaussianTransfer', 'nn.Module')

function GaussianTransfer:__init()
    parent.__init(self)
	
	self.gaussianParam0 = 2.718281828459 -- e
	self.gaussianParam1 = -0.5 -- must be negative
end
   
function GaussianTransfer:updateOutput(input)
    self.output:resizeAs(input):copy(input)
	self.output:cmul(self.output) -- x ^ 2
	self.output:mul(self.gaussianParam1) -- p1 * (x ^ 2)
	local eTensor = self.output:clone():fill(self.gaussianParam0)
	eTensor:cpow(self.output) -- e ^ ( p1 * (x ^ 2) )
	self.output = eTensor
    
    return self.output
end

function GaussianTransfer:updateGradInput(input, gradOutput)
	if self.gradInput then
        self.gradInput:resizeAs(gradOutput):copy(gradOutput) 
    else
        self.gradInput = torch.Tensor():resizeAs(gradOutput):copy(gradOutput) 
    end
	
	self.gradInput:mul(2*self.gaussianParam1) -- 2 * p1
	self.gradInput:cmul(input) -- 2 * p1 * x
	self.gradInput:cmul(self.output) -- (2 * p1 * x) * e ^ ( p1 * (x ^ 2) )
	
	return self.gradInput
end

