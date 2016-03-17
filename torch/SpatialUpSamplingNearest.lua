require 'image'
local SpatialUpSamplingNearest, parent = torch.class('nn.SpatialUpSamplingNearest', 'nn.Module')

--[[
Applies a 2D up-sampling over an input image composed of several input planes.

The upsampling is done using the simple nearest neighbor technique.

The Y and X dimensions are assumed to be the last 2 tensor dimensions.  For
instance, if the tensor is 4D, then dim 3 is the y dimension and dim 4 is the x.

owidth  = width*scale_factor
oheight  = height*scale_factor
--]]

function SpatialUpSamplingNearest:__init(scale)
   parent.__init(self)

   self.scale_factor = scale
   if self.scale_factor < 1 then
     error('scale_factor must be greater than 1')
   end
   if math.floor(self.scale_factor) ~= self.scale_factor then
     error('scale_factor must be integer')
   end
   self.inputSize = torch.LongStorage(4)
   self.outputSize = torch.LongStorage(4)
   self.usage = nil
end

function SpatialUpSamplingNearest:updateOutput(input)
   if input:dim() ~= 4 and input:dim() ~= 3 then
     error('SpatialUpSamplingNearest only support 3D or 4D tensors')
   end
   
   local xdim = input:dim()
   local ydim = input:dim() - 1
   for i = 1, input:dim() do
     self.inputSize[i] = input:size(i)
     self.outputSize[i] = input:size(i)
   end
   self.outputSize[ydim] = self.outputSize[ydim] * self.scale_factor
   self.outputSize[xdim] = self.outputSize[xdim] * self.scale_factor
   
   -- Resize the output if needed
   if input:dim() == 3 then
        self.output:resize(self.outputSize[1], self.outputSize[2],
         self.outputSize[3], self.outputSize[4])
        if input:type()=='torch.CudaTensor' then
            input_temp=input:double();
            output_temp=image.scale(input_temp,self.outputSize[2],self.outputSize[3],'bilinear');
            self.output = output_temp:typeAs(input);
        else
            self.output=image.scale(input,self.outputSize[2],self.outputSize[3],'bilinear');
        end
        -- print('self.output:size() '..self.output:size());
   -- input.nn.SpatialUpSamplingNearest_updateOutput(self, input)
   else
        self.output:resize(self.outputSize)
        print('output:size() ')
        print (self.outputSize)
        if input:type()=='torch.CudaTensor' then
            input_temp=input:double();
            for im_no=1,self.outputSize[1] do
                output_temp=image.scale(input_temp[im_no],self.outputSize[3],self.outputSize[4],'bilinear');
                self.output[im_no] = output_temp:typeAs(input);
            end
        else
            for im_no=1,self.outputSize[1] do
              self.output[im_no] =image.scale(input_temp[im_no],self.outputSize[3],self.outputSize[4],'bilinear');
            end
    end
    -- print('self.output:size() '..self.output:size());
     -- input.nn.SpatialUpSamplingNearest_updateOutput(self, input)
   
     -- self.output:resize(self.outputSize)
     -- input.nn.SpatialUpSamplingNearest_updateOutput(self, input)
   end
    
   return self.output
end

function SpatialUpSamplingNearest:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)
   input.nn.SpatialUpSamplingNearest_updateGradInput(self, input, gradOutput)
   return self.gradInput
end
