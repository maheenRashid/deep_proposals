require 'cunn'
local model = {}
-- warning: module 'data [type 5]' not found
table.insert(model, {'conv1', ccn2.SpatialConvolution(3, 96, 11, 4, 0, 1)})
table.insert(model, {'relu1', nn.ReLU(true)})
table.insert(model, {'norm1', ccn2.SpatialConvolution(3, 96, 11, 4, 0, 1)})
table.insert(model, {'pool1', ccn2.SpatialMaxPooling(3, 2)})
table.insert(model, {'conv2', ccn2.SpatialConvolution(96, 256, 5, 1, 2, 2)})
table.insert(model, {'relu2', nn.ReLU(true)})
table.insert(model, {'norm2', ccn2.SpatialConvolution(96, 256, 5, 1, 2, 2)})
table.insert(model, {'pool2', ccn2.SpatialMaxPooling(3, 2)})
table.insert(model, {'conv3', ccn2.SpatialConvolution(256, 384, 3, 1, 1, 1)})
table.insert(model, {'relu3', nn.ReLU(true)})
table.insert(model, {'conv4', ccn2.SpatialConvolution(384, 384, 3, 1, 1, 2)})
table.insert(model, {'relu4', nn.ReLU(true)})
table.insert(model, {'conv5', ccn2.SpatialConvolution(384, 256, 3, 1, 1, 2)})
table.insert(model, {'relu5', nn.ReLU(true)})
table.insert(model, {'pool5', ccn2.SpatialMaxPooling(3, 2)})
table.insert(model, {'torch_view', nn.View(-1):setNumInputDims(3)})
table.insert(model, {'fc6', nn.Linear(9216, 4096)})
table.insert(model, {'relu6', nn.ReLU(true)})
table.insert(model, {'drop6', nn.Dropout(0.500000)})
table.insert(model, {'fc7', nn.Linear(4096, 4096)})
table.insert(model, {'relu7', nn.ReLU(true)})
table.insert(model, {'drop7', nn.Dropout(0.500000)})
table.insert(model, {'fc8', nn.Linear(4096, 16000)})
-- warning: module 'reshape_coarse_global [type 0]' not found
-- warning: module 'loss_flow_coarse [type 39]' not found
return model