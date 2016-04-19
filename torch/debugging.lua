do  
    local data = torch.class('data')

    function data:__init(args)
		local img_path='/disk2/aprilExperiments/deep_proposals/flow_all_humans/results_flow/COCO_train2014_000000536498_292472_flow.png';
		local im = image.read(img_path);
	end

end

check=data();