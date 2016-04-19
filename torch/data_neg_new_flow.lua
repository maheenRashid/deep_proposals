
do  
    local data = torch.class('data')

    function data:__init(args)
        -- print ('initing');
        self.file_path_positive=args.file_path_positive
        -- '/disk2/februaryExperiments/deep_proposals/positive_data.txt';
        self.file_path_negative=args.file_path_negative
        -- '/disk2/marchExperiments/deep_proposals/negatives.txt';
        self.includeFlow=true;

        self.batch_size_seg=32;
        self.batch_size_positive_score=16;
        self.batch_size_negative_score=16;


        self.start_idx_train=1;
        self.start_idx_positive_seg=1;
        self.start_idx_positive_score=1;
        self.start_idx_negative_score=1;

        self.params={jitter_size=16,
            crop_size={224,224},
            scale_range={285,224},
            mean={122,117,104},
            tolerance=32,
            tolerance_scale={0.5,2},
            mean_flow={128,128,128},
            max_dim=128};

        self.training_set_seg={};
        self.training_set_score={};
        self.lines_seg=self:readDataFileFlow(self.file_path_positive);
        self.lines_positive=self:readDataFileFlow(self.file_path_positive);
        self.lines_negative=self:readDataFileFlow(self.file_path_negative);
        
        self.lines_seg=self:shuffleLines(self.lines_seg);
        self.lines_positive=self:shuffleLines(self.lines_positive);
        self.lines_negative=self:shuffleLines(self.lines_negative);

        -- local path_to_db_pos='/disk2/februaryExperiments/deep_proposals/positive_data_100.hdf5';
        -- f = hdf5.open(path_to_db_pos, 'r')
        
        -- self.pos_data_db_im=f:read('/images'):all();
        -- self.pos_data_db_mask=f:read('/masks'):all();
        --         im_all = f:read('/images'):partial({1,10},{1,size[2]},{1,size[3]},{1,size[4]});
        -- masks_all = f:read('/masks'):partial({1,10},{1,1},{1,size[3]},{1,size[4]});
        print (#self.lines_seg);
        print (#self.lines_positive);
        print (#self.lines_negative);
        -- return self;
    end


    function data:shuffleLines(lines)
        local x=lines;
        local len=#lines;
        local shuffle = torch.randperm(len)
        local lines={};
        for idx=1,len do
            lines[idx]=x[shuffle[idx]];
        end
        return lines;
    end

    function data:getTrainingDataToyScore()
        local total_batch_size=self.batch_size_positive_score+self.batch_size_negative_score;
        
        local start_idx_negative_score_before = self.start_idx_negative_score
        local start_idx_positive_score_before = self.start_idx_positive_score

        self.training_set_score.data=torch.zeros(total_batch_size,3,self.params.crop_size[1]
            ,self.params.crop_size[2]);
        self.training_set_score.data_flow=torch.zeros(total_batch_size,3,self.params.crop_size[1]
            ,self.params.crop_size[2]);
        self.training_set_score.label=torch.zeros(total_batch_size,1);

        self.start_idx_positive_score=self:addTrainingDataPositive(self.training_set_score,
            self.batch_size_positive_score,
            self.lines_positive,self.start_idx_positive_score,self.params,false)
        
        self.start_idx_negative_score=self:addTrainingDataNegativeScore(self.training_set_score,
            self.batch_size_negative_score,
            self.lines_negative,self.start_idx_negative_score,self.params,self.batch_size_positive_score+1)

        if self.start_idx_negative_score<start_idx_negative_score_before then
            print ('shuffling neg'..self.start_idx_negative_score..' '..start_idx_negative_score_before )
            self.lines_negative=self:shuffleLines(self.lines_negative);
        end

        if self.start_idx_positive_score<start_idx_positive_score_before then
            print ('shuffling pos'..self.start_idx_positive_score..' '..start_idx_positive_score_before )
            self.lines_positive=self:shuffleLines(self.lines_positive);
        end

    end

    function data:getTrainingDataToy()
        local start_idx_positive_seg_before = self.start_idx_positive_seg

        self.training_set_seg.data=torch.zeros(self.batch_size_seg,3,self.params.crop_size[1]
            ,self.params.crop_size[2]);
        self.training_set_seg.label=torch.zeros(self.batch_size_seg,1,self.params.crop_size[1],
        	self.params.crop_size[2]);
        
        self.training_set_seg.data_flow=torch.zeros(self.batch_size_seg,3,self.params.crop_size[1]
            ,self.params.crop_size[2]);

        self.start_idx_positive_seg=self:addTrainingDataPositive(self.training_set_seg,self.batch_size_seg,
        	self.lines_seg,self.start_idx_positive_seg,self.params,true)

        if self.start_idx_positive_seg<start_idx_positive_seg_before then
            print ('shuffling seg'..self.start_idx_positive_seg..' '..start_idx_positive_seg_before )
            self.lines_seg=self:shuffleLines(self.lines_seg);
        end
    end

    function data:readDataFile(file_path)
        -- print(file_path);
        local file_lines = {};
        for line in io.lines(file_path) do 
            -- print(file_path);
            local start_idx, end_idx = string.find(line, ' ');
            -- print(line);
            -- print (start_idx)
            local img_path=string.sub(line,1,start_idx-1);
            local img_label=string.sub(line,end_idx+1,#line);
            file_lines[#file_lines+1]={img_path,img_label};
            -- if #file_lines==100 then
            --     break;
            -- end
            -- print(file_path);
        end 
        -- print(#file_lines);
        return file_lines

    end

    function data:readDataFileFlow(file_path)
        local file_lines = {};
        for line in io.lines(file_path) do 
            -- print(file_path);
            local start_idx, end_idx = string.find(line, ' ');
            -- print(line);
            -- print (start_idx)
            local img_path=string.sub(line,1,start_idx-1);
            local string_temp=string.sub(line,end_idx+1,#line);
            -- print (string_temp)
            local start_idx, end_idx = string.find(string_temp, ' ');

            local img_label=string.sub(string_temp,1,start_idx-1);
            local flow_path=string.sub(string_temp,end_idx+1,#string_temp);

            file_lines[#file_lines+1]={img_path,img_label,flow_path};

            -- if #file_lines==100 then
            --     break;
            -- end
            -- print(file_path);
        end 
        -- print(#file_lines);
        return file_lines
    end

    function data:jitterImage(img,mask,jitter_size,crop_size,flow_img)
        local x=math.random(jitter_size-1);
        local y=math.random(jitter_size-1);
        -- print ('x y '..x..' '..y);
        local x_e=x+crop_size[1];
        local y_e=y+crop_size[2];
        -- print (img:size())
        -- print (type(img))
        local img=image.crop(img:clone(),x,y,x_e,y_e);
        local mask=image.crop(mask:clone(),x,y,x_e,y_e);
        local flow_img=image.crop(flow_img:clone(),x,y,x_e,y_e);
        return img,mask,flow_img
    end

    function data:scaleImage(img,mask,scale_range,crop_size,flow_img)
        local new_scale=math.random(scale_range[1],scale_range[2]);
        -- print ('new_scale '..new_scale);
        local img=image.scale(img,new_scale);
        -- print (img:size())
        local mask=image.scale(mask,new_scale,'simple');
        -- print (mask:size())
        local flow_img=image.scale(flow_img,new_scale);
        -- print (img:size())

        local start_x=math.floor((img:size()[2]-crop_size[1])/2);
        -- print (start_x)
        local start_y=math.floor((img:size()[3]-crop_size[2])/2);
        -- print (start_y);
        local img=image.crop(img,start_x,start_y,start_x+crop_size[1],start_y+crop_size[2]);
        -- print (img:size())
        local mask=image.crop(mask,start_x,start_y,start_x+crop_size[1],start_y+crop_size[2]);
        -- print (mask:size());
        local flow_img=image.crop(flow_img,start_x,start_y,start_x+crop_size[1],start_y+crop_size[2]);
        -- print (img:size())
        
        return img,mask,flow_img
    end


    function data:processImAndMaskPositive(img,mask,params,flow_img)

        -- bring img to 0 255 range
        img:mul(255);
        
        -- bring mask to -1 +1 range
        mask:mul(255);
        mask:mul(2);
        mask:csub(1);

        -- do the same for flow_img
        flow_img:mul(255);

        -- jitter, scale or flip 

        local rand=math.random(3);
        if rand==1 then
            img,mask,flow_img=self:jitterImage(img,mask,params.jitter_size,params.crop_size,flow_img);
        elseif rand==2 then
            img,mask,flow_img=self:scaleImage(img,mask,params.scale_range,params.crop_size,flow_img);
        else
            img,mask,flow_img=self:jitterImage(img,mask,params.jitter_size,params.crop_size,flow_img);
            image.hflip(img,img);
            image.hflip(mask,mask);
            image.hflip(flow_img,flow_img);
        end

        -- subtract the mean
        for i=1,img:size()[1] do
            img[i]:csub(params.mean[i])
        end

        -- subtract the mean for flow_img
        for i=1,flow_img:size()[1] do
            flow_img[i]:csub(params.mean_flow[i])
        end

        -- return
        return img,mask,flow_img

    end


    function data:addTrainingDataPositive(training_set,num_im,list_files,start_idx,params,segFlag)
        local list_idx=start_idx;
        local list_size=#list_files;
        -- local count=0;
        -- print('list_idx_start '..list_idx);
        local curr_idx=1;

        while curr_idx<= num_im do
            -- count=count+1;
            -- add mod length here to make the batch size rotate
            -- print ('curr_idx',curr_idx,'list_idx ',list_idx);

            local img_path=list_files[list_idx][1];
            local mask_path=list_files[list_idx][2];
            local flow_img_path=list_files[list_idx][3];


            local status_img,img=pcall(image.load,img_path);
            local status_flow;
            local flow_img;
            if self.includeFlow then
                status_flow,flow_img=pcall(image.load,flow_img_path);
            else
                status_flow=true;
                -- flow_img=img:clone();
            end
            local status_mask,mask=pcall(image.load,mask_path);
            -- print (curr_idx,status_img,status_mask,status_flow);

            if status_img and status_flow and status_mask then
                if img:size()[1]==1 then
                    img= torch.cat(img,img,1):cat(img,1)
                end
                
                if not self.includeFlow then
                    flow_img= img:clone();
                end


                img,mask,flow_img=self:processImAndMaskPositive(img,mask,params,flow_img)

                training_set.data[curr_idx]=img:int();
                -- print (img:size(),flow_img:size())
                training_set.data_flow[curr_idx]=flow_img:int();
                if segFlag then
                    training_set.label[curr_idx]=mask;
                else
                    training_set.label[curr_idx][1]=1;
                end
            else
                print ('PROBLEM READING INPUT');
                curr_idx=curr_idx-1;
            end
            
            list_idx=(list_idx%list_size)+1;
            curr_idx=curr_idx+1;
        end
        -- print( count);
        -- print('list_idx_end '..list_idx);
        return list_idx;
    end


    -- function data:addTrainingDataPositive(training_set,num_im,list_files,start_idx,params,segFlag)
    --     local list_idx=start_idx;
    --     local list_size=#list_files;
    --     -- local count=0;
    --     -- print('list_idx_start '..list_idx);
    --     for curr_idx=1,num_im do
    --         -- count=count+1;
    --         -- add mod length here to make the batch size rotate
    --         -- print ('list_idx '..list_idx);
    --         local img_path=list_files[list_idx][1];
    --         local mask_path=list_files[list_idx][2];
    --         local flow_img_path=list_files[list_idx][3];

    --         local img=image.load(img_path);
    --         local flow_img=image.load(flow_img_path);

    --         if img:size()[1]==1 then
    --             img= torch.cat(img,img,1):cat(img,1)

    --         end
    --         local mask=image.load(mask_path);
            
    --         img,mask,flow_img=self:processImAndMaskPositive(img,mask,params,flow_img)

    --         training_set.data[curr_idx]=img:int();
    --         training_set.data_flow[curr_idx]=flow_img:int();
    --         if segFlag then
    --             training_set.label[curr_idx]=mask;
    --         else
    --             training_set.label[curr_idx][1]=1;
    --         end
    --         list_idx=(list_idx%list_size)+1;
    --     end
    --     -- print( count);
    --     -- print('list_idx_end '..list_idx);
    --     return list_idx;
    -- end

    function data:scaleNegImage(img_org,bbox,flow_img_org)
        local scale_factor=math.random();
        scale_factor = (scale_factor*1.5)+0.5;
        local img=image.scale(img_org,'*'..scale_factor)
        local flow_img=image.scale(flow_img_org,'*'..scale_factor)
        bbox=torch.mul(bbox,scale_factor);
        bbox=bbox:floor();
        -- print ('scale factor',scale_factor);
        return img,bbox,flow_img;
    end


    function data:addTrainingDataNegativeScore(training_set,num_im,list_files,start_idx,params,training_data_idx)
        local list_idx=start_idx;
        local list_size=#list_files;
        
        local curr_idx= training_data_idx;
        while curr_idx<=training_data_idx+num_im-1 do
            -- print ('curr_idx',curr_idx,'list_idx',list_idx);

            local img_path=list_files[list_idx][1];
            local npy_path=list_files[list_idx][2];
            local flow_path=list_files[list_idx][3];
            
            local status_img,img_org=pcall(image.load,img_path);
            local status_flow;
            local flow_org;
            if self.includeFlow then
                status_flow,flow_org=pcall(image.load,flow_path);
            else
                status_flow=true;
            end

            if status_img and status_flow then
                -- make sure image has 3 channels
                if img_org:size()[1]==1 then
                    img_org= torch.cat(img_org,img_org,1):cat(img_org,1)
                end

                if not self.includeFlow then
                    flow_org=img_org:clone();
                end

                -- make sure image has range 255
                img_org:mul(255);
                -- subtract mean
                for i=1,img_org:size()[1] do
                    img_org[i]:csub(params.mean[i])
                end

                -- do the same for flow image
                flow_org=flow_org:mul(255);
                for i=1,flow_org:size()[1] do
                    flow_org[i]:csub(params.mean_flow[i])
                end

                -- load bbox and make in to int
                local bbox_org=npy4th.loadnpy(npy_path);
                bbox_org=bbox_org:floor();

                local crop_box={};
                local img;
                local bbox;
                local counter=0;
                local counter_lim=1000;
                local no_neg=false;
                while 1 do
                    --scale the image and the bbox
                    img,bbox,flow_img=self:scaleNegImage(img_org:clone(),bbox_org:clone(),flow_org:clone());
                    if counter>=counter_lim then
                        print 'NO NEG OUTER'
                        no_neg=true;
                        break;
                    end

                    -- get max starting point of crop
                    local x_max=img:size()[3]-params.crop_size[2];
                    local y_max=img:size()[2]-params.crop_size[1];
                    if x_max>1 and y_max>1 then
                        -- create a valid crop box that does not violate any positive examples
                        local start_x;
                        local start_y;
                        
                        
                        start_x=torch.random(0,x_max-1);
                        start_y=torch.random(0,y_max-1);
                        local end_x=start_x+params.crop_size[2];
                        local end_y=start_y+params.crop_size[1];
                        crop_box={start_x,start_y,end_x,end_y};
                        
                        -- check if crop is valid negative
                        if self:isValidNegative(crop_box,bbox,params.max_dim,params.tolerance,params.tolerance_scale,params.crop_size) then
                            break;
                        end
                    end
                    counter=counter+1;
                end
                if no_neg then
                    curr_idx=curr_idx-1;
                else
                    -- add to training data
                    local img_crop= image.crop(img,crop_box[1],crop_box[2],crop_box[3],crop_box[4]);
                    local flow_img_crop= image.crop(flow_img,crop_box[1],crop_box[2],crop_box[3],crop_box[4]);
                    if math.random()<=0.5 then
                        img_crop=image.hflip(img_crop);
                        flow_img_crop=image.hflip(flow_img_crop);
                    end
                    -- figure out how to dstack flow_img_crop and img_crop
                    training_set.data[curr_idx]=img_crop:int();
                    training_set.data_flow[curr_idx]=flow_img_crop:int();
                    training_set.label[curr_idx][1]=-1;

                end
            else
                print ('PROBLEM READING INPUT');
                curr_idx=curr_idx-1;
            end
            list_idx=(list_idx%list_size)+1;
            curr_idx=curr_idx+1;
        end
        
        return list_idx;  
    end

    -- function data:addTrainingDataNegativeScore(training_set,num_im,list_files,start_idx,params,training_data_idx)
    --     local list_idx=start_idx;
    --     local list_size=#list_files;
        
    --     local curr_idx= training_data_idx;
    --     while curr_idx<=training_data_idx+num_im-1 do
        
    --         local img_path=list_files[list_idx][1];
    --         local npy_path=list_files[list_idx][2];
    --         local img_org=image.load(img_path);

    --         -- make sure image has 3 channels
    --         if img_org:size()[1]==1 then
    --             img_org= torch.cat(img_org,img_org,1):cat(img_org,1)
    --         end
    --         -- make sure image has range 255
    --         img_org:mul(255);
    --         -- subtract mean
    --         for i=1,img_org:size()[1] do
    --             img_org[i]:csub(params.mean[i])
    --         end

    --         -- do the same for flow image
    --         local flow_path=list_files[list_idx][3];
    --         local flow_org=image.load(flow_path);
    --         flow_org=flow_org:mul(255);
    --         for i=1,flow_org:size()[1] do
    --             flow_org[i]:csub(params.mean_flow[i])
    --         end

    --         -- load bbox and make in to int
    --         local bbox_org=npy4th.loadnpy(npy_path);
    --         bbox_org=bbox_org:floor();

    --         local crop_box={};
    --         local img;
    --         local bbox;
    --         local counter=0;
    --         local counter_lim=1000;
    --         local no_neg=false;
    --         while 1 do
    --             --scale the image and the bbox
    --             img,bbox,flow_img=self:scaleNegImage(img_org:clone(),bbox_org:clone(),flow_org:clone());
    --             if counter>=counter_lim then
    --                 print 'NO NEG OUTER'
    --                 no_neg=true;
    --                 break;
    --             end

    --             -- get max starting point of crop
    --             local x_max=img:size()[3]-params.crop_size[2];
    --             local y_max=img:size()[2]-params.crop_size[1];
    --             if x_max>1 and y_max>1 then
    --                 -- create a valid crop box that does not violate any positive examples
    --                 local start_x;
    --                 local start_y;
                    
                    
    --                 start_x=torch.random(0,x_max-1);
    --                 start_y=torch.random(0,y_max-1);
    --                 local end_x=start_x+params.crop_size[2];
    --                 local end_y=start_y+params.crop_size[1];
    --                 crop_box={start_x,start_y,end_x,end_y};
                    
    --                 -- check if crop is valid negative
    --                 if self:isValidNegative(crop_box,bbox,params.max_dim,params.tolerance,params.tolerance_scale,params.crop_size) then
    --                     break;
    --                 end
    --             end
    --             counter=counter+1;
    --         end
    --         if no_neg then
    --             curr_idx=curr_idx-1;
    --         else
    --             -- add to training data
    --             local img_crop= image.crop(img,crop_box[1],crop_box[2],crop_box[3],crop_box[4]);
    --             local flow_img_crop= image.crop(flow_img,crop_box[1],crop_box[2],crop_box[3],crop_box[4]);
    --             if math.random()<=0.5 then
    --                 img_crop=image.hflip(img_crop);
    --                 flow_img_crop=image.hflip(flow_img_crop);
    --             end
    --             -- figure out how to dstack flow_img_crop and img_crop
    --             training_set.data[curr_idx]=img_crop:int();
    --             training_set.data_flow[curr_idx]=flow_img_crop:int();
    --             training_set.label[curr_idx][1]=-1;

    --         end
    --         list_idx=(list_idx%list_size)+1;
    --         curr_idx=curr_idx+1;
    --     end
        
    --     return list_idx;  
    -- end  

    function data:isValidNegative(crop_box,bbox_all,max_dim,tolerance_distance,tolerance_scale,crop_size)
        local isValid=true;
        for bbox_idx=1,bbox_all:size(1) do
            local bbox=bbox_all[bbox_idx];
            local dims={bbox[3],bbox[4]};
            local max_dim_pos=torch.max(torch.Tensor(dims));
            local min_dim_pos=torch.min(torch.Tensor(dims));
            local scale=max_dim/max_dim_pos;
            local new_tolerance=tolerance_distance/scale;
        
            local center_box=torch.Tensor({bbox[1]+bbox[3]/2.0,bbox[2]+bbox[4]/2.0});
            local center_crop=torch.Tensor({crop_box[1]+(crop_box[3]-crop_box[1])/2,crop_box[2]+(crop_box[4]-crop_box[2])/2});
            local dist_centers=torch.sqrt(torch.sum(torch.pow(torch.csub(center_box,center_crop),2)));
            
            if dist_centers<new_tolerance then
                local new_crop_size_x=crop_size[1]/scale;
                local new_crop_size_y=crop_size[2]/scale;
                local area_pos=new_crop_size_x*new_crop_size_y;
                local area_crop=(crop_box[3]-crop_box[1])*(crop_box[4]-crop_box[2]);
                local scale_diff=area_crop/area_pos;
                if scale_diff<tolerance_scale[2] and scale_diff>tolerance_scale[1] then
                    isValid=false;

                end
            end
            if isValid==false then
                break;
            end
        end
        return isValid;
    end

end

return data