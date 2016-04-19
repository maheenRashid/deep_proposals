npy4th = require 'npy4th'
require 'image'

do  
    local data = torch.class('data')

    function data:__init(args)
        -- print ('initing');
        self.file_path_positive='/disk2/februaryExperiments/deep_proposals/positive_data.txt';
        self.file_path_negative='/disk2/marchExperiments/deep_proposals/negatives.txt';

        self.batch_size_seg=50;
        self.batch_size_positive_score=25;
        self.batch_size_negative_score=25;


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
            max_dim=128};
        self.training_set_seg={};
        self.training_set_score={};
        self.lines_seg=self:readDataFile(self.file_path_positive,self.batch_size_seg);
        self.lines_positive=self:readDataFile(self.file_path_positive,self.batch_size_positive_score);
        self.lines_negative=self:readDataFile(self.file_path_negative,self.batch_size_negative_score);
        
        self.lines_seg=self:shuffleLines(self.lines_seg);
        self.lines_positive=self:shuffleLines(self.lines_positive);
        -- self.lines_negative=self:shuffleLines(self.lines_negative);

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

        self.start_idx_positive_seg=self:addTrainingDataPositive(self.training_set_seg,self.batch_size_seg,
        	self.lines_seg,self.start_idx_positive_seg,self.params,true)

        if self.start_idx_positive_seg<start_idx_positive_seg_before then
            print ('shuffling seg'..self.start_idx_positive_seg..' '..start_idx_positive_seg_before )
            self.lines_seg=self:shuffleLines(self.lines_seg);
        end
    end

    function data:getTrainingDataToyDB()
        self.training_set_seg.data=torch.zeros(self.batch_size_seg,3,self.params.crop_size[1],self.params.crop_size[2]);
        self.training_set_seg.label=torch.zeros(self.batch_size_seg,1,self.params.crop_size[1],
            self.params.crop_size[2]);

        self.start_idx_positive_seg=self:addTrainingDataPositiveDB(self.training_set_seg,self.batch_size_seg,
            self.lines,self.start_idx_positive_seg,self.params,true)
    end

    function data:readDataFile(file_path,num_to_read)
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
            if #file_lines==num_to_read then
                break;
            end
            -- print(file_path);
        end 
        -- print(#file_lines);
        return file_lines
    end

    function data:unique(input)
      local b = {}
      -- print (input:numel())
      local input=torch.reshape(input, input:numel())
      -- print (input:size());
      for i=1,input:numel() do
            -- print (i)
            -- print (input[i])
            if input[i]==-1 then
                b[100]=true;
            else
                b[input[i]] = true
            end
       end
      local out = {}
      for i in pairs(b) do
          table.insert(out,i)
       end
      return out
    end

    function data:jitterImage(img,mask,jitter_size,crop_size)
        local x=math.random(jitter_size-1);
        local y=math.random(jitter_size-1);
        -- print ('x y '..x..' '..y);
        local x_e=x+crop_size[1];
        local y_e=y+crop_size[2];
        -- print (img:size())
        -- print (type(img))
        local img=image.crop(img:clone(),x,y,x_e,y_e);
        local mask=image.crop(mask:clone(),x,y,x_e,y_e);
        return img,mask
    end

    function data:scaleImage(img,mask,scale_range,crop_size)
        local new_scale=math.random(scale_range[1],scale_range[2]);
        -- print ('new_scale '..new_scale);
        local img=image.scale(img,new_scale);
        -- print (img:size())
        local mask=image.scale(mask,new_scale,'simple');
        -- print (mask:size())
        local start_x=math.floor((img:size()[2]-crop_size[1])/2);
        -- print (start_x)
        local start_y=math.floor((img:size()[3]-crop_size[2])/2);
        -- print (start_y);
        local img=image.crop(img,start_x,start_y,start_x+crop_size[1],start_y+crop_size[2]);
        -- print (img:size())
        local mask=image.crop(mask,start_x,start_y,start_x+crop_size[1],start_y+crop_size[2]);
        -- print (mask:size());
        return img,mask
    end

    function data:processImAndMaskPositive(img,mask,params)

        if img:size()[1]==1 then
            img= torch.cat(img,img,1):cat(img,1)
        end

        -- bring img to 0 255 range
        img:mul(255);
        
        -- bring mask to -1 +1 range
        mask:mul(255);
        mask:mul(2);
        mask:csub(1);

        -- jitter, scale or flip 

        local rand=math.random(3);
        if rand==1 then
            img,mask=self:jitterImage(img,mask,params.jitter_size,params.crop_size);
        elseif rand==2 then
            img,mask=self:scaleImage(img,mask,params.scale_range,params.crop_size);
        else
            img,mask=self:jitterImage(img,mask,params.jitter_size,params.crop_size);
            image.hflip(img,img);
            image.hflip(mask,mask);
        end

        -- subtract the mean
        for i=1,img:size()[1] do
            img[i]:csub(params.mean[i])
        end
        -- return
        return img,mask

    end

    function data:addTrainingDataPositiveDB(training_set,num_im,list_files,start_idx,params,segFlag)
        local list_idx=start_idx;
        local list_size=#list_files;
        -- local count=0;
        -- print('list_idx_start '..list_idx);
        for curr_idx=1,num_im do
            -- count=count+1;
          

            local img = self.pos_data_db_im[list_idx]:double();
            local mask = self.pos_data_db_mask[list_idx]:double();
            -- print (torch.min(img)..' '..torch.max(img))
            -- print (torch.min(mask)..' '..torch.max(mask))
            
            img,mask=self:processImAndMaskPositive(img,mask,params)
            -- print (torch.min(img)..' '..torch.max(img))
            -- print (torch.min(mask)..' '..torch.max(mask))
            
            training_set.data[curr_idx]=img;
            if segFlag then
                training_set.label[curr_idx]=mask;
            else
                training_set.label[curr_idx][1]=1;
            end
            list_idx=(list_idx%list_size)+1;
        end
        -- print( count);
        -- print('list_idx_end '..list_idx);
        return list_idx;
    end

    function data:addTrainingDataPositive(training_set,num_im,list_files,start_idx,params,segFlag)
        local list_idx=start_idx;
        local list_size=#list_files;
        -- local count=0;
        -- print('list_idx_start '..list_idx);
        for curr_idx=1,num_im do
            -- count=count+1;
            -- add mod length here to make the batch size rotate
            -- print ('list_idx '..list_idx);
            local img_path=list_files[list_idx][1];
            local mask_path=list_files[list_idx][2];
            local img=image.load(img_path);
            if img:size()[1]==1 then
                img= torch.cat(img,img,1):cat(img,1)
            end
            local mask=image.load(mask_path);
            
            img,mask=self:processImAndMaskPositive(img,mask,params)
            training_set.data[curr_idx]=img;
            if segFlag then
                training_set.label[curr_idx]=mask;
            else
                training_set.label[curr_idx][1]=1;
            end
            list_idx=(list_idx%list_size)+1;
        end
        -- print( count);
        -- print('list_idx_end '..list_idx);
        return list_idx;
    end

    function data:isValidNegative(crop_coord,bbox,tolerance)
        local start_x=crop_coord[1];
        local start_y=crop_coord[2];
        local end_x=crop_coord[3];
        local end_y=crop_coord[4];
        local isValid=true;
        for box_no=1,bbox:size()[1] do
            local x_min_b=bbox[box_no][1];
            local y_min_b=bbox[box_no][2];
            local x_max_b=x_min_b+bbox[box_no][3];
            local y_max_b=y_min_b+bbox[box_no][4];
            local x_dim_b=bbox[box_no][3];
            local y_dim_b=bbox[box_no][4];
            local max_dim=(x_dim_b>y_dim_b) and x_dim_b or y_dim_b;
            -- is it entirely contained in the crop?
            if x_min_b>=start_x and y_min_b>=start_y and x_max_b<=end_x and y_max_b<=end_y then
                -- is it not too big or not too small
                if max_dim<128*2 and max_dim>128/2 then
                    -- is it offset enough?
                    if x_min_b-start_x<tolerance or y_min_b-start_y<tolerance then
                        isValid=false;
                        -- print ('isValid');
                        -- print (box_no);
                        break;
                    end
                end
            end
        end
        return isValid;
    end

    function data:scaleNegImage(img_org,bbox,crop_size)
        local scale_factor=math.random();
        scale_factor = (scale_factor*1.5)+0.5;

        local x_dim_new = math.floor(img_org:size()[3]*scale_factor)
        local y_dim_new = math.floor(img_org:size()[2]*scale_factor)

        -- print (img_org:size());
        -- print (scale_factor..' '..x_dim_new..' '..y_dim_new);

        -- img=image.scale(img_org,'*'..scale_factor)
        -- print (img:size())
        local counter =0;
        local counter_lim=100;
        local no_neg=false;

        -- x_dim_new=crop_size[2]-1;
        -- y_dim_new=crop_size[1]-1;
            
        while x_dim_new<=crop_size[2] or y_dim_new<=crop_size[1] do
            -- print ('counter inner '..counter);
            if counter>=counter_lim then
                print 'NO NEG INNER'
                no_neg=true;
                break;
            end

            scale_factor = math.random();
            scale_factor = (scale_factor*1.5)+0.5;

            x_dim_new = math.floor(img_org:size()[3]*scale_factor)
            y_dim_new = math.floor(img_org:size()[2]*scale_factor)
            -- print (scale_factor..' '..x_dim_new..' '..y_dim_new);
            -- min_dim = math.min(x_dim_new,y_dim_new);                

            -- x_dim_new=crop_size[2]-1;
            -- y_dim_new=crop_size[1]-1;
            counter=counter+1;
        end

        if no_neg then
            img=-1;
        else
            img=image.scale(img_org,'*'..scale_factor)
            -- print (img:size())
            bbox=torch.mul(bbox,scale_factor);
            bbox=bbox:floor();
        end
        return img,bbox

    end


    function data:scaleNegImageNew(img_org,bbox)
        local scale_factor=math.random();
        -- scale_factor = (scale_factor*1.5)+0.5;
        scale_factor=1.6913094386469;
        local img=image.scale(img_org,'*'..scale_factor)
        -- print (img:size())
        bbox=torch.mul(bbox,scale_factor);
        bbox=bbox:floor();
        print ('scale factor',scale_factor);
        return img,bbox
        -- ,scale_factor
    end

    function data:addTrainingDataNegativeScore(training_set,num_im,list_files,start_idx,params,training_data_idx)
        local list_idx=start_idx;
        local list_size=#list_files;
        -- print('list_idx_start '..list_idx);
        local curr_idx= training_data_idx;
        while curr_idx<=training_data_idx+num_im-1 do
            -- print('curr_idx '..curr_idx..'list_idx '..list_idx);

            local img_path=list_files[list_idx][1];
            local npy_path=list_files[list_idx][2];
            local img_org=image.load(img_path);

            -- make sure image has 3 channels
            if img_org:size()[1]==1 then
                img_org= torch.cat(img_org,img_org,1):cat(img_org,1)
            end
            -- make sure image has range 255
            img_org:mul(255);
            -- subtract mean
            for i=1,img_org:size()[1] do
                img_org[i]:csub(params.mean[i])
            end
            -- load bbox and make in to int
            local bbox_org=npy4th.loadnpy(npy_path);
            bbox_org=bbox_org:floor();

            local crop_box={};
            local img;
            local bbox;
            local counter=0;
            local counter_lim=100;
            local no_neg=false;
            while 1 do
                --scale the image and the bbox
                img,bbox=self:scaleNegImage(img_org:clone(),bbox_org:clone(),params.crop_size);
                if counter>=counter_lim or img==-1 then
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
                    if self:isValidNegative(crop_box,bbox,params.tolerance) then
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
                if math.random()<=0.5 then
                    img_crop=image.hflip(img_crop);
                end
                training_set.data[curr_idx]=img_crop;
                training_set.label[curr_idx][1]=-1;
            end
            list_idx=(list_idx%list_size)+1;
            curr_idx=curr_idx+1;
        end
        
        return list_idx;  
    end  

    function data:testNegScaling(list_files,out_dir)
        list_idx=1;
        local img_path=list_files[list_idx][1];
        local npy_path=list_files[list_idx][2];
        local img=image.load(img_path);

        -- make sure image has 3 channels
        if img:size()[1]==1 then
            img= torch.cat(img,img,1):cat(img,1)
        end
        
        -- load bbox and make in to int
        local bbox=npy4th.loadnpy(npy_path)
        bbox:floor();
        
        --scale the image and the bbox
        for i=1,10 do
            if i==1 then
                scale_factor=1.0;
            else
                scale_factor=math.random();
                scale_factor=(scale_factor*1.5)+0.5;
            end
            
            print (scale_factor);
            print(img:size())
            -- dims=torch.mul(img:size():double(),scale_factor)
            -- print(dims)
            local x_dim_new=math.floor(img:size()[3]*scale_factor)
            local y_dim_new= math.floor(img:size()[2]*scale_factor)
            local max_dim = math.max(x_dim_new,y_dim_new);
            print (x_dim_new..' '..y_dim_new..' '..max_dim);
            img_curr=image.scale(img,max_dim)
            print(img_curr:size())
            bbox_curr=torch.mul(bbox,scale_factor);
            -- 
            img_name=out_dir..'/img_'..i..'.png';
            

            -- visualize the scaled image and bbox
            image.save(img_name,img_curr);
            print(bbox_curr);
            print (img_curr:size())
            print (bbox_curr:size());
            bbox_curr=bbox_curr:floor();
            print (bbox_curr:size());
            for row=1,bbox_curr:size(1) do
                crop_name=out_dir..'/img_crop_'..row..'_'..i..'.png';
                bbox_curr_curr=bbox_curr[row];
                crop_curr=image.crop(img_curr, bbox_curr_curr[1], bbox_curr_curr[2], 
                        bbox_curr_curr[1]+bbox_curr_curr[3], bbox_curr_curr[2]+bbox_curr_curr[4]);
                image.save(crop_name,crop_curr);
            end
        end

    end


    function data:scaleTest(bbox,crop_box,tolerance_scale)
        local area_pos=bbox[3]*bbox[4];
        local area_crop=(crop_box[3]-crop_box[1])*(crop_box[4]-crop_box[2]);
        local scale_diff=area_crop/area_pos;
        isValid=true;

        if scale_diff<tolerance_scale[2] and scale_diff>tolerance_scale[1] then
            isValid=false;
        end
        -- print (area_pos,area_crop,scale_diff,isValid);
        return isValid;
    end

    function data:isValidNegativeNew(crop_box,bbox_all,max_dim,tolerance_distance,tolerance_scale)
        local isValid=true;
        for bbox_idx=1,1 do
            -- bbox_all:size(1) do
            -- print ('bbox_idx,isValid',bbox_idx,isValid)
            local bbox=bbox_all[bbox_idx];
            local dims={bbox[3],bbox[4]};
            local max_dim_pos=torch.max(torch.Tensor(dims));
            local min_dim_pos=torch.min(torch.Tensor(dims));
            local scale=max_dim/max_dim_pos;
            local new_tolerance=tolerance_distance/scale;
        
            local center_box=torch.Tensor({bbox[1]+bbox[3]/2.0,bbox[2]+bbox[4]/2.0});
            local center_crop=torch.Tensor({crop_box[1]+(crop_box[3]-crop_box[1])/2,crop_box[2]+(crop_box[4]-crop_box[2])/2});
            local dist_centers=torch.sqrt(torch.sum(torch.pow(torch.csub(center_box,center_crop),2)));
            -- local isValid=true;
            
            -- local new_crop_size=224/scale;
            -- print (scale,max_dim_pos,max_dim_pos*scale,new_crop_size)



            -- print (center_box,center_crop)
            -- print (new_tolerance,dist_centers,dist_centers<new_tolerance)

            if dist_centers<new_tolerance then
                local new_crop_size=224/scale;
                -- local r1=new_crop_size-max_dim_pos;
                -- local r2=new_crop_size-min_dim_pos;
                -- local area_pos_conv=bbox[3]*bbox[4];
                local area_pos=new_crop_size*new_crop_size;
                local area_crop=(crop_box[3]-crop_box[1])*(crop_box[4]-crop_box[2]);
                local scale_diff=area_crop/area_pos;
                -- isValid=true;
                -- print (scale_diff);
                if scale_diff<tolerance_scale[2] and scale_diff>tolerance_scale[1] then
                    isValid=false;

                end
                -- print (area_pos_conv,area_crop/area_pos_conv,area_pos,area_crop,scale_diff,isValid);
            end
            if isValid==false then
                break;
            end
            -- break;
        end
        return isValid;
    end

    function data:getNegData(num_im,list_files,start_idx,params)
        local list_idx=start_idx;
        local list_size=#list_files;
        print (list_idx);
        local img_path=list_files[list_idx][1];
        print (list_idx,img_path)
        local npy_path=list_files[list_idx][2];
        local img_org=image.load(img_path);

        -- make sure image has 3 channels
        if img_org:size()[1]==1 then
            img_org= torch.cat(img_org,img_org,1):cat(img_org,1)
        end
        
        local bbox_org=npy4th.loadnpy(npy_path);
        bbox_org=bbox_org:floor();

        -- return img_org,bbox_org,{1,1,1,1}
        -- ,{0};

        local img
        -- =img_org:clone();
        local bbox
        -- =bbox_org:clone();
        img,bbox=self:scaleNegImageNew(img_org:clone(),bbox_org:clone());

        local start_idx_x=1;
        local start_idx_y=1;
    
        local end_idx_x=img:size(3)-params.crop_size[2];
        local end_idx_y=img:size(2)-params.crop_size[1];

        print ('start_idx_x,start_idx_y,end_idx_x,end_idx_y');
        print (start_idx_x,start_idx_y,end_idx_x,end_idx_y);

        local crop_boxes={};
        for start_idx_x=1,end_idx_x do
            for start_idx_y=1,end_idx_y do
                local crop_box_curr={start_idx_x,start_idx_y,start_idx_x+params.crop_size[1],start_idx_y+params.crop_size[2]};
                crop_boxes[#crop_boxes+1]=crop_box_curr;
            end
        end

        print ('#crop_boxes',#crop_boxes);
        local crop_box_pos={};
        local crop_box_neg={};
        for idx=1,#crop_boxes do
            local crop_box=crop_boxes[idx];
            if self:isValidNegativeNew(crop_box,bbox,params.max_dim,params.tolerance,params.tolerance_scale) then
                crop_box_pos[#crop_box_pos+1]=crop_box;
            else
                crop_box_neg[#crop_box_neg+1]=crop_box;
            end
        end

        print ('#crop_box_pos',#crop_box_pos);
        print ('#crop_box_neg',#crop_box_neg);

        -- local crop_box={};
        -- local img;
        -- local bbox;
        -- local scale;
        -- local counter=0;
        -- local counter_lim=100;
        -- local no_neg=false;

        -- while 1 do
        
        --     img,bbox=self:scaleNegImageNew(img_org:clone(),bbox_org:clone());
            
        --     if counter>=counter_lim or img==-1 then
        --         print 'NO NEG OUTER'
        --         no_neg=true;
        --         break;
        --     end

        --     -- get max starting point of crop
        --     local x_max=img:size()[3]-params.crop_size[2];
        --     local y_max=img:size()[2]-params.crop_size[1];
            
        --     if x_max>1 and y_max>1 then
        --         -- create a valid crop box that does not violate any positive examples
        --         local start_x;
        --         local start_y;
                
                
        --         start_x=torch.random(0,x_max-1);
        --         start_y=torch.random(0,y_max-1);
        --         local end_x=start_x+params.crop_size[2];
        --         local end_y=start_y+params.crop_size[1];
        --         crop_box={start_x,start_y,end_x,end_y};
                
        --         -- check if crop is valid negative
        --         if self:isValidNegativeNew(crop_box,bbox,params.max_dim,params.tolerance,params.tolerance_scale) then
        --             break;
        --         end
        --     end
        --     counter=counter+1;
        -- end

        -- if no_neg then
        --     crop_box=torch.Tensor({1,1,1,1})
        -- end

        return img,bbox,crop_box_pos,crop_box_neg
    end  

    function data:saveNegData()
        local out_dir='/disk2/aprilExperiments/testing_neg_fixed_test/';
        num_im=100;
        for idx=1,1 do
            -- num_im do
            -- img,bbox,crop_box=self:getNegData(num_im,self.lines_negative,idx%(#self.lines_negative)+1,self.params)
            img,bbox,crop_box_pos,crop_box_neg=self:getNegData(num_im,self.lines_negative,6,self.params)
            -- print (crop_box,torch.Tensor(crop_box))
            out_file_im=out_dir..idx..'.png';
            out_file_bbox=out_dir..idx..'_bbox.npy';
            out_file_crop=out_dir..idx..'_crop.npy';
            
            out_file_crop_pos=out_dir..idx..'_crop_pos.npy';
            out_file_crop_neg=out_dir..idx..'_crop_neg.npy';
            
            image.save(out_file_im,img);
            npy4th.savenpy(out_file_bbox,bbox);
            -- npy4th.savenpy(out_file_crop,torch.Tensor(crop_box));

            npy4th.savenpy(out_file_crop_pos,torch.Tensor(crop_box_pos));
            npy4th.savenpy(out_file_crop_neg,torch.Tensor(crop_box_neg));
            -- break;
        end
    end

    function data:readNegData()
        local out_dir='/disk2/aprilExperiments/testing_neg_fixed_complete/';
        local idx=60;
        local img_path=out_dir..idx..'.png';
        local bbox_path=out_dir..idx..'_bbox.npy';
        local crop_path=out_dir..idx..'_crop.npy';
        local img=image.load(img_path);
        local crop_box=npy4th.loadnpy(crop_path);
        print (img_path,crop_box);
        local bbox=npy4th.loadnpy(bbox_path);
        isValid=self:isValidNegativeNew(crop_box,bbox,128,32,{0.5,2});
        -- print ('final',isValid);
    end

end

return data