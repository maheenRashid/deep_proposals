	

    
    # path_to_anno='temp';
    # train_anno=os.path.join(path_to_anno,'small.json');
    # anno=json.load(open(train_anno,'rb'))
    # # ['annotations']
    # # anno=anno[:10000];
    # # json.dump(anno,open('temp/small.json','wb'));
    # # return
    # # ['annotations'];
    # # id_no=curr_anno['image_id'];
    # # seg=curr_anno['segmentation'];
    # #iscrowd
    # ids_all=[];
    # for curr_anno in anno:
    #     id_no=curr_anno['image_id'];
    #     ids_all.append(id_no);

    # ids_all=np.array(ids_all);
    # print ids_all.shape
    # ids_uni=np.unique(ids_all);
    # print ids_uni.shape

    # for id_uni in ids_uni:
    #     idx_rel=np.where(ids_all==id_uni)[0];
        
    #     bboxes=[];
    #     for idx_curr in idx_rel:
    #         print id_uni,idx_curr
    #         anno_rel=anno[idx_curr];
    #         assert anno_rel['image_id']==id_uni;
    #         if anno_rel['iscrowd']==0:
    #             bboxes.append(anno_rel['bbox']);

        
    #     raw_input();


    # return
    im_path='temp/im.jpg';
    seg_path='temp/seg.p';
    out_file_im='temp/im_resize.jpg';
    out_file_mask='temp/bb.npy';
    # out_file_im='temp/im_crop.png';
    # out_file_mask='temp/im_crop_mask.png';

    max_dim_size=128;
    total_size=240;
    seg=pickle.load(open(seg_path,'rb'));
    seg=[seg]
    # # saveMaskAndCrop((im_path,seg,max_dim_size,total_size,out_file_im,out_file_mask));
    #



    saveFullImAndBBCanonical((im_path,seg,max_dim_size,total_size,out_file_im,out_file_mask))
    im_org=Image.open(im_path);
    print (im_org.size)
    im=Image.open(out_file_im);
    print (im.size)
    crop_bb=np.load(out_file_mask);
    print crop_bb
    showIm(im);
    ImageDraw.Draw(im).rectangle(list(crop_bb));
    showIm(im);
    
    # print im.size,crop_bb;
    # showIm(mask);
    # print im_np.shape,mask.shape;

    raw_input();
    return
    out_dir ='/disk2/februaryExperiments/deep_proposals' 
    dir_content_file=os.path.join(out_dir,'contents_positives.txt');
    pre_dir=os.path.join(out_dir,'positives');
    img_dir=os.path.join(out_dir,'positives');
    postfix='.png';
    ignore_amount=-1;
    out_file_text=os.path.join(out_dir,'positive_data.txt');
    writeTrainingDataFiles(dir_content_file,pre_dir,img_dir,out_file_text,ignore_amount,postfix)




    return
    

	return



	
    seg=pickle.load(open('temp/seg_1.p','rb'));
    # im=scipy.misc.imread('temp/im_1.jpg');
    max_dim_size=128;
    total_size=240;
    im_path='temp/im_1.jpg';
    # se
    out_file_mask='temp/mask_1.png';
    out_file_im='temp/im_crop_1.png';
    
    seg_new=saveMaskAndCrop((im_path,seg,max_dim_size,total_size,out_file_im,out_file_mask))

    # im=scipy.misc.imread(out_file_im);
    # im=Image.fromarray(im);
    im=Image.open(out_file_im);
    showIm(im);
    im=Image.open(out_file_mask);
    showIm(im);

    raw_input();




    return


	return
#read numpy
	prop_data=np.load(train_prop);
	# {image_id, x, y, w, h, cat_id, score}

	im_path=addLeadingZeros(int(prop_data[0][0]),os.path.join(path_to_im,im_pre),'.jpg');
	# im=scipy.misc.imread(open(im_path,'rb'));
	# .Image.fromarray
	im=Image.open(im_path);
	for r in prop_data:
		rect=[r[1],r[2],r[3]+r[1],r[4]+r[2]]
		print rect
		ImageDraw.Draw(im).rectangle(rect,outline=(255,255,255));
	im.save(os.path.join(out_dir,'check_prop.png'));




	return

	# save a subset
	# prop_data=np.load(train_prop);
	# print prop_data[:10];
	# idx_to_keep=prop_data[:,0]==id_no;
	# rows_to_keep=prop_data[idx_to_keep,:];
	# np.save(out_file,rows_to_keep);
	




	return


	#get one input
	curr_anno=data['annotations'][0];
	id_no=curr_anno['image_id'];
	seg=curr_anno['segmentation'];
	
	file_pre=os.path.join(path_to_im,im_pre);
	im_path=addLeadingZeros(id_no,im_pre=file_pre,im_post='.jpg');
	
	#read image
	im=scipy.misc.imread(im_path);
	print im.shape;

	#make mask
	mask=getMask(seg[0],im.shape,3);
	
	#save mask and gt image
	print mask.shape,np.min(mask),np.max(mask);
	masked_im=im*mask;
	print masked_im.shape,np.min(masked_im),np.max(masked_im);
	print masked_im[:10,:10,0]
	print mask[:10,:10,0]
	print im[:10,:10,0]
	#makes sense?
	scipy.misc.imsave(open(os.path.join(out_dir,'mask.png'),'wb'),mask);
	scipy.misc.imsave(open(os.path.join(out_dir,'im.png'),'wb'),im);
	scipy.misc.imsave(open(os.path.join(out_dir,'im_mask.png'),'wb'),masked_im);


	print 'hello';
