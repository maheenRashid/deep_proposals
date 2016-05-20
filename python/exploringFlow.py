import processScoreResults as sr;
import visualize;
import scipy.misc;
import numpy as np;

import os;
import cPickle as pickle;
import math;
import multiprocessing;
import imp
import sys
sys.path.append('/home/maheenrashid/Downloads/debugging_jacob/python/')
import processOutput as po;
import util;

def getOverlappingWindowCoord(bbox,im_size,window,step_size,thresh=0):
    bbox=[int(val) for val in bbox];
    min_coord=[bbox[idx]-window for idx in range(2)]
    max_coord=[int(min(bbox[idx]+window,im_size[idx%2])) for idx in range(2,4)];
    start_coord=[int(max(0,math.floor(min_coord_curr/float(step_size)))) for min_coord_curr in min_coord];
    bboxes=[];
    ious=[];
    for r_curr in range(start_coord[0],max_coord[0]-window,step_size):
        for c_curr in range(start_coord[1],max_coord[1]-window,step_size):
            bbox_crop=[r_curr,c_curr,r_curr+window,c_curr+window];
            iou=sr.getBBoxIOU(bbox,bbox_crop);
            if iou>thresh:
                bboxes.append(bbox_crop);
                ious.append(iou);
    return bboxes,ious

def saveOverlappingBBoxViz(img_path,boxes_overlap,box_rel,out_file_curr):
    colors=[(255,255,255)]*len(boxes_overlap);
    colors.append((255,0,0))
    boxes_overlap.append(box_rel);
    visualize.plotBBox(img_path,boxes_overlap,out_file_curr,colors)


def saveBBoxCrops(im,boxes_overlap,out_files):
    assert len(boxes_overlap)==len(out_files);
    for idx,box_curr in enumerate(boxes_overlap):
        if len(im.shape)>2:
            im_curr=im[box_curr[0]:box_curr[2],box_curr[1]:box_curr[3],:]; 
        else:
            im_curr=im[box_curr[0]:box_curr[2],box_curr[1]:box_curr[3]];
        # out_file_curr=out_file_pre+'_'+str(idx)+'.png';
        out_file_curr=out_files[idx];
        scipy.misc.imsave(out_file_curr,im_curr);

def script_saveBBoxCrops((img_path,box_rel,scale,thresh,window,step_size,out_file_pre,idx_im)):
    print idx_im
    im=scipy.misc.imread(img_path);
    im_org_shape=im.shape;

    im=scipy.misc.imresize(im,float(scale));
    

    # print 'box_rel bef',box_rel
    dims=[box_rel[idx+2]-box_rel[idx] for idx in range(2)];
    max_dim=max(dims);
    padding=(64/96.0)*max_dim;
    new_dim=max_dim+padding;
    padding_to_add=[(new_dim-dim_curr)/2 for dim_curr in dims];
    box_rel=[max(box_rel[0]-padding_to_add[0],0),max(box_rel[1]-padding_to_add[1],0),
    min(box_rel[2]+padding_to_add[0],im_org_shape[0]),min(box_rel[3]+padding_to_add[1],im_org_shape[1])]
    # print 'dims,max_dims',dims,max_dim
    # print 'new_dim',new_dim
    # print 'padding to add',padding_to_add
    # print 'box_rel aft',box_rel
    box_rel=[b*scale for b in box_rel];
    

    # raw_input();
    max_dim=dims.index(max(dims));
    max_dim_size=max(dims);


    im_size=im.shape; 
    boxes_overlap,ious=getOverlappingWindowCoord(box_rel,im_size,window,step_size,thresh);
    out_files_crops=[out_file_pre+'_'+str(idx_curr)+'.png' for idx_curr in range(len(boxes_overlap))];
    saveBBoxCrops(im,boxes_overlap,out_files_crops)
    for box_idx,box_curr in enumerate(boxes_overlap):
        out_file_curr=out_file_pre+'_'+str(box_idx)+'_onImg.png';
        saveOverlappingBBoxViz(im,[box_curr],box_rel,out_file_curr);

    out_file_pickle=out_file_pre+'_record.p';
    pickle.dump([boxes_overlap,ious],open(out_file_pickle,'wb'));


def script_saveImCrops(path_to_npy,path_to_im,ext,lim,out_file,out_dir_scratch,window,step_size,thresh,scale_info,scale,scale_images,lim_cases):
    # path_to_npy='/disk2/mayExperiments/validation_anno/';
    # path_to_im='/disk2/ms_coco/val2014/';
    # ext='.jpg'
    # lim=100;
    # out_file=os.path.join(path_to_npy,'index_'+str(lim)+'.p');
    # out_dir_scratch='/disk2/mayExperiments/flow_resolution_scratch/im_viz';
    # window=160;
    # step_size=16;
    # thresh=0.30;
    # scale_info=pickle.load(open(out_file,'rb'));
    # scale='large';
    # scale_images=[1];
    # lim_cases=10;
    
    args=[];

    for scale_image in scale_images:
        out_dir_curr_scale=os.path.join(out_dir_scratch,scale+'_'+str(scale_image));
        util.mkdir(out_dir_curr_scale);
        for idx_im,im_curr_info in enumerate(scale_info[scale][:lim_cases]):
            im_name=im_curr_info[0];

            out_dir_im=os.path.join(out_dir_curr_scale,im_name);
            util.mkdir(out_dir_im);

            bbox_idx=im_curr_info[2];

            img_path=os.path.join(path_to_im,im_name+ext);
            box_path=os.path.join(path_to_npy,im_name+'.npy');
            box_info=np.load(box_path);

            # im_name='COCO_val2014_000000000192';
            # for bbox_idx in range(box_info.shape[0]):
            box_rel=box_info[bbox_idx,:];
            box_rel=sr.convertBBoxFormatToStandard(box_rel)
            out_file_pre=os.path.join(out_dir_im,im_name+'_'+str(bbox_idx));        
            args.append((img_path,box_rel,scale_image,thresh,window,step_size,out_file_pre,idx_im));
            # else:
            #     box_rel=box_info[bbox_idx,:];
            #     box_rel=sr.convertBBoxFormatToStandard(box_rel)
            #     out_file_pre=os.path.join(out_dir_im,im_name+'_'+str(bbox_idx));        
            #     args.append((img_path,box_rel,scale_image,thresh,window,step_size,out_file_pre,idx_im));

        # for arg in args:
        #     script_saveBBoxCrops(arg);
        p=multiprocessing.Pool(multiprocessing.cpu_count())
        p.map(script_saveBBoxCrops,args)


def script_writeHTMLForOverlap(path_to_npy,path_to_im,ext,lim,out_file,out_dir_scratch,window,step_size,thresh,scale_info,scale_all, scale_images,lim_cases,gpu,model_file,clusters_file):
    img_dirs_all=[];
    for scale in scale_all:
        for scale_image in scale_images:
            dir_scale=os.path.join(out_dir_scratch,scale+'_'+str(scale_image));
            scale_info=pickle.load(open(out_file,'rb'));
            img_dirs=[os.path.join(dir_scale,im_curr_info[0]) for im_curr_info in scale_info[scale][:lim_cases]]
            img_dirs_all=img_dirs_all+img_dirs;

    # record_files_all=[];
    img_files_record={};

    for img_dir in img_dirs_all:
        record_files=[os.path.join(img_dir,file_curr) for file_curr in util.getFilesInFolder(img_dir,'.p')];
        for record_file in record_files:
            record=pickle.load(open(record_file,'rb'));
            if len(record[0])==0:
                continue;

            
            img_name=img_dir[img_dir.rindex('/')+1:];
            rel_files=[];
            # print len(record[0])
            # print record_file
            img_name_ac=record_file[record_file.rindex('/')+1:record_file.rindex('_')];
            for idx_curr in range(len(record[0])):
                rel_file_curr=[];
                rel_file_curr.append(os.path.join(img_dir,img_name_ac+'_'+str(idx_curr)+'_onImg.png'));
                rel_file_curr.append(os.path.join(img_dir,img_name_ac+'_'+str(idx_curr)+'.png'));
                rel_file_curr.append(os.path.join(img_dir+'_pred_flo_viz',img_name_ac+'_'+str(idx_curr)+'.png'));
                rel_file_curr.append(record[1][idx_curr]);
                rel_files.append(rel_file_curr);
            if img_name_ac in img_files_record:
                img_files_record[img_name_ac].extend(rel_files);
            else:
                img_files_record[img_name_ac]=rel_files;

            print len(img_files_record);
            # print img_files_record[img_files_record.keys()[0]]; 



    out_file_html=os.path.join(out_dir_scratch,'visualize.html');
    img_paths_html=[];
    captions_html=[];

    for img_name in img_files_record.keys():
        img_paths_row=[];
        captions_row=[];
        rec=img_files_record[img_name];
        rec_np=np.array(rec);
        print rec_np.shape
        sort_idx=np.argsort(rec_np[:,-1])[::-1];
        for idx_curr in sort_idx[::5]:
            img_paths_row.extend(rec[idx_curr][:-1]);
            captions_row.extend([str(rec[idx_curr][-1]),str(rec[idx_curr][-1]),str(rec[idx_curr][-1])]);
        img_paths_row=[util.getRelPath(path_curr) for path_curr in img_paths_row];

        img_paths_html.append(img_paths_row);
        captions_html.append(captions_row);

    visualize.writeHTML(out_file_html,img_paths_html,captions_html);

def script_doEverything(path_to_npy,path_to_im,ext,lim,out_file,out_dir_scratch,window,step_size,thresh,scale_info,scale_all, scale_images,lim_cases,gpu,model_file,clusters_file,train_val_file=None,overwrite=False):
    for scale in scale_all:
        # script_saveImCrops(path_to_npy,path_to_im,ext,lim,out_file,out_dir_scratch,window,step_size,thresh,scale_info,scale,scale_images,lim_cases)

        for scale_image in scale_images:
            dir_scale=os.path.join(out_dir_scratch,scale+'_'+str(scale_image));
            
            scale_info=pickle.load(open(out_file,'rb'));
            img_dirs=[os.path.join(dir_scale,im_curr_info[0]) for im_curr_info in scale_info[scale][:lim_cases]]
            
            for img_dir in img_dirs:
                img_paths=util.getFilesInFolder(img_dir,ext='.png');
                
                if len(img_paths)==0:
                    'CONTINUING'
                    continue;

                img_paths=[img_path for img_path in img_paths if not img_path.endswith('onImg.png')];
                out_dir_flo=img_dir+'_pred_flo';
                out_dir_flo_viz=img_dir+'_pred_flo_viz';
                util.mkdir(out_dir_flo);
                util.mkdir(out_dir_flo_viz);
                po.script_saveFlosAndViz(img_paths,out_dir_flo,out_dir_flo_viz,gpu,model_file,clusters_file,train_val_file=train_val_file,overwrite=overwrite)
                img_names=util.getFileNames(img_paths,ext=False);
                out_dir_flo=img_dir+'_pred_flo';
                out_dir_flo_viz=img_dir+'_pred_flo_viz';
                out_file_html=img_dir+'.html';

                
                img_paths_html=[];
                captions_all=[];
                for img_name in img_names:
                    row_curr=[];
                    row_curr.append(util.getRelPath(os.path.join(img_dir,img_name+'_onImg.png')));
                    row_curr.append(util.getRelPath(os.path.join(img_dir,img_name+'.png')));
                    row_curr.append(util.getRelPath(os.path.join(out_dir_flo_viz,img_name+'.png')));
                    captions=['','','']
                    img_paths_html.append(row_curr);
                    captions_all.append(captions);

                visualize.writeHTML(out_file_html,img_paths_html,captions_all);



def script_testOnYoutube():
    val_file='/disk2/mayExperiments/finetuning_youtube_hmdb_llr/val_eq.txt'
    out_dir='/disk2/mayExperiments/eval_ucf_finetune';
    clusters_file='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction_test/examples/opticalflow/clusters.mat';
    gpu=0;

    util.mkdir(out_dir);
    # model_file='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction_test/examples/opticalflow/final.caffemodel';
    # out_dir_model=os.path.join(out_dir,'original_model');

    model_file='/disk2/mayExperiments/ft_youtube_hmdb_ucfClusters/OptFlow_youtube_hmdb__iter_55000.caffemodel';
    out_dir_model=os.path.join(out_dir,'ft_ucf_model');

    util.mkdir(out_dir_model);
    out_dir_flo=os.path.join(out_dir_model,'flo');
    out_dir_flo_viz=os.path.join(out_dir_model,'flo_viz');
    util.mkdir(out_dir_flo);util.mkdir(out_dir_flo_viz)

    num_to_pick=20;

    img_paths=util.readLinesFromFile(val_file);
    img_paths=[img_path[:img_path.index(' ')] for img_path in img_paths];
    class_names=[file_curr[:file_curr.index('_')] for file_curr in util.getFileNames(img_paths)];
    classes=list(set(class_names));
    class_names=np.array(class_names);
    
    img_paths_test=[];
    for class_curr in classes:
        idx_rel=np.where(class_names==class_curr)[0];
        idx_rel=idx_rel[:num_to_pick];
        img_paths_test.extend([img_paths[idx_curr] for idx_curr in idx_rel]);

    # po.script_saveFlosAndViz(img_paths_test,out_dir_flo,out_dir_flo_viz,gpu,model_file,clusters_file);

    out_file_html=os.path.join(out_dir,'model_comparison.html');
    out_dirs_flo_viz=[os.path.join(out_dir,'original_model','flo_viz'),os.path.join(out_dir,'ft_ucf_model','flo_viz')];
    out_dirs_flo_viz_captions=['original_model','ft_ucf_model'];
    img_paths_html=[];
    captions_html=[];
    img_names=util.getFileNames(img_paths_test,ext=False);
    for img_path_test,img_name in zip(img_paths_test,img_names):
        row_curr=[];
        row_curr.append(util.getRelPath(img_path_test));
        for out_dir_curr in out_dirs_flo_viz:
            file_curr=os.path.join(out_dir_curr,img_name+'.png');
            row_curr.append(util.getRelPath(file_curr));
        captions_curr=[img_name]+out_dirs_flo_viz_captions;
        img_paths_html.append(row_curr)
        captions_html.append(captions_curr);
    visualize.writeHTML(out_file_html,img_paths_html,captions_html);


def saveFlosFromValFile(out_dir_model,val_file,num_to_pick,model_file,clusters_file,gpu,train_val_file=None,overwrite=False):

    
    out_dir_flo=os.path.join(out_dir_model,'flo');
    out_dir_flo_viz=os.path.join(out_dir_model,'flo_viz');
    util.mkdir(out_dir_flo);util.mkdir(out_dir_flo_viz)



    img_paths=util.readLinesFromFile(val_file);
    img_paths=[img_path[:img_path.index(' ')] for img_path in img_paths];
    class_names=[file_curr[:file_curr.index('_')] for file_curr in util.getFileNames(img_paths)];
    classes=list(set(class_names));
    class_names=np.array(class_names);
    
    img_paths_test=[];
    for class_curr in classes:
        idx_rel=np.where(class_names==class_curr)[0];
        idx_rel=idx_rel[:num_to_pick];
        img_paths_test.extend([img_paths[idx_curr] for idx_curr in idx_rel]);

    po.script_saveFlosAndViz(img_paths_test,out_dir_flo,out_dir_flo_viz,gpu,model_file,clusters_file,train_val_file=train_val_file,overwrite=overwrite);
    return img_paths_test;


def main():
    val_file='/disk2/mayExperiments/finetuning_youtube_hmdb_llr/val_eq.txt'
    out_dir='/disk2/mayExperiments/eval_nC_zS_youtube';
    model_file='/disk3/maheen_data/ft_youtube_40_images_cluster_suppress_yjConfig/opt_noFix_conv1_conv2_conv3_conv4_conv5_llr__iter_50000.caffemodel'
    clusters_file='/disk3/maheen_data/youtube_train_40/clusters_100000.mat';
    train_val_file='/disk3/maheen_data/ft_youtube_40_images_cluster_suppress_yjConfig/train_val_conv1_conv2_conv3_conv4_conv5.prototxt';
    util.mkdir(out_dir);
    out_dir_model=os.path.join(out_dir,'ft_youtube_model');
    num_to_pick=100;
    util.mkdir(out_dir_model);
    gpu=0;


    model_file_orig='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction_test/examples/opticalflow/final.caffemodel'
    clusters_file_orig='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction_test/examples/opticalflow/clusters.mat';
    out_dir_model_orig=os.path.join(out_dir,'ft_original_model');
    util.mkdir(out_dir_model_orig);
    
    img_paths_test=saveFlosFromValFile(out_dir_model_orig,val_file,num_to_pick,model_file_orig,clusters_file_orig,gpu,train_val_file=None,overwrite=True);

    out_file_html=os.path.join(out_dir,'model_comparison.html');
    out_dirs_flo_viz=[os.path.join(out_dir_model_orig,'flo_viz'),os.path.join(out_dir_model,'flo_viz')];
    out_dirs_flo_viz_captions=['original_model','ft_youtube_model'];
    img_paths_html=[];
    captions_html=[];
    img_names=util.getFileNames(img_paths_test,ext=False);
    for img_path_test,img_name in zip(img_paths_test,img_names):
        row_curr=[];
        row_curr.append(util.getRelPath(img_path_test));
        for out_dir_curr in out_dirs_flo_viz:
            file_curr=os.path.join(out_dir_curr,img_name+'.png');
            row_curr.append(util.getRelPath(file_curr));
        captions_curr=[img_name]+out_dirs_flo_viz_captions;
        img_paths_html.append(row_curr)
        captions_html.append(captions_curr);
    visualize.writeHTML(out_file_html,img_paths_html,captions_html);

    return
    path_to_npy='/disk2/mayExperiments/validation_anno/';
    path_to_im='/disk2/ms_coco/val2014/';
    ext='.jpg'
    lim=100;
    out_file=os.path.join(path_to_npy,'index_'+str(lim)+'.p');
    out_dir_scratch='/disk2/mayExperiments/flow_resolution_scratch/im_viz';
    out_dir_scratch='/disk2/mayExperiments/flow_resolution_scratch/im_viz_new_3';
    
    # out_dir_scratch='/disk2/mayExperiments/flow_resolution_scratch/im_viz_padding';
    # model_file='/disk2/mayExperiments/finetuning_youtube_hmdb_llr/OptFlow_youtube_hmdb_iter_50000.caffemodel';

    out_dir_scratch='/disk2/mayExperiments/flow_resolution_scratch/im_viz_padding_orig_model';
    model_file='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction_test/examples/opticalflow/final.caffemodel'
    

    # out_dir_scratch='/disk2/mayExperiments/flow_resolution_scratch/im_viz_padding_ft_ucf_model';
    # out_dir_scratch='/disk2/mayExperiments/flow_resolution_scratch/im_viz_padding_ft_ucf_model';
    # util.mkdir(out_dir_scratch);
    # model_file='/disk2/mayExperiments/ft_youtube_hmdb_ucfClusters/OptFlow_youtube_hmdb__iter_55000.caffemodel';

    out_dir_scratch='/disk2/mayExperiments/flow_resolution_scratch/im_viz_padding_ft_ucf_model_imagenet';
    util.mkdir(out_dir_scratch);
    model_file='/disk3/maheen_data/ft_imagenet_ucf/OptFlow_imagenet_hlr__iter_22000.caffemodel'
    clusters_file='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction_test/examples/opticalflow/clusters.mat';


    out_dir_scratch='/disk2/mayExperiments/flow_resolution_scratch/im_viz_padding_ft_nC_sZ_youtube';
    util.mkdir(out_dir_scratch);
    model_file='/disk3/maheen_data/ft_youtube_40_images_cluster_suppress_yjConfig/opt_noFix_conv1_conv2_conv3_conv4_conv5_llr__iter_50000.caffemodel'
    clusters_file='/disk3/maheen_data/youtube_train_40/clusters_100000.mat';
    train_val_file='/disk3/maheen_data/ft_youtube_40_images_cluster_suppress_yjConfig/train_val_conv1_conv2_conv3_conv4_conv5.prototxt';
    overwrite=True;
    window=160;
    step_size=16;
    thresh=0.50;
    scale_info=pickle.load(open(out_file,'rb'));
    # scale_all=['small','large','medium'];
    scale_all=['large','medium'];
    scale_pow=np.arange(-2,1.5,0.5);
    print scale_pow;
    scale_images=[2**pow_curr for pow_curr in scale_pow];
    # scale_images=scale_images[];
    print scale_images;

    # return
    # scale_images=[0.5,1,2];
    lim_cases=20;
    gpu=0;
    
    print po.NUM_THREADS
    po.NUM_THREADS=multiprocessing.cpu_count();
    print po.NUM_THREADS

    # return
    
    

    script_doEverything(path_to_npy,path_to_im,ext,lim,out_file,out_dir_scratch,window,step_size,thresh,scale_info,scale_all, scale_images,lim_cases,gpu,model_file,clusters_file,train_val_file=train_val_file,overwrite=overwrite);

    script_writeHTMLForOverlap(path_to_npy,path_to_im,ext,lim,out_file,out_dir_scratch,window,step_size,thresh,scale_info,scale_all, scale_images,lim_cases,gpu,model_file,clusters_file)


            # break
        # break;




            


    # return

    





        

if __name__=='__main__':
    main();