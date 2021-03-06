import sys;
sys.path.append('/home/maheenrashid/Downloads/debugging_jacob/python/')
import inputProcessing as ip;
import processOutput as po;

import processScoreResults as psr;
import data_preprocessing as dp;
import numpy as np;
import json;
import cPickle as pickle;
import os;
import util;
import visualize;
from collections import namedtuple
import multiprocessing;
import shutil;
import glob;
import time;

def createParams(type_Experiment):
    if type_Experiment == 'getResultsForCatSpecific':
        list_params=['file_anno',
                    'category_id',
                    'input_im_pre',
                    'im_post',
                    'mask_post',
                    'num_to_pick',
                    'append_idx',
                    'out_dir_test',
                    'out_dir_im',
                    'out_dir_bbox',
                    'scale_idx_range',
                    'out_file_sh_small',
                    'out_file_sh_big',
                    'gpu_small',
                    'gpu_big',
                    'range_small',
                    'range_big',
                    'path_to_test_file_small',
                    'path_to_test_file_big',
                    'model',
                    'limit']
        params = namedtuple('Params_getResultsForCatSpecific',list_params);
    elif type_Experiment == 'saveAllTopPredViz':
        list_params=['dir_im',
                    'dir_gt_boxes',
                    'dir_mat_overlap',
                    'dir_overlap_viz',
                    'num_to_pick',
                    'pred_color',
                    'gt_color',
                    'num_threads',
                    'overwrite'];
        params = namedtuple('Params_saveAllTopPredViz',list_params);
    elif type_Experiment == 'saveAllWithGTOverlap':
        list_params=['dir_im',
                    'dir_gt_boxes',
                    'dir_mat_overlap',
                    'dir_overlap_viz',
                    'overlap_thresh',
                    'pred_color',
                    'gt_color',
                    'num_threads',
                    'overwrite'];
        params = namedtuple('Params_saveAllWithGTOverlap',list_params);
    elif type_Experiment == 'saveMatOverlapsVal':
        list_params=['dir_gt_boxes',
            'dir_canon_im',
            'dir_meta_test',
            'dir_result_check',
            'power_scale_range',
            'power_step_size',
            'top',
            'stride',
            'w',
            'out_dir_overlap',
            'overwrite'];
        params = namedtuple('Params_saveMatOverlapsVal',list_params);
    elif type_Experiment == 'saveMetaMatOverlapInfo':
        list_params=['out_dir_overlap',
                    'out_file',
                    'thresh_overlap']
        params = namedtuple('Params_saveMetaMatOverlapInfo',list_params);
    elif type_Experiment == 'getDataForTestingWithFlo':
        list_params = ['out_dir_test',
                        'out_dir_test_old',
                        'out_dir_im',
                        'out_dir_bbox',
                        'scale_idx',
                        'out_dir_im_withPadding',
                        'out_file_jTest_text',
                        'out_dir_flo',
                        'out_dir_flo_im',
                        'caffe_model_file',
                        'clusters_file',
                        'gpu',
                        'out_file_test',
                        'stride',
                        'w',
                        'mode',
                        'num_threads',
                        'continue_count']
        params = namedtuple('Params_getDataForTestingWithFlo',list_params);
    elif type_Experiment=='saveWithFloResultsInCompatibleFormat':
        list_params = ['dir_npy',
                        'dir_out',
                        'test_data_file',
                        'batchsize',
                        'num_threads'];
        params = namedtuple('Params_saveWithFloResultsInCompatibleFormat',list_params);
    else:
        params=None;

    return params;

def getSubsetByCategory(anno,category_id,im_pre,im_post='.png',mask_post='_mask.png',append_idx=True):
    if type(anno)==type('str'):
        anno=json.load(open(anno,'rb'))['annotations']

    tuples_by_class=[];

    for idx_curr_anno,curr_anno in enumerate(anno):
        if curr_anno['iscrowd']!=0:
            continue
        
        if curr_anno['category_id']!=category_id:
            continue;

        id_no=curr_anno['image_id'];
        im_path=dp.addLeadingZeros(id_no,im_pre);

        if append_idx:
            out_file=im_path+'_'+str(idx_curr_anno);
        else:
            out_file=im_path;

        out_file_im=out_file+im_post;
        if mask_post is not None:
            out_file_mask=out_file+mask_post;
            tuples_by_class.append((out_file_im,out_file_mask));
        else:
            tuples_by_class.append(out_file_im);

    return tuples_by_class  

def writeTextFilesForTesting(out_dir_meta,scale_idx_range,ext='.jpg',text_file_post=''):
    out_files_text=[]
    for scale_idx in scale_idx_range:
        dir_curr=os.path.join(out_dir_meta,str(scale_idx));
        ims=util.getFilesInFolder(dir_curr,ext=ext);
        file_list=[];
        for im_curr in util.getFileNames(ims,ext=False):
            file_curr=os.path.join(dir_curr,im_curr+ext)
            # print file_curr
            assert os.path.exists(file_curr)
            file_list.append(file_curr);
        # print len(file_list);
        out_file_text=dir_curr+text_file_post+'.txt';
        out_files_text.append(out_file_text);
        util.writeFile(out_file_text,file_list);
    return out_files_text;

def saveCatSpecificBboxFiles(anno,out_dir,im_pre,cat_id,im_ids):
    if type(anno)==type('str'):
        anno=json.load(open(anno,'rb'))['annotations']

    bbox_dict={};

    for idx in range(len(anno)):
        im_id=anno[idx]['image_id'];
        if im_id in im_ids:
            if anno[idx]['iscrowd']==1:
                continue;
            if anno[idx]['category_id']!=cat_id:
                continue;

            bbox=anno[idx]['bbox'];
            
            im_path=dp.addLeadingZeros(im_id,os.path.join(out_dir,im_pre),'.npy');
            if im_path in bbox_dict:
                bbox_dict[im_path].append(bbox);
            else:
                bbox_dict[im_path]=[bbox];

    for im_path in bbox_dict.keys():
        bbox_curr=bbox_dict[im_path];
        bbox_curr=np.array(bbox_curr);
        print im_path
        # raw_input();
        np.save(im_path,bbox_curr);

def script_getResultsForCatSpecific(params):
    file_anno = params.file_anno
    category_id = params.category_id
    input_im_pre = params.input_im_pre
    im_post = params.im_post
    mask_post = params.mask_post
    num_to_pick = params.num_to_pick
    append_idx = params.append_idx
    out_dir_test = params.out_dir_test
    out_dir_im = params.out_dir_im
    out_dir_bbox = params.out_dir_bbox
    scale_idx_range = params.scale_idx_range
    out_file_sh_small = params.out_file_sh_small
    out_file_sh_big = params.out_file_sh_big
    gpu_small = params.gpu_small
    gpu_big = params.gpu_big
    range_small = params.range_small
    range_big = params.range_big
    path_to_test_file_small = params.path_to_test_file_small
    path_to_test_file_big = params.path_to_test_file_big
    model = params.model
    limit = params.limit

    util.mkdir(out_dir_test);
    util.mkdir(out_dir_im);
    util.mkdir(out_dir_bbox);



    ims=getSubsetByCategory(file_anno,category_id,input_im_pre,im_post,mask_post,append_idx);
    ims=list(set(ims));
    ims.sort();
    ims=ims[:num_to_pick];

    for im in ims:
        assert os.path.exists(im);

    ip.rescaleImAndSaveMeta(ims,out_dir_im);

    out_files_text=writeTextFilesForTesting(out_dir_im,scale_idx_range);
    
    out_files_text_small=[os.path.join(out_dir_im,str(num)+'.txt') for num in range_small];
    out_dirs_small=[os.path.join(out_dir_bbox,str(num)) for num in range_small];
    
    out_files_text_big=[os.path.join(out_dir_im,str(num)+'.txt') for num in range_big];
    out_dirs_big=[os.path.join(out_dir_bbox,str(num)) for num in range_big];
    
    
    psr.script_writeTestSH(out_files_text_small,out_dirs_small,out_file_sh_small,path_to_test_file_small,model,limit,gpu_small)
    psr.script_writeTestSH(out_files_text_big,out_dirs_big,out_file_sh_big,path_to_test_file_big,model,limit,gpu_big)
    print '___';
    print out_file_sh_big
    print out_file_sh_small

def saveTopPredViz((out_file,im_path,mat_file,gt_file_name,num_to_pick,gt_color,pred_color,idx)):
    print idx;
    gt_boxes=np.load(gt_file_name);
    gt_boxes=np.array([psr.convertBBoxFormatToStandard(gt_box) for gt_box in gt_boxes]);
    
    mats_dict=np.load(mat_file);
    pred_boxes_rel=mats_dict['pred_boxes'][:num_to_pick,:];

    pred_scores=mats_dict['pred_scores'];
    if len(pred_scores.shape)<2:
        pred_scores=np.expand_dims(pred_scores,axis=1);
    # print pred_scores.shape
    pred_scores=pred_scores[:num_to_pick,0];

    
    colors=[pred_color]*pred_boxes_rel.shape[0]+[gt_color]*gt_boxes.shape[0];
    
    bboxes=np.vstack((pred_boxes_rel,gt_boxes));
    gt_boxes_labels=['gt']*len(gt_boxes);
    labels=list(pred_scores)+gt_boxes_labels
    visualize.plotBBox(im_path,bboxes,out_file,colors=colors,labels=labels);

def saveWithGTOverlapViz((out_file,im_path,mat_file,gt_file_name,overlap_thresh,gt_color,pred_color,idx)):
    print idx;
    gt_boxes=np.load(gt_file_name);
    gt_boxes=np.array([psr.convertBBoxFormatToStandard(gt_box) for gt_box in gt_boxes]);
    
    mats_dict=np.load(mat_file);
    mat_overlap=mats_dict['mat_overlap']
    mat_overlap_max=np.max(mat_overlap,axis=1);
    pred_boxes_rel=mats_dict['pred_boxes'][mat_overlap_max>=overlap_thresh,:];

    
    colors=[pred_color]*pred_boxes_rel.shape[0]+[gt_color]*gt_boxes.shape[0];
    
    bboxes=np.vstack((pred_boxes_rel,gt_boxes));
    visualize.plotBBox(im_path,bboxes,out_file,colors=colors);    
    

def script_saveAllTopPredViz(params):
    dir_im = params.dir_im
    dir_gt_boxes = params.dir_gt_boxes
    dir_mat_overlap = params.dir_mat_overlap
    dir_overlap_viz = params.dir_overlap_viz
    num_to_pick = params.num_to_pick
    pred_color = params.pred_color
    gt_color = params.gt_color
    num_threads = params.num_threads
    overwrite = params.overwrite

    util.mkdir(dir_overlap_viz);
    
    mats_all=util.getFilesInFolder(dir_mat_overlap,'.npz');
    im_names=util.getFileNames(mats_all,ext=False);
    # im_names=im_names[:1];


    args=[];
    idx=0;
    for mat_curr,im_name in zip(mats_all,im_names):
        im_path=os.path.join(dir_im,im_name+'.jpg');
        out_file=os.path.join(dir_overlap_viz,im_name+'.png');
        gt_file_name=os.path.join(dir_gt_boxes,im_name+'.npy');
        if os.path.exists(out_file) and not overwrite:
            continue;
        args.append((out_file,im_path,mat_curr,gt_file_name,num_to_pick,gt_color,pred_color,idx));
        idx+=1;

    # print args;
    # for arg in args:
    #     print arg;
    #     saveTopPredViz(arg);
    #     raw_input();
    p=multiprocessing.Pool(num_threads);
    p.map(saveTopPredViz,args);
    print len(args);
    
    visualize.writeHTMLForFolder(dir_overlap_viz,ext='.png',height=500,width=500);


def script_saveAllWithGTOverlap(params):
    dir_im = params.dir_im
    dir_gt_boxes = params.dir_gt_boxes
    dir_mat_overlap = params.dir_mat_overlap
    dir_overlap_viz = params.dir_overlap_viz
    overlap_thresh = params.overlap_thresh
    pred_color = params.pred_color
    gt_color = params.gt_color
    num_threads = params.num_threads
    overwrite = params.overwrite

    util.mkdir(dir_overlap_viz);
    
    mats_all=util.getFilesInFolder(dir_mat_overlap,'.npz');
    im_names=util.getFileNames(mats_all,ext=False);
    
    args=[];
    idx=0;
    for mat_curr,im_name in zip(mats_all,im_names):
        im_path=os.path.join(dir_im,im_name+'.jpg');
        out_file=os.path.join(dir_overlap_viz,im_name+'.png');
        gt_file_name=os.path.join(dir_gt_boxes,im_name+'.npy');
        if os.path.exists(out_file) and not overwrite:
            continue;
        args.append((out_file,im_path,mat_curr,gt_file_name,overlap_thresh,gt_color,pred_color,idx));
        idx+=1;

    p=multiprocessing.Pool(num_threads);
    p.map(saveWithGTOverlapViz,args);
    print len(args);
    
    visualize.writeHTMLForFolder(dir_overlap_viz,ext='.png',height=500,width=500);


def script_saveMatOverlapsVal(params):
    
    dir_gt_boxes = params.dir_gt_boxes
    dir_canon_im = params.dir_canon_im
    dir_meta_test = params.dir_meta_test
    dir_result_check = params.dir_result_check
    power_scale_range = params.power_scale_range
    power_step_size = params.power_step_size
    top = params.top
    stride = params.stride
    w = params.w
    out_dir_overlap = params.out_dir_overlap
    overwrite=params.overwrite;

    util.mkdir(out_dir_overlap);

    power_range=np.arange(power_scale_range[0],power_scale_range[1]+power_step_size,power_step_size);
    scales=[2**val for val in power_range];

    im_names=util.getFileNames(util.getFilesInFolder(dir_result_check,ext='_box.npy'),ext=False);
    im_names=[im_name[:im_name.rindex('_')] for im_name in im_names];
    print len(im_names);

    error_files=[];
    im_names_to_keep=[];
    for im_name in im_names:

        gt_file=os.path.join(dir_gt_boxes,im_name+'.npy');
        # print dir_gt_boxes
        if not os.path.exists(gt_file):
            error_files.append(gt_file);
        else:
            im_names_to_keep.append(im_name);
    print len(error_files),len(im_names_to_keep);

    
    args=[];
    for idx_im_name,im_name in enumerate(im_names_to_keep):
        im_path_canonical=os.path.join(dir_canon_im,im_name+'.jpg');
        out_file=os.path.join(out_dir_overlap,im_name+'.npz');
            
        if os.path.exists(out_file) and not overwrite:
            continue;

        args.append((im_name,im_path_canonical,dir_meta_test,scales,dir_gt_boxes,out_file,stride,w,top,idx_im_name))
    # raw_input();
    # print len(args);
    # print args[0]
    p=multiprocessing.Pool(multiprocessing.cpu_count());
    p.map(psr.saveMatOverlap,args)


def script_saveMetaMatOverlapInfoWithIndex(params):

    out_dir_overlap = params.out_dir_overlap
    out_file = params.out_file
    thresh_overlap = params.thresh_overlap
    
    mats_overlap=util.getFilesInFolder(params.out_dir_overlap,ext='.npz');

    # mats_overlap=mats_overlap[:10];
    args=[];
    for idx_mat_overlap,mat_overlap in enumerate(mats_overlap):
        args.append((mat_overlap,thresh_overlap,idx_mat_overlap));

    p=multiprocessing.Pool(multiprocessing.cpu_count());
    results=p.map(psr.getIdxCorrectWithImageId,args);
    results=np.vstack(results);

    print results.shape
    np.save(out_file,results);


def script_saveMetaMatOverlapInfoMultiThresh(params):

    out_dir_overlap = params.out_dir_overlap
    out_file = params.out_file
    threshes_overlap = params.thresh_overlap
    if not hasattr(threshes_overlap,'__iter__'):    
        threshes_overlap=[threshes_overlap];
    mats_overlap=util.getFilesInFolder(params.out_dir_overlap,ext='.npz');

    # mats_overlap=mats_overlap[:10];
    results_all=[];
    for thresh_overlap in threshes_overlap:
        args=[];
        for idx_mat_overlap,mat_overlap in enumerate(mats_overlap):
            args.append((mat_overlap,thresh_overlap,idx_mat_overlap));

        p=multiprocessing.Pool(multiprocessing.cpu_count());
        results=p.map(psr.getIdxCorrectWithImageId,args);
        # WithImageId
        results=np.vstack(results);
        results_all.append(results);
        print results.shape

    # print results.shape
    # np.save(out_file,results);
    pickle.dump(results_all,open(out_file,'wb'))

def script_saveMetaMatOverlapInfo(params):

    out_dir_overlap = params.out_dir_overlap
    out_file = params.out_file
    thresh_overlap = params.thresh_overlap
    
    mats_overlap=util.getFilesInFolder(params.out_dir_overlap,ext='.npz');

    # mats_overlap=mats_overlap[:10];
    # results_al
    # for thresh_overlap in threshes_overlap:
    args=[];
    for idx_mat_overlap,mat_overlap in enumerate(mats_overlap):
        args.append((mat_overlap,thresh_overlap,idx_mat_overlap));

    p=multiprocessing.Pool(multiprocessing.cpu_count());
    results=p.map(psr.getIdxCorrect,args);
    results=np.vstack(results);


    print results.shape
    np.save(out_file,results);


# def script_saveRecallCurveInfoMulti(meta_file,out_file_pre):
#     meta_info=pickle.load(open(meta_file,'rb'));


def getCountsCorrectByIdx((mat_overlap,img_idx_curr)): 
    meta_info=mat_overlap[mat_overlap[:,2]==img_idx_curr,:];
    threshes_size=[32**2,96**2];

    rel_idx=[];
    rel_idx.append(meta_info[:,1]<threshes_size[0]);
    rel_idx.append(np.logical_and(meta_info[:,1]<=threshes_size[1],meta_info[:,1]>=threshes_size[0]));
    rel_idx.append(meta_info[:,1]>threshes_size[1]);
    rel_idx.append(meta_info[:,1]>=-1);
    assert np.sum(rel_idx[-1])==meta_info.shape[0]

    counts_correct=[[],[],[],[]]
    for idx_rel_idx_curr,rel_idx_curr in enumerate(rel_idx):
        if np.sum(rel_idx_curr)>0:
            for top_curr in range(1000):    
                overlaps=meta_info[rel_idx_curr,0];
                correct=np.sum(np.logical_and(overlaps>=0,overlaps<=top_curr));
                total=overlaps.shape[0];
                ratio=correct/float(total);
                counts_correct[idx_rel_idx_curr].append(ratio);


    # print [len(counts_correct[i]) for i in range(len(counts_correct))];
    # print counts_correct[2][:10]
    return counts_correct;
    # raw_input();


def script_saveRecallCurveByImageIdx(meta_file,out_file_plot):

    mats_overlap=pickle.load(open(meta_file,'rb'));
    
    counts_all=[];
    for mat_overlap in mats_overlap:
        img_idx=np.unique(mat_overlap[:,2]);
        print img_idx.shape;
        args=[]
        for idx_img_idx_curr,img_idx_curr in enumerate(img_idx):
            # print idx_img_idx_curr
            args.append((mat_overlap,img_idx_curr));

        p=multiprocessing.Pool(multiprocessing.cpu_count());
        counts_correct_all=p.map(getCountsCorrectByIdx,args);


        counts_comb=[[],[],[],[]];
        for r in range(4):
            for counts_correct in counts_correct_all:
                if len(counts_correct[r])>0:
                    counts_comb[r].append(counts_correct[r]);
        counts_comb=[np.array(curr) for curr in counts_comb];
        avgs=[np.mean(curr,axis=0) for curr in counts_comb];
        counts_all.append(avgs);


    counts_all=np.array(counts_all);
    counts_all=np.mean(counts_all,axis=0);
    counts_all=[list(counts_all[idx,:]) for idx in range(counts_all.shape[0])];
    to_write= [counts_all[-1][idx-1] for idx in [10,100,1000]];
    print to_write;
    with open(out_file_plot[:out_file_plot.rindex('.')]+'.txt','wb') as f:
        f.write(str(to_write)+'\n');

    xAndYs=[(range(1,1001),counts_curr) for counts_curr in counts_all];
    legend=['small','medium','large','total'];
    visualize.plotSimple(xAndYs,out_file_plot,title='Average Recall',xlabel='Number of Proposals',ylabel='Average Recall',legend_entries=legend,loc=0,logscale=True);
    print out_file_plot.replace('/disk3','vision3.cs.ucdavis.edu:1001');



def script_saveRecallCurve(meta_file,out_file_plot):

    meta_info_all=np.load(meta_file);
    
    threshes_size=[32**2,96**2];
    counts_all=[];
    for meta_info in meta_info_all:
        rel_idx=[];
        rel_idx.append(meta_info[:,1]<threshes_size[0]);
        rel_idx.append(np.logical_and(meta_info[:,1]<=threshes_size[1],meta_info[:,1]>=threshes_size[0]));
        rel_idx.append(meta_info[:,1]>threshes_size[1]);
        rel_idx.append(meta_info[:,1]>=-1);
        
        counts_correct=[[],[],[],[]]
        for top_curr in range(1000):

            for idx_rel_idx_curr,rel_idx_curr in enumerate(rel_idx):
                overlaps=meta_info[rel_idx_curr,0];
                correct=np.sum(np.logical_and(overlaps>=0,overlaps<=top_curr));
                total=overlaps.shape[0];
                ratio=correct/float(total);
                counts_correct[idx_rel_idx_curr].append(ratio);

                # print overlaps.shape,np.sum(rel_idx_curr);
        print np.mean(counts_correct[0])
        print len(counts_correct)
        counts_all.append(counts_correct);

    # print len(counts_all),len(counts_all[0]),[len(counts_all[0][i]) for i in range(4)];

    counts_all=np.array(counts_all);
    counts_all=np.mean(counts_all,axis=0);
    counts_all=[list(counts_all[idx,:]) for idx in range(counts_all.shape[0])];
    to_write= [counts_all[-1][idx-1] for idx in [10,100,1000]];
    print to_write;
    with open(out_file_plot[:out_file_plot.rindex('.')]+'.txt','wb') as f:
        f.write(str(to_write)+'\n');

    xAndYs=[(range(1,1001),counts_curr) for counts_curr in counts_all];
    legend=['small','medium','large','total'];
    visualize.plotSimple(xAndYs,out_file_plot,title='Average Recall',xlabel='Number of Proposals',ylabel='Average Recall',legend_entries=legend,loc=0,logscale=True);
    print out_file_plot.replace('/disk3','vision3.cs.ucdavis.edu:1001');


def script_getDataForTestingWithFlo(params):

    out_dir_test = params.out_dir_test
    out_dir_test_old = params.out_dir_test_old
    out_dir_im = params.out_dir_im
    out_dir_bbox = params.out_dir_bbox
    scale_idx = params.scale_idx
    out_dir_im_withPadding = params.out_dir_im_withPadding
    out_file_jTest_text = params.out_file_jTest_text
    out_dir_flo = params.out_dir_flo
    out_dir_flo_im = params.out_dir_flo_im
    caffe_model_file = params.caffe_model_file
    clusters_file = params.clusters_file
    gpu = params.gpu
    out_file_test = params.out_file_test
    stride = params.stride
    w = params.w
    mode = params.mode
    num_threads = params.num_threads
    continue_count = params.continue_count

    util.mkdir(out_dir_im_withPadding);
    util.mkdir(out_dir_flo);
    util.mkdir(out_dir_flo_im);
    
    dirs_in=[os.path.join(out_dir_im,str(scale_idx_curr)) for scale_idx_curr in scale_idx]
    dirs_mat=[os.path.join(out_dir_bbox,str(scale_idx_curr)) for scale_idx_curr in scale_idx]
    dirs_out=[];
    for scale_idx_curr in scale_idx:
        dir_curr=os.path.join(out_dir_im_withPadding,str(scale_idx_curr));
        util.mkdir(dir_curr);
        dirs_out.append(dir_curr);

    args=[];
    for dir_in,dir_out,dir_mat in zip(dirs_in,dirs_out,dirs_mat):
        im_files=util.getFilesInFolder(dir_in,ext='.jpg');
        im_names=util.getFileNames(im_files,ext=True);
        out_files=[os.path.join(dir_out,im_name) for im_name in im_names];
        mat_files=[os.path.join(dir_mat,im_name[:im_name.rindex('.')]+'.npy') for im_name in im_names];
        for im_file,out_file,mat_file in zip(im_files,out_files,mat_files):
            args.append((im_file,mat_file,out_file,stride,w,mode));

    if continue_count<=0:
        p=multiprocessing.Pool(num_threads);
        p.map(psr.savePaddedImFromScoreMat,args);

    
    args_slidingWindows=[];

    for idx,arg in enumerate(args):
        im_file=arg[2];
        out_dir_curr=im_file[:im_file.rindex('/')];
        args_slidingWindows.append((im_file,stride,w,out_dir_curr,idx));

    if continue_count<=1:
        p = multiprocessing.Pool(num_threads);
        out_files_all = p.map(dp.saveSlidingWindows,args_slidingWindows);

        out_files_all = [file_curr for file_list in out_files_all for file_curr in file_list];
        util.writeFile(out_file_jTest_text,out_files_all);

    img_paths=util.readLinesFromFile(out_file_jTest_text);
    
    file_rec={};
    for img_path in img_paths:
        containing_dir=img_path[:img_path.rindex('/')]
        containing_dir=containing_dir[containing_dir.rindex('/')+1:];
        out_dir_curr=os.path.join(out_dir_flo,containing_dir);
        if out_dir_curr in file_rec:
            file_rec[out_dir_curr].append(img_path);
        else:
            file_rec[out_dir_curr]=[img_path];
            util.mkdir(out_dir_curr);

    print file_rec.keys();
    for k in file_rec.keys():
        print k,len(file_rec[k]);

    # return

    batch_size_flo=5000;
    count_all=0;
    count_done=0;
    for out_dir_curr_all in file_rec.keys():
        img_paths_all = file_rec[out_dir_curr_all];
        
        
        batch_idx_range=util.getIdxRange(len(img_paths_all),batch_size_flo);
        for batch_no,batch_start in enumerate(batch_idx_range[:-1]):
            batch_end=batch_idx_range[batch_no+1];
            img_paths=img_paths_all[batch_start:batch_end];
            
            out_dir_curr=os.path.join(out_dir_curr_all,str(batch_no));
            util.mkdir(out_dir_curr);
            
            out_dir_flo_im_curr=os.path.join(out_dir_curr,'flo_im');
            util.mkdir(out_dir_flo_im_curr);
            
            img_paths=[img_path for im_name,img_path in zip(util.getFileNames(img_paths,ext=False),img_paths) if not os.path.exists(os.path.join(out_dir_flo_im_curr,im_name+'.png'))];

            if len(img_paths) :
                if continue_count<=2:
                    po.NUM_THREADS = num_threads;
                    po.script_saveFlos(img_paths,out_dir_curr,gpu,caffe_model_file,clusters_file,overwrite=True);
                
                flo_dir=os.path.join(out_dir_curr,'flo_files');
                match_info_file=os.path.join(out_dir_curr,'match_info.txt');
                
                # files=util.getFilesInFolder(flo_dir,'.flo');
                # assert len(files)==len(img_paths);

                if continue_count<=3:
                    dp.saveFlowImFromFloDir(flo_dir,match_info_file,out_dir_flo_im_curr)

                shutil.rmtree(flo_dir);
                shutil.rmtree(os.path.join(out_dir_curr,'results'));

                count_all+=1;
            else:
                count_done+=1;

            # print batch_no,count_all,count_done;
        print out_dir_curr_all,count_all,count_done;
    print out_dir_curr_all,count_all,count_done;
        #     break
        # break
    return
    lines_to_write=[];
    for out_dir_curr_all in file_rec.keys():

        img_paths_all = file_rec[out_dir_curr_all];
        
        batch_idx_range=util.getIdxRange(len(img_paths_all),batch_size_flo);
        
        im_files_to_write=[];
        flo_im_files_to_write=[];
    
        for batch_no,batch_start in enumerate(batch_idx_range[:-1]):
            if batch_no<=3:
                continue;

            batch_end=batch_idx_range[batch_no+1];
            img_paths=img_paths_all[batch_start:batch_end];
            out_dir_curr=os.path.join(out_dir_curr_all,str(batch_no));
            
            out_dir_flo_im_curr=os.path.join(out_dir_curr,'flo_im');

            flo_im_paths=[os.path.join(out_dir_flo_im_curr,im_name+'.png') for im_name in util.getFileNames(img_paths,ext=False)];

            for flo_im_curr in flo_im_paths:
                assert os.path.exists(flo_im_curr);

            im_files_to_write.extend(img_paths);
            flo_im_files_to_write.extend(flo_im_paths);

            
            
            lines_to_write_curr=[a+' '+b for a,b in zip(im_files_to_write,flo_im_files_to_write)]
            lines_to_write.extend(lines_to_write_curr);           
            print out_dir_curr_all[out_dir_curr_all.rindex('/')+1:],batch_no,len(batch_idx_range)-1,len(lines_to_write);
            
            if batch_no==8:
                break;

    util.writeFile(out_file_test,lines_to_write);

    return
    lines_to_write=[];
    for im_dir in file_rec.keys():
        im_files=file_rec[im_dir];
        im_dir_num=im_dir[im_dir.rindex('/')+1:];
        flo_im_path_meta=os.path.join(out_dir_flo,im_dir_num,'flo_im');
        file_names=util.getFileNames(im_files,ext=False);
        flo_files=[os.path.join(flo_im_path_meta,file_curr+'.png') for file_curr in file_names];
        
        for flo_file in flo_files:
            assert os.path.exists(flo_file);
        
        lines_to_write.extend(zip(im_files,flo_files));
    
    lines_to_write=[a+' '+b for a,b in lines_to_write];
    util.writeFile(out_file_test,lines_to_write);


def getImFilesCropped(im_name,im_files):
    im_filenames=util.getFileNames(im_files,ext=False);
    rel_files=[];
    row_col=[]
    for im_file,im_filename in zip(im_files,im_filenames):
        if im_filename.startswith(im_name) and len(im_filename)>len(im_name):
            rel_files.append(im_file);
            # im_filename=im_filename[:im_filename.rindex('.')];
            im_filename_split=im_filename.split('_');
            # print im_filename_split
            r=int(im_filename_split[-2]);
            c=int(im_filename_split[-1]);
            row_col.append([r,c,]);

    return rel_files,row_col;


def script_sanityCheckFloCrops(dir_npy,dir_im,scale_idx_range=range(7)):
    # scale_idx_range = range(7)
    dirs_npy=[os.path.join(dir_npy,str(scale_idx_curr)) for scale_idx_curr in scale_idx_range];
    dirs_im=[os.path.join(dir_im,str(scale_idx_curr)) for scale_idx_curr in scale_idx_range];

    for dir_npy,dir_im in zip(dirs_npy,dirs_im):
        list_npy=util.getFilesInFolder(dir_npy,'_box.npy');
        im_names=util.getFileNames(list_npy);
        im_names=[im_name[:im_name.rindex('_')] for im_name in im_names];
        
        im_files=util.getFilesInFolder(dir_im,'.jpg');
        
        for im_name in im_names:
            im_rel_filenames,row_col=getImFilesCropped(im_name,im_files);
            score_mat=np.load(os.path.join(dir_npy,im_name+'.npy'))[0][0];
            # print score_mat.shape;
            # print len(im_rel_filenames);
            assert score_mat.size==len(im_rel_filenames);
            # raw_input();


def script_saveWithFloResultsInCompatibleFormat(params):
    dir_out = params.dir_out;
    dir_npy = params.dir_npy;
    test_data_file = params.test_data_file;
    batchsize = params.batchsize;
    num_threads = params.num_threads;

    lines = util.readLinesFromFile(test_data_file);
    lines = [line[:line.index(' ')] for line in lines];
    folders=[im.split('/')[-2] for im in lines];
    ims=util.getFileNames(lines,ext=False);
    
    score_files=psr.getScoreFileToTestFileMapping(dir_npy,batchsize,len(ims))

    ims_pre=['_'.join(im.split('_')[:-2]) for im in ims];
    folder_im_str=[folder+'_'+im for folder,im in zip(folders,ims_pre)];
    folder_im_str=np.array(folder_im_str);

    uni_folder_im_str=np.unique(folder_im_str);
        
    args=[];
    for idx_file,folder_im_str_curr in enumerate(uni_folder_im_str):

        folder_im_str_curr=uni_folder_im_str[idx_file];

        folder_curr=folder_im_str_curr[:folder_im_str_curr.index('_')];
        im_pre=folder_im_str_curr[folder_im_str_curr.index('_')+1:];
        out_folder_curr=os.path.join(dir_out,folder_curr);
        util.mkdir(out_folder_curr);
        out_file_pre=os.path.join(out_folder_curr,im_pre);
        # print folder_im_str_curr,
        # print out_file_pre,batchsize,len(score_files),len(ims),folder_im_str,idx_file
        args.append((folder_im_str_curr,out_file_pre,batchsize,score_files,ims,folder_im_str,idx_file));
        # break;

    print len(args);

    p=multiprocessing.Pool(num_threads);
    p.map(psr.saveWithFloResultsInCompatibleFormatMP,args);

def script_runFullResultGenerationStandard(dir_gt_boxes,dir_canon_im,dir_meta_test,dir_meta_out,dir_im,overwrite):
    params_dict={};
    params_dict['dir_gt_boxes'] = dir_gt_boxes;    
    params_dict['dir_canon_im'] = dir_canon_im;
    params_dict['dir_meta_test'] = dir_meta_test;
    params_dict['dir_result_check'] = os.path.join(params_dict['dir_meta_test'],'4');
    params_dict['power_scale_range'] = (-2,1);
    params_dict['power_step_size'] = 0.5
    params_dict['top'] = 1000;
    params_dict['stride'] = 16;
    params_dict['w'] = 160;
    params_dict['out_dir_overlap'] = os.path.join(dir_meta_out,'mat_overlaps_'+str(params_dict['top']));
    params_dict['overwrite'] = overwrite;
    params=createParams('saveMatOverlapsVal');
    params=params(**params_dict);
    script_saveMatOverlapsVal(params);
    pickle.dump(params._asdict(),open(os.path.join(params.out_dir_overlap,'params.p'),'wb'));

    out_dir_overlap = params_dict['out_dir_overlap'];
    params_dict={};
    params_dict['out_dir_overlap'] = out_dir_overlap
    params_dict['out_file'] = os.path.join(dir_meta_out,'meta_info_multi.p');
    params_dict['thresh_overlap'] = np.arange(0.5,1,0.05);
    params=createParams('saveMetaMatOverlapInfo');
    params=params(**params_dict);
    script_saveMetaMatOverlapInfoMultiThresh(params)
    pickle.dump(params._asdict(),open(os.path.join(params.out_dir_overlap,'params_saveMetaInfo_multi.p'),'wb'));

    meta_file=os.path.join(dir_meta_out,'meta_info_multi.p');
    out_file_plot=meta_file[:meta_file.rindex('.')]+'.png';
    script_saveRecallCurve(meta_file,out_file_plot)

    params_dict={};
    params_dict['dir_im']=dir_im;
    params_dict['dir_gt_boxes']=dir_gt_boxes;    
    params_dict['dir_mat_overlap']=out_dir_overlap;
    params_dict['num_to_pick']=10;
    params_dict['dir_overlap_viz']=os.path.join(dir_meta_out,'top_10_viz');
    params_dict['pred_color']=(255,255,255)
    params_dict['gt_color']=(255,0,0);
    params_dict['num_threads']=multiprocessing.cpu_count();
    params_dict['overwrite']=overwrite;

    params=createParams('saveAllTopPredViz');
    params=params(**params_dict);
    script_saveAllTopPredViz(params);
    pickle.dump(params._asdict(),open(os.path.join(params.dir_overlap_viz,'params.p'),'wb'));

    params_dict={};
    params_dict['dir_im']=dir_im;
    params_dict['dir_gt_boxes']=dir_gt_boxes;    
    params_dict['dir_mat_overlap']=out_dir_overlap;
    params_dict['dir_overlap_viz']=os.path.join(dir_meta_out,'gt_overlap_50_viz');
    params_dict['overlap_thresh']=0.5;
    params_dict['pred_color']=(255,255,255)
    params_dict['gt_color']=(255,0,0);
    params_dict['num_threads']=multiprocessing.cpu_count();
    params_dict['overwrite']=overwrite;

    params=createParams('saveAllWithGTOverlap');
    params=params(**params_dict);
    script_saveAllWithGTOverlap(params);
    pickle.dump(params._asdict(),open(os.path.join(params.dir_overlap_viz,'params.p'),'wb'));


def saveFloFileList((im_name,out_file,all_folders)):
    files_im=[];
    for folder_curr in all_folders:
        folder_files=util.readLinesFromFile(folder_curr+'.txt');
        folder_file_names=util.getFileNames(folder_files);

        files_all=[file_curr for file_curr,file_name in zip(folder_files,folder_file_names) if file_name.startswith(im_name)]
        files_im.append(files_all);
    
    files_im=[file_curr for file_list in files_im for file_curr in file_list];
    files_im=np.array(files_im);
    
    np.save(out_file,files_im);

def saveFolderIndexFile():
    pass;
    # print len(all_folders);
    # print all_folders[0];
    # print folder_count;

    # out_file_commands=os.path.join(flo_dir_meta,'commands_to_get_list.sh');
    # commands_all=[];
    # for folder_curr in all_folders:
    #     command_curr = 'find '+folder_curr+'/flo_im/*.png >> '+folder_curr+'.txt';
    #     print command_curr;
    #     commands_all.append(command_curr);

    # util.writeFile(out_file_commands,commands_all);
    # print out_file_commands;

def writeTestFloFile(im_paths,im_folder_meta,out_file_test):
    lines_to_write=[];
    for im_path in im_paths:
        flo_files=np.load(im_path);
        flo_file_names=util.getFileNames(flo_files,ext=False);
        for flo_file,flo_file_name in zip(flo_files,flo_file_names):
            scale_curr=flo_file.rsplit('/',4);
            scale_curr=scale_curr[1];
            im_file=os.path.join(im_folder_meta,scale_curr,flo_file_name+'.jpg');
            line_curr=im_file+' '+flo_file;
            lines_to_write.append(line_curr);


    print len(lines_to_write);
    util.writeFile(out_file_test,lines_to_write);

    # print im_file;
    # print flo_file;
    # assert os.path.exists(im_file);
    # raw_input();

def main():
    dir_gt_boxes='/disk3/maheen_data/val_anno_human_only_300'
    dir_canon_im='/disk3/maheen_data/headC_160_withFlow_justHuman_retrain/im_rem/4';
    dir_meta_test='/disk3/maheen_data/headC_160_withFlow_human_xavier_unit_floStumpPretrained_fullTraining/results_res_30000_compiled'
    dir_meta_out='/disk3/maheen_data/headC_160_withFlow_human_xavier_unit_floStumpPretrained_fullTraining/accuracy'

    util.mkdir(dir_meta_out);
    overwrite=True;
    dir_im='/disk2/ms_coco/val2014'
    script_runFullResultGenerationStandard(dir_gt_boxes,dir_canon_im,dir_meta_test,dir_meta_out,dir_im,overwrite)


    return
    params_dict={};
    
    dir_meta='/disk3/maheen_data/headC_160_withFlow_human_xavier_unit_floStumpPretrained_fullTraining'
    # '/disk3/maheen_data/headC_160_withFlow_justHuman_retrain/test_train_data';

    dir_out = os.path.join(dir_meta,'results_res_30000_compiled');
    util.mkdir(dir_out);
    test_data_txt_dir = os.path.join(dir_meta,'test_files_split');
    test_data_res_dir = os.path.join(dir_meta,'results_res_30000');


    for batch_no in range(6):
        params_dict['dir_npy'] = os.path.join(test_data_res_dir,str(batch_no));
        params_dict['dir_out'] = dir_out
        params_dict['test_data_file'] = os.path.join(test_data_txt_dir,str(batch_no)+'.txt')
        params_dict['batchsize'] = 32;
        params_dict['num_threads'] = multiprocessing.cpu_count();

        
        params=createParams('saveWithFloResultsInCompatibleFormat');
        params=params(**params_dict);
        script_saveWithFloResultsInCompatibleFormat(params);
        pickle.dump(params._asdict(),open(os.path.join(params.dir_out,'params.p'),'wb'));

    return
    dir_test='/disk3/maheen_data/headC_160_withFlow_justHuman_retrain'
    flo_dir_meta=os.path.join(dir_test,'im_with_padding','flo');
    scale_idx_range=range(7);
    
    # out_dir_test_files=os.path.join(dir_test,'test_files_split');
    # util.mkdir(out_dir_test_files);
    im_folder_meta=os.path.join(dir_test,'im_with_padding');
    im_folder_full_im=os.path.join(dir_test,'im_rem')

    out_dir_im_lists = os.path.join(im_folder_meta,'flo_im_path_by_im');
    print out_dir_im_lists
    util.mkdir(out_dir_im_lists);

    batch_size=50;

    im_path_files=util.getFilesInFolder(out_dir_im_lists,'.npy');
    batch_idx = util.getIdxRange(len(im_path_files),batch_size);

    out_dir='/disk3/maheen_data/headC_160_withFlow_human_xavier_unit_floStumpPretrained_fullTraining';
    util.mkdir(out_dir);
    
    out_dir_test_files=os.path.join(out_dir,'test_files_split');
    util.mkdir(out_dir_test_files);
    
    out_dir_res=os.path.join(out_dir,'results_res_30000');
    util.mkdir(out_dir_res);

    for idx_batch_idx,start_idx in enumerate(batch_idx[:-1]):
        end_idx=batch_idx[idx_batch_idx+1];
        im_paths=im_path_files[start_idx:end_idx];
        out_file_test=os.path.join(out_dir_test_files,str(idx_batch_idx)+'.txt');
        # print out_file_test;
        # print len(im_paths);
        writeTestFloFile(im_paths,im_folder_meta,out_file_test);

        print 'th /home/maheenrashid/Downloads/deep_proposals/torch_new/test_command_line_flow.th -model /disk3/maheen_data/headC_160/withFlow_human_xavier_unit_floStumpPretrained_fullTraining_res/intermediate/model_all_30000.dat -testFile '+ out_file_test+' -outDir '+out_dir_res+'/'+str(idx_batch_idx);


    return

    dir_gt_boxes='/disk2/aprilExperiments/negatives_npy_onlyHuman';
    dir_canon_im='/disk3/maheen_data/headC_160_noFlow_justHuman_retrain/train_data_test/im/4';
    # dir_meta_test='/disk3/maheen_data/headC_160_withFlow_justHuman_retrain/results_res';
    # dir_meta_test='/disk3/maheen_data/headC_160_noFlow_justHuman_retrain/results';
    # dir_meta_out='/disk3/maheen_data/headC_160_withFlow_justHuman_retrain/res';

    # dir_meta_test='/disk3/maheen_data/headC_160_withFlow_justHuman_retrain/results_rem_compiled'
    # dir_meta_out='/disk3/maheen_data/headC_160_withFlow_justHuman_retrain/results_300_final'    

    # dir_meta_test='/disk3/maheen_data/headC_160_withFlow_justHuman_retrain/test_train_data/results_200000_compiled'
    # dir_meta_out='/disk3/maheen_data/headC_160_withFlow_justHuman_retrain/test_train_data/accuracy_200000'    
    dir_meta_test='/disk3/maheen_data/headC_160_noFlow_justHuman_retrain/train_data_test/results'
    dir_meta_out='/disk3/maheen_data/headC_160_noFlow_justHuman_retrain/train_data_test/accuracy'

    util.mkdir(dir_meta_out);
    overwrite=True;
    dir_im='/disk2/ms_coco/train2014'
    script_runFullResultGenerationStandard(dir_gt_boxes,dir_canon_im,dir_meta_test,dir_meta_out,dir_im,overwrite)


    return
    params_dict={};
    
    dir_meta='/disk3/maheen_data/headC_160_withFlow_justHuman_retrain/test_train_data';

    dir_out = os.path.join(dir_meta,'results_200000_compiled');
    util.mkdir(dir_out);
    test_data_txt_dir = os.path.join(dir_meta,'test_files_split');
    test_data_res_dir = os.path.join(dir_meta,'results_200000');


    for batch_no in range(6):
        params_dict['dir_npy'] = os.path.join(test_data_res_dir,str(batch_no));
        params_dict['dir_out'] = dir_out
        params_dict['test_data_file'] = os.path.join(test_data_txt_dir,str(batch_no)+'.txt')
        params_dict['batchsize'] = 32;
        params_dict['num_threads'] = multiprocessing.cpu_count();

        
        params=createParams('saveWithFloResultsInCompatibleFormat');
        params=params(**params_dict);
        script_saveWithFloResultsInCompatibleFormat(params);
        pickle.dump(params._asdict(),open(os.path.join(params.dir_out,'params.p'),'wb'));


    return

    dir_test='/disk3/maheen_data/headC_160_withFlow_justHuman_retrain'
    flo_dir_meta=os.path.join(dir_test,'im_with_padding','flo');
    scale_idx_range=range(7);
    
    # out_dir_test_files=os.path.join(dir_test,'test_files_split');
    # util.mkdir(out_dir_test_files);
    im_folder_meta=os.path.join(dir_test,'im_with_padding');
    im_folder_full_im=os.path.join(dir_test,'im_rem')

    out_dir_im_lists = os.path.join(im_folder_meta,'flo_im_path_by_im');
    print out_dir_im_lists
    util.mkdir(out_dir_im_lists);

    batch_size=50;

    im_path_files=util.getFilesInFolder(out_dir_im_lists,'.npy');
    batch_idx = util.getIdxRange(len(im_path_files),batch_size);

    out_dir='/disk3/maheen_data/headC_160_withFlow_human_xavier_unit_floStumpPretrained_fullTraining';
    util.mkdir(out_dir);

    out_dir_test_files=os.path.join(out_dir,'test_files_split');
    util.mkdir(out_dir_test_files);
    
    out_dir_res=os.path.join(out_dir,'results_res_30000');
    util.mkdir(out_dir_res);

    for idx_batch_idx,start_idx in enumerate(batch_idx[:-1]):
        end_idx=batch_idx[idx_batch_idx+1];
        im_paths=im_path_files[start_idx:end_idx];
        out_file_test=os.path.join(out_dir_test_files,str(idx_batch_idx)+'.txt');
        # print out_file_test;
        # print len(im_paths);
        writeTestFloFile(im_paths,im_folder_meta,out_file_test);

        print 'th /home/maheenrashid/Downloads/deep_proposals/torch_new/test_command_line_flow.th -model /disk3/maheen_data/headC_160/withFlow_human_xavier_unit_floStumpPretrained_fullTraining_res/intermediate/model_all_30000.dat -testFile '+ out_file_test+' -outDir '+out_dir_res+'/'+str(idx_batch_idx);

        # print 'th /home/maheenrashid/Downloads/deep_proposals/torch_new/test_command_line_flow.th -model /disk3/maheen_data/headC_160/withFlow_gaussian_human_softmax/final/model_all_final.dat -testFile '+ out_file_test+' -outDir /disk3/maheen_data/headC_160_withFlow_justHuman_retrain/test_train_data/results_100000/'+str(idx_batch_idx);
        # print /disk3/maheen_data/headC_160_withFlow_justHuman_retrain/test_files_split/0.txt



    return
    # dir_test='/disk3/maheen_data/headC_160_withFlow_justHuman_retrain/test_train_data'
    # flo_dir_meta=os.path.join(dir_test,'im_with_padding','flo');
    # scale_idx_range=range(7);
    
    # im_folder_meta=os.path.join(dir_test,'im_with_padding');
    # im_folder_full_im='/disk3/maheen_data/headC_160_noFlow_justHuman_retrain/train_data_test/im'
    

    # out_dir_im_lists=os.path.join(im_folder_meta,'flo_im_path_by_im');
    # print out_dir_im_lists
    # util.mkdir(out_dir_im_lists);

    # num_threads=multiprocessing.cpu_count();

    # # get all the folders with flo_im
    # folders_dict={};
    # for scale_curr in scale_idx_range:
    #     dir_curr=os.path.join(flo_dir_meta,str(scale_curr));

    #     folders_dict[dir_curr]=[];

    #     inner_folder_num=0;
    #     while True:
    #         dir_check=os.path.join(dir_curr,str(inner_folder_num));
    #         if os.path.exists(dir_check):
    #             folders_dict[dir_curr].append(dir_check);
    #         else:
    #             break;
    #         inner_folder_num+=1;

    # folder_count=0;
    # for folder in folders_dict.keys():
    #     print folder,len(folders_dict[folder]);
    #     folder_count+=len(folders_dict[folder]);

    # all_folders=[folder_curr for folder_list in folders_dict.values() for folder_curr in folder_list];

    # # print len(all_folders);
    # # print all_folders[0];
    # # print folder_count;

    # # out_file_commands=os.path.join(flo_dir_meta,'commands_to_get_list.sh');
    # # commands_all=[];
    # # for folder_curr in all_folders:
    # #     command_curr = 'find '+folder_curr+'/flo_im/*.png >> '+folder_curr+'.txt';
    # #     print command_curr;
    # #     commands_all.append(command_curr);

    # # util.writeFile(out_file_commands,commands_all);
    # # print out_file_commands;

    # # return

    # im_names = util.getFileNames(util.getFilesInFolder(os.path.join(im_folder_full_im,str(scale_idx_range[0])),ext='.jpg'),ext=False);

    # args=[];
    # for im_name in im_names:
    #     out_file=os.path.join(out_dir_im_lists,im_name+'.npy')
    #     if not os.path.exists(out_file):
    #         arg_curr=(im_name,out_file,all_folders);
    #         args.append(arg_curr);

    # p = multiprocessing.Pool(num_threads);
    # p.map(saveFloFileList,args);



    # return

    params_dict = {};
    params_dict['out_dir_test'] = '/disk3/maheen_data/headC_160_withFlow_justHuman_retrain/test_train_data';
    params_dict['out_dir_test_old'] = '/disk3/maheen_data/headC_160_noFlow_justHuman_retrain/train_data_test';
    params_dict['out_dir_im'] = os.path.join(params_dict['out_dir_test_old'],'im');
    params_dict['out_dir_bbox'] = os.path.join(params_dict['out_dir_test_old'],'results');
    params_dict['scale_idx'] = range(7);
    params_dict['out_dir_im_withPadding'] = os.path.join(params_dict['out_dir_test'],'im_with_padding');
    params_dict['out_file_jTest_text'] = os.path.join(params_dict['out_dir_im_withPadding'],'test.txt');
    params_dict['out_dir_flo'] = os.path.join(params_dict['out_dir_im_withPadding'],'flo');
    params_dict['out_dir_flo_im'] = os.path.join(params_dict['out_dir_im_withPadding'],'flo_im');
    params_dict['caffe_model_file'] = '/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction_test/examples/opticalflow/final.caffemodel';
    params_dict['clusters_file'] = '/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction_test/examples/opticalflow/clusters.mat';
    params_dict['gpu'] = 0;
    params_dict['out_file_test'] = os.path.join(params_dict['out_dir_test'],'test_data_flo.txt');
    params_dict['stride'] = 16;
    params_dict['w'] = 160;
    params_dict['mode'] = 'edge';
    params_dict['num_threads'] = multiprocessing.cpu_count();
    params_dict['continue_count'] = 2;

    params=createParams('getDataForTestingWithFlo');
    params=params(**params_dict);

    script_getDataForTestingWithFlo(params);
    pickle.dump(params._asdict(),open(os.path.join(params.out_dir_test,'params_getFloData.p'),'wb'));

    return

    out_dir_test='/disk3/maheen_data/headC_160_noFlow_justHuman_retrain/train_data_test';
    util.mkdir(out_dir_test);

    params_dict={};
    params_dict['file_anno'] = '/disk2/ms_coco/annotations/instances_train2014.json';
    params_dict['category_id'] = 1;
    params_dict['input_im_pre'] = '/disk2/ms_coco/train2014/COCO_train2014_';
    params_dict['im_post'] = '.jpg';
    params_dict['mask_post'] = None;
    params_dict['num_to_pick'] = 300;
    params_dict['append_idx'] = False;
    params_dict['out_dir_test'] = out_dir_test;
    params_dict['out_dir_im'] = os.path.join(params_dict['out_dir_test'],'im');
    params_dict['out_dir_bbox'] = os.path.join(params_dict['out_dir_test'],'results');
    params_dict['scale_idx_range'] = range(7);
    params_dict['out_file_sh_small'] = os.path.join(params_dict['out_dir_bbox'],'test_human_small.sh');
    params_dict['out_file_sh_big'] = os.path.join(params_dict['out_dir_bbox'],'test_human_big.sh');
    params_dict['gpu_small'] = 1;
    params_dict['gpu_big'] = 1;
    params_dict['range_small'] = range(5);
    params_dict['range_big'] = range(5,7);
    params_dict['path_to_test_file_small'] = '/home/maheenrashid/Downloads/deep_proposals/torch_new/test_command_line.th';
    params_dict['path_to_test_file_big'] = '/home/maheenrashid/Downloads/deep_proposals/torch_new/test_command_line_bigIm.th';
    params_dict['model'] = '/disk3/maheen_data/headC_160/noFlow_gaussian_human_softmax/final/model_all_final.dat';
    params_dict['limit'] = -1;

    params=createParams('getResultsForCatSpecific');
    params=params(**params_dict);
    script_getResultsForCatSpecific(params);
    pickle.dump(params._asdict(),open(os.path.join(params.out_dir_test,'params.p'),'wb'))
    
# /disk3/maheen_data/headC_160_noFlow_justHuman
    return
    # get some training data names (300);
    pos_data='/disk3/maheen_data/headC_160/noFlow_gaussian_human/pos_flos/positives_onlyHuman_withFlow.txt';
    pos_data=util.readLinesFromFile(pos_data);
    im_data=[pos_data_curr[:pos_data_curr.index(' ')] for pos_data_curr in pos_data];
    file_names=util.getFileNames(im_data,ext=False);
    file_names=[im_curr[:im_curr.rindex('_')] for im_curr in file_names];
    file_names=list(np.unique(file_names));

    # get their results on no flow
    
    # do the same for their results with flow





    return
    params_dict = {};
    params_dict['out_dir_test'] = '/disk3/maheen_data/headC_160_withFlow_justHuman';
    params_dict['out_dir_test_old'] = '/disk3/maheen_data/headC_160_noFlow_justHuman';
    params_dict['out_dir_im'] = os.path.join(params_dict['out_dir_test_old'],'im');
    params_dict['out_dir_bbox'] = os.path.join(params_dict['out_dir_test_old'],'results');
    params_dict['scale_idx'] = range(7);
    params_dict['out_dir_im_withPadding'] = os.path.join(params_dict['out_dir_test'],'im_with_padding');
    params_dict['out_file_jTest_text'] = os.path.join(params_dict['out_dir_im_withPadding'],'test.txt');
    params_dict['out_dir_flo'] = os.path.join(params_dict['out_dir_im_withPadding'],'flo');
    params_dict['out_dir_flo_im'] = os.path.join(params_dict['out_dir_im_withPadding'],'flo_im');
    params_dict['caffe_model_file'] = '/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction_test/examples/opticalflow/final.caffemodel';
    params_dict['clusters_file'] = '/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction_test/examples/opticalflow/clusters.mat';
    params_dict['gpu'] = 0;
    params_dict['out_file_test'] = os.path.join(params_dict['out_dir_test'],'test_data_flo.txt');
    params_dict['stride'] = 16;
    params_dict['w'] = 160;
    params_dict['mode'] = 'edge';
    params_dict['num_threads'] = multiprocessing.cpu_count();
    params_dict['continue_count'] = 4;

    params=createParams('getDataForTestingWithFlo');
    params=params(**params_dict);

    script_getDataForTestingWithFlo(params);
    pickle.dump(params._asdict(),open(os.path.join(params.out_dir_test,'params_getFloData.p'),'wb'));




if __name__=='__main__':
    main();