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

    error_files=[];
    im_names_to_keep=[];
    for im_name in im_names:
        gt_file=os.path.join(dir_gt_boxes,im_name+'.npy');
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
    results_al
    for thresh_overlap in threshes_overlap:
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


    for out_dir_curr in file_rec.keys():
        img_paths = file_rec[out_dir_curr];
        
        if continue_count<=2:
            po.NUM_THREADS = num_threads;
            po.script_saveFlos(img_paths,out_dir_curr,gpu,caffe_model_file,clusters_file,overwrite=False);
        
        flo_dir=os.path.join(out_dir_curr,'flo_files');
        match_info_file=os.path.join(out_dir_curr,'match_info.txt');
        
        files=util.getFilesInFolder(flo_dir,'.flo');
        assert len(files)==len(img_paths);

        out_dir_flo_im_curr=os.path.join(out_dir_curr,'flo_im');
        util.mkdir(out_dir_flo_im_curr);
        
        if continue_count<=3:
            dp.saveFlowImFromFloDir(flo_dir,match_info_file,out_dir_flo_im_curr)

    
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
        out_folder_curr=os.path.join(dir_npy,folder_curr);
        util.mkdir(out_folder_curr);
        out_file_pre=os.path.join(out_folder_curr,im_pre);
        args.append((folder_im_str_curr,out_file_pre,batchsize,score_files,ims,folder_im_str,idx_file));

    p=multiprocessing.Pool(num_threads);
    p.map(psr.saveWithFloResultsInCompatibleFormatMP,args);



def main():

    # '/disk3/maheen_data/pedro_val/mat_overlap_check'
    # [0.1861061112539642, 0.38541755949830003, 0.55634696151538543]
    # [0.1861061112539642, 0.38541755949830003, 0.55634696151538543]
    # [0.1861061112539642, 0.38541755949830003, 0.55634696151538543]


    params_dict={};

    params_dict['out_dir_overlap'] = os.path.join('/disk3/maheen_data/headC_160_noFlow_bbox','mat_overlaps_no_neg_1000');
    params_dict['out_file'] = os.path.join('/disk3/maheen_data/headC_160_noFlow_bbox','meta_info_multi.p');
    params_dict['thresh_overlap'] = np.arange(0.5,1,0.05);

    params=createParams('saveMetaMatOverlapInfo');
    params=params(**params_dict);
    script_saveMetaMatOverlapInfoMultiThresh(params)
    pickle.dump(params._asdict(),open(os.path.join(params.out_dir_overlap,'params_saveMetaInfo_multi.p'),'wb'));

    thresh_multi_file=params.out_file;
    script_saveRecallCurve(thresh_multi_file,thresh_multi_file+'ng');
    # script_saveRecallCurveByImageIdx(thresh_multi_file,thresh_multi_file[:thresh_multi_file.rindex('.')]+'_byIdx.png');




    return
    params_dict={};
    params_dict['out_dir_overlap'] = '/disk3/maheen_data/pedro_val/mat_overlap_check'
    params_dict['out_file'] = os.path.join('/disk3/maheen_data/pedro_val','meta_info_multi.p');
    params_dict['thresh_overlap'] = np.arange(0.5,1,0.05);
    params=createParams('saveMetaMatOverlapInfo');
    params=params(**params_dict);
    # script_saveMetaMatOverlapInfoMultiThresh(params)
    # pickle.dump(params._asdict(),open(os.path.join(params.out_dir_overlap,'params_saveMetaInfo_multi.p'),'wb'));

    thresh_multi_file='/disk3/maheen_data/pedro_val/meta_info_multi.p'
    # script_saveRecallCurve(thresh_multi_file,thresh_multi_file+'ng');
    script_saveRecallCurveByImageIdx(thresh_multi_file,thresh_multi_file[:thresh_multi_file.rindex('.')]+'_byIdx.png');


    return

    params_dict={};
    params_dict['out_dir_overlap'] = '/disk3/maheen_data/headC_160_withFlow_justHuman/mat_overlaps_1000'
    params_dict['out_file'] = os.path.join('/disk3/maheen_data/headC_160_withFlow_justHuman','meta_info_multi.p');
    params_dict['thresh_overlap'] = np.arange(0.5,1,0.05);
    params=createParams('saveMetaMatOverlapInfo');
    params=params(**params_dict);
    script_saveMetaMatOverlapInfoMultiThresh(params)
    pickle.dump(params._asdict(),open(os.path.join(params.out_dir_overlap,'params_saveMetaInfo_multi.p'),'wb'));

    thresh_multi_file='/disk3/maheen_data/headC_160_withFlow_justHuman/meta_info_multi.p'
    script_saveRecallCurve(thresh_multi_file,thresh_multi_file+'ng');

    # thresh_multi=pickle.load(open(thresh_multi_file,'rb'));
    # print len(thresh_multi);
    # for thresh_multi_curr in thresh_multi:
    #     print thresh_multi_curr.shape;

    # print thresh_multi_curr

    return
    mat_overlap_dir=os.path.join('/disk3/maheen_data/headC_160_noFlow_justHuman','mat_overlaps_1000')
    params_dict=pickle.load(open(os.path.join(mat_overlap_dir,'params_saveMetaInfo.p'),'rb'));
    # print params_dict

    params=createParams('saveMetaMatOverlapInfo');
    params=params(**params_dict);
    script_saveMetaMatOverlapInfo(params);
    pickle.dump(params._asdict(),open(os.path.join(params.out_dir_overlap,'params_saveMetaInfo.p'),'wb'));

    meta_file=params.out_file
    out_file_plot=meta_file[:meta_file.rindex('.')]+'.png';
    script_saveRecallCurve(meta_file,out_file_plot)


    return
    params_dict={};
    params_dict['dir_gt_boxes'] = '/disk3/maheen_data/val_anno_human_only';    
    params_dict['dir_canon_im'] = os.path.join('/disk3/maheen_data/headC_160_noFlow_justHuman','im','4');
    params_dict['dir_meta_test'] = os.path.join('/disk3/maheen_data/headC_160_noFlow_justHuman','results')
    params_dict['dir_result_check'] = os.path.join(params_dict['dir_meta_test'],'4');
    params_dict['power_scale_range'] = (-2,1);
    params_dict['power_step_size'] = 0.5
    params_dict['top'] = 1000;
    params_dict['stride'] = 16;
    params_dict['w'] = 160;
    params_dict['out_dir_overlap'] = os.path.join('/disk3/maheen_data/headC_160_noFlow_justHuman','mat_overlaps_'+str(params_dict['top']));


    # mat_overlap_dir=os.path.join('/disk3/maheen_data/headC_160_withFlow_justHuman','mat_overlaps_1000')
    # params_dict=pickle.load(open(os.path.join(mat_overlap_dir,'params.p'),'rb'));
    
    params_dict['overwrite']=True
    print params_dict
    # raw_input();
    params=createParams('saveMatOverlapsVal');
    params=params(**params_dict);
    script_saveMatOverlapsVal(params);
    pickle.dump(params._asdict(),open(os.path.join(params.out_dir_overlap,'params.p'),'wb'));


    return
    params_dict={};
    params_dict['dir_im']='/disk2/ms_coco/val2014';
    params_dict['dir_gt_boxes']='/disk2/mayExperiments/validation_anno';    
    
    params_dict['dir_mat_overlap']='/disk3/maheen_data/pedro_val/mat_overlap_check'
    params_dict['dir_overlap_viz']='/disk3/maheen_data/pedro_val/top_10_viz'
    
    params_dict['num_to_pick']=10;
    params_dict['pred_color']=(255,255,255)
    params_dict['gt_color']=(255,0,0);
    params_dict['num_threads']=multiprocessing.cpu_count();
    params_dict['overwrite']=True;


    params=createParams('saveAllTopPredViz');
    params=params(**params_dict);
    script_saveAllTopPredViz(params);
    pickle.dump(params._asdict(),open(os.path.join(params.dir_overlap_viz,'params.p'),'wb'));
    
    return

    params_dict={};

    params_dict['out_dir_overlap'] = os.path.join('/disk3/maheen_data/headC_160_noFlow_bbox','mat_overlaps_no_neg_1000');
    params_dict['out_file'] = os.path.join('/disk3/maheen_data/headC_160_noFlow_bbox','meta_info_no_neg.npy');
    params_dict['thresh_overlap'] = 0.5;

    params=createParams('saveMetaMatOverlapInfo');
    params=params(**params_dict);
    # script_saveMetaMatOverlapInfo(params);
    # pickle.dump(params._asdict(),open(os.path.join(params.out_dir_overlap,'params_saveMetaInfo.p'),'wb'));

    meta_file=params.out_file
    out_file_plot=meta_file[:meta_file.rindex('.')]+'.png';
    script_saveRecallCurve(meta_file,out_file_plot)

    # params_dict={};
    # params_dict['dir_gt_boxes'] = '/disk2/mayExperiments/validation_anno';    
    # params_dict['dir_canon_im'] = '/disk2/mayExperiments/validation/rescaled_images/4';
    # params_dict['dir_meta_test'] = '/disk3/maheen_data/headC_160_noFlow_bbox'
    # params_dict['dir_result_check'] = os.path.join(params_dict['dir_meta_test'],'4');
    # params_dict['power_scale_range'] = (-2,1);
    # params_dict['power_step_size'] = 0.5
    # params_dict['top'] = 1000;
    # params_dict['stride'] = 16;
    # params_dict['w'] = 160;
    # params_dict['out_dir_overlap'] = os.path.join('/disk3/maheen_data/headC_160_noFlow_bbox','mat_overlaps_no_neg_'+str(params_dict['top']));
    # params_dict['overwrite']=True;

    # params=createParams('saveMatOverlapsVal');
    # params=params(**params_dict);
    # script_saveMatOverlapsVal(params);
    # pickle.dump(params._asdict(),open(os.path.join(params.out_dir_overlap,'params.p'),'wb'));

    
    # problem_file='/disk3/maheen_data/headC_160_noFlow_bbox/mat_overlaps_1000/COCO_val2014_000000013632.npz';
    
    # im_name='COCO_val2014_000000013632';
    # dir_canonical='/disk2/mayExperiments/validation/rescaled_images/4';
    # im_path_canonical=os.path.join(dir_canonical,im_name+'.jpg');    
    # dir_meta_test='/disk3/maheen_data/headC_160_noFlow_bbox';
    
    # dir_gt_boxes='/disk2/mayExperiments/validation_anno';
    # out_dir=os.path.join(dir_meta_test,'test_neg');
    # util.mkdir(out_dir)
    # out_file=os.path.join(out_dir,im_name+'.npz');
    # stride=16;
    # w=160;
    # top=1000;
    # idx_im_name=0;
    # power_scale_range = (-2,1);
    # power_step_size = 0.5

    # power_range=np.arange(power_scale_range[0],power_scale_range[1]+power_step_size,power_step_size);
    # scales=[2**val for val in power_range];

    
    
    # psr.saveMatOverlap((im_name,im_path_canonical,dir_meta_test,scales,dir_gt_boxes,out_file,stride,w,top,idx_im_name))

    # meta_info=np.load(out_file);
    # pred_boxes=meta_info['pred_boxes']

    # print np.min(pred_boxes);


    print 'hello';



if __name__=='__main__':
    main();