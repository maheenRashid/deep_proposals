import sys;
sys.path.append('/home/maheenrashid/Downloads/debugging_jacob/python/')
import inputProcessing as ip;

import numpy as np
import util;
import os;
import visualize;
from PIL import Image,ImageDraw;
import matplotlib.pyplot as plt;
import scipy.misc;
import cPickle as pickle;
import math;
import subprocess;
import nms;
import multiprocessing;
import json;


def saveWithFloResultsInCompatibleFormatMP((folder_im_str_curr,out_file_pre,batchsize,score_files,ims,folder_im_str,idx)):

    print idx
    idx_rel=np.where(folder_im_str==folder_im_str_curr)[0];
    score_files_rel = np.unique(score_files[idx_rel]);
    
    # load all the score and box files
    score_dict={};
    box_dict={};
    for score_file in score_files_rel:
        score=np.load(score_file);
        box_file=score_file[:score_file.rindex('.')]+'_box.npy'
        box=np.load(box_file);
        # print score.shape,box.shape
        box_dict[box_file]=box;
        score_dict[score_file]=score;

    ims_rel = [ims[idx_curr] for idx_curr in idx_rel];
    row_cols=[];
    for im_curr in ims_rel:
        im_split=im_curr.split('_');
        row_curr=int(im_split[-2]);
        col_curr=int(im_split[-1]);
        row_cols.append((row_curr,col_curr));

    rows,cols=zip(*row_cols);
    
    score_mat=np.zeros((np.max(rows)+1,np.max(cols)+1));
    box_mat=np.zeros((np.max(rows)+1,np.max(cols)+1,4));

    for idx_curr in idx_rel:
        im_curr=ims[idx_curr]
        im_split=im_curr.split('_');
        row_curr=int(im_split[-2]);
        col_curr=int(im_split[-1]);
        
        score_file=score_files[idx_curr];
        box_file=score_file[:score_file.rindex('.')]+'_box.npy';

        idx_in_file,idx_file=getIdxInFile(idx_curr,batchsize);
        score_file_num = int(score_file[score_file.rindex('/')+1:score_file.rindex('.')]);
        
        assert score_file_num==idx_file;
        score_curr=score_dict[score_file][idx_in_file,0];
        box_curr=box_dict[box_file][idx_in_file,:];

        score_mat[row_curr,col_curr]=score_curr;
        box_mat[row_curr,col_curr,:]=box_curr;
    
    score_mat=np.expand_dims(score_mat,0);
    score_mat=np.expand_dims(score_mat,1);

    np.save(out_file_pre+'.npy',score_mat);
    np.save(out_file_pre+'_box.npy',box_mat);


def getIdxInFile(idx_check,batchsize):
    idx_file=idx_check/batchsize;
    idx_in_file=idx_check-(idx_file*batchsize);
    
    return idx_in_file, idx_file

def getScoreFileToTestFileMapping(dir_npy,batchsize,len_ims):
    score_files=[];
    curr_idx=0;
    idx_count=0;
    while curr_idx<len_ims:
        end_idx=min(curr_idx+batchsize,len_ims);
        batchsize_curr=end_idx-curr_idx;
        file_curr=os.path.join(dir_npy,str(idx_count)+'.npy');
        files_curr=[file_curr]*batchsize_curr;
        score_files=score_files+files_curr;
        idx_count=idx_count+1;
        curr_idx=end_idx;
    
    return np.array(score_files)

def savePaddedImFromScoreMat((im,score_mat_size,out_file,stride,w,mode)):
    if type(im)==type('str'):
        im=scipy.misc.imread(im);
    if type(score_mat_size)==type('str'):
        score_mat_size=np.load(score_mat_size)[0][0];
        score_mat_size=score_mat_size.shape;
    print im.shape,score_mat_size
    img_size_org=im.shape;
    diffs=getBufferSize(img_size_org,score_mat_size,stride,w);
    start_padding=[diff/2 for diff in diffs];
    end_padding=[diff-start_padding[idx] for idx,diff in enumerate(diffs)];
    paddings=zip(start_padding,end_padding);
    if len(im.shape)>2:
        paddings.append((0,0));

    im_padded=np.pad(im,paddings,mode);
    scipy.misc.imsave(out_file,im_padded);
    # return im_padded;


def getBBox(img_size_org,score_mat_size,r_idx,c_idx,stride=16,w=160,keepNeg=False):
    diffs=getBufferSize(img_size_org,score_mat_size,stride,w);
    idx_score=[r_idx,c_idx];
    start_idx=[idx*stride for idx in idx_score];
    start_idx=[idx-(diff/2) for idx,diff in zip(start_idx,diffs)];
    end_idx=[idx+w for idx in start_idx];
    if keepNeg:
        start_idx=[idx for idx in start_idx];
        end_idx=[idx for idx_idx, idx in enumerate(end_idx)];
    else:
        start_idx=[max(idx,0) for idx in start_idx];
        end_idx=[min(idx,img_size_org[idx_idx]) for idx_idx, idx in enumerate(end_idx)];
        
    return start_idx+end_idx;


def getBufferSize(img_size_org,score_mat_size,stride=16,w=160):
    total_size=[];
    for i in score_mat_size:
        new_size=((i-1)*stride)+w;
        total_size.append(new_size);
    # print total_size,img_size_org

    diffs=[total_size[i]-img_size_org[i] for i in range(len(total_size))];
    return diffs

def getBBoxArr(img_size_org,score_mat,stride,w):
    score_mat_size=score_mat.shape;
    bboxes=np.zeros((score_mat.size,5));
    idx=0;
    for r in range(score_mat_size[0]):
        for c in range(score_mat_size[1]):
            bboxes[idx,:4]=getBBox(img_size_org,score_mat_size,r,c,stride,w);
            bboxes[idx,4]=score_mat[r,c]
            idx+=1;
    return bboxes;

def getBBoxTight(bbox_big,seg,thresh_seg):
    idx=np.where(seg>thresh_seg);
    bbox=[np.min(idx[0]),np.min(idx[1]),np.max(idx[0]),np.max(idx[1])];
    bbox_ac=[bbox_big[idx%2]+val for idx,val in enumerate(bbox)];
    bbox_ac.append(bbox_big[4]);
    return bbox_ac;

def getBBoxArrTight(bboxArr,seg_mat,thresh_seg):
    idx_pos=np.where(bboxArr[:,4]>0);
    idx_pos=np.sort(idx_pos)[0];
    bbox_seg_all=[];
    for idx_val,val in enumerate(idx_pos):
        # print idx_val
        seg_rel=seg_mat[idx_val];
        bbox_seg_all.append(getBBoxTight(bboxArr[val],seg_rel,thresh_seg));

    return np.array(bbox_seg_all);


def plotPos(img_path,out_file,bboxArr,thresh):
    im=Image.open(img_path);
    draw = ImageDraw.Draw(im)
    idx_rel=np.where(bboxArr[:,4]>thresh)[0];
    for idx_curr in idx_rel:
        bbox_plot=bboxArr[idx_curr,[1,0,3,2]];
        # print bbox_plot
        draw.rectangle([int(val) for val in bbox_plot]);
    im.save(out_file);
    
def convertBBoxFormatToStandard(box_rel):
    box_rel=[box_rel[1],box_rel[0],box_rel[1]+box_rel[3],box_rel[0]+box_rel[2]];
    return box_rel;


def script_saveGTScaleInfo(path_to_npy,out_file=None,lim=5000):
    files=util.getFilesInFolder(path_to_npy,'.npy');
    if lim is not None:
        files.sort();
        files=files[:lim];

    file_names=util.getFileNames(files,ext=False);  
    
    size_record={};
    size_record={'medium':[],'small':[],'large':[]}
    
    small_thresh=32**2;
    medium_thresh=96**2;
    idx=0;
    for file_path,file_name in zip(files,file_names):

        bbox_all=np.load(file_path);
        for bbox_idx,bbox in enumerate(bbox_all):
            # print bbox_all,bbox,bbox_idx;
            area=bbox[2]*bbox[3];
            if area<small_thresh:
                size_record['small'].append((file_name,area,bbox_idx));
            elif area<medium_thresh:
                size_record['medium'].append((file_name,area,bbox_idx));
            else:
                assert area>=medium_thresh
                size_record['large'].append((file_name,area,bbox_idx));
        idx=idx+1;
        # raw_input();

    for key_curr in size_record.keys():
        print key_curr,len(size_record[key_curr])

    if out_file is None:
        out_file=os.path.join(path_to_npy,'index_'+str(lim)+'.p');

    pickle.dump(size_record,open(out_file,'wb'));
    return out_file;

def getBBoxIOU(bbox1,bbox2):
    (right,left)=(bbox1,bbox2) if bbox1[0] < bbox2[0] else (bbox2,bbox1);
    (top,bottom)=(bbox1,bbox2) if bbox1[1] < bbox2[1] else (bbox2,bbox1);
    if right[2]<left[0] or top[3]<bottom[1]:
        return 0.0;
    else:
        return util.getIOU(np.array(bbox1,dtype=int),np.array(bbox2,dtype=int));


def script_saveValidationTexts(dir_validations,out_files,ext='.jpg'):
    for dir_curr,out_file_curr in zip(dir_validations,out_files):
        files=util.getFilesInFolder(dir_curr,ext);
        util.writeFile(out_file_curr,files);


def parseTorchCommand(path_to_test_file,model,testFile,outDir,splitModel=False,limit=-1,gpu=1,overwrite=False,saveSeg=False):
    command=['th'];
    command.append(path_to_test_file);
    command.extend(['-model',model]);
    command.extend(['-testFile',testFile]);
    command.extend(['-outDir',outDir]);
    if splitModel:
        command.append('-splitModel');
    if limit!=-1:
        command.extend(['-limit',str(limit)]);
    if overwrite:
        command.append('-overwrite');
    if saveSeg:
        command.append('-saveSeg');

    command.extend(['-gpu',str(gpu)]);
    command=' '.join(command);
    return command;


def script_writeTestSH(out_files,out_dirs,out_file_sh,path_to_test_file,model,limit,gpu,overwrite=False,saveSeg=False):
    commands_all=[];
    for test_file,out_dir in zip(out_files,out_dirs):
        # print test_file,out_dir
        command=parseTorchCommand(path_to_test_file,model,test_file,out_dir,limit=limit,gpu=gpu,overwrite=overwrite,saveSeg=saveSeg);
        print command
        commands_all.append(command);

    util.writeFile(out_file_sh,commands_all);


def getPredBoxesCanonical(file_score,file_seg,scale_curr,im_size,stride=16,w=160,thresh_seg=0.2):
    pred_scores=np.load(file_score)[0][0];
    pred_segs=np.load(file_seg);

    
    if len(im_size)>2:
        img_size_curr=(im_size[0]*scale_curr,im_size[1]*scale_curr,im_size[2]);
    else:
        img_size_curr=(im_size[0]*scale_curr,im_size[1]*scale_curr);

    bboxArr=getBBoxArr(img_size_curr,pred_scores,stride,w);
    pred_boxes=getBBoxArrTight(bboxArr,pred_segs,thresh_seg);

    # print pred_boxes.shape

    confidence=pred_boxes[:,-1];
    pred_boxes=pred_boxes[:,:-1]/float(scale_curr);
    return pred_boxes,confidence;


def getBoxContainingCanonical(row,col,im_size,scale_curr,seg_shape,stride,w):
    if len(im_size)>2:
        img_size_curr=(im_size[0]*scale_curr,im_size[1]*scale_curr,im_size[2]);
    else:
        img_size_curr=(im_size[0]*scale_curr,im_size[1]*scale_curr);
    pred_box=getBBox(img_size_curr,seg_shape,row,col,stride,w,keepNeg=True)
    pred_box=np.array(pred_box)/float(scale_curr);
    return pred_box;    

def getPredBoxesCanonicalFromBBoxTight(box_curr,row,col,im_size,scale_curr,seg_shape,stride,w):
    if len(im_size)>2:
        img_size_curr=(im_size[0]*scale_curr,im_size[1]*scale_curr,im_size[2]);
    else:
        img_size_curr=(im_size[0]*scale_curr,im_size[1]*scale_curr);
    bbox_containing=getBBox(img_size_curr,seg_shape,row,col,stride,w,keepNeg=True)
    pred_box=getBBoxTightCanonical(bbox_containing,box_curr)
    pred_box=np.array(pred_box)/float(scale_curr);
    pred_box_bef=pred_box
    pred_box=[max(pred_box[0],0),max(pred_box[1],0),min(pred_box[2],im_size[0]),min(pred_box[3],im_size[1])]
    return pred_box;


def getBBoxTightCanonical(bbox_big,bbox):
    bbox_ac=[bbox_big[idx%2]+val for idx,val in enumerate(bbox)];
    # bbox_ac.append(bbox_big[4]);
    return bbox_ac;


def script_checkGPUImplementation():

    dir_meta_validation='/disk2/mayExperiments/validation/rescaled_images'
    dir_meta_test='/disk3/maheen_data/headC_160_noFlow_bbox/'
    dir_validations=[os.path.join(dir_meta_test,str(num)) for num in [6]];
    
    dir_old=dir_validations[0];
    dir_new=dir_old+'_test';

    files=util.getFileNames(util.getFilesInFolder(dir_new,ext='.npy'));
    print files;
    for file_curr in files:
        new=np.load(os.path.join(dir_new,file_curr));
        old=np.load(os.path.join(dir_old,file_curr));
        print new.shape,old.shape,np.allclose(new,old,atol=1e-4);
        print new
        print old


    # return    

def getTopBBoxAndScores(im_name,im_path_canonical,dir_meta_test,scales,stride,w,top):
    # im_name=im_names[1];
    # im_path=im_list[1];
    # print im_name,im_path,
    # im_path_canonical=os.path.join(dir_canon_im,im_name+'.jpg')
    im_size=scipy.misc.imread(im_path_canonical);
    im_size=im_size.shape;
    # print im_size;


    pred_info=[]

    for scale_idx in range(len(scales)):
        out_dir_curr=os.path.join(dir_meta_test,str(scale_idx));
        
        scale_curr=scales[scale_idx];
        # print out_dir_curr,scale_curr
        file_seg=os.path.join(out_dir_curr,im_name+'_box.npy');
        file_score=os.path.join(out_dir_curr,im_name+'.npy');
        if os.path.exists(file_seg):
            assert os.path.exists(file_score);
    
            pred_info.append([file_score,file_seg,scale_curr]);

    files_score=[pred_info_curr[0] for pred_info_curr in pred_info];

    [files_score,files_seg,scales]=zip(*pred_info);
    
    files_score=list(files_score);
    files_seg=list(files_seg);
    scales=list(scales);


    scores=[];
    
    for idx_file,file_score in enumerate(files_score):
        scores_curr=np.load(file_score)[0][0];
        for i in range(scores_curr.shape[0]):
            for j in range(scores_curr.shape[1]):
                scores.append([scores_curr[i,j],idx_file,i,j]);

    scores=np.array(scores);
    segs=[np.load(file_seg) for file_seg in files_seg];

    scores_sorted=scores[np.argsort(scores[:,0])[::-1],:];
    

    bboxes=np.zeros((top,4));
    scores_record=np.zeros((top,4));

    idx_to_fill=0;
    for score_curr in scores_sorted:
        # print idx_to_fill
        # print score_curr
        file_idx=int(score_curr[1]);
        row=int(score_curr[2]);
        col=int(score_curr[3]);
        scale_curr=scales[file_idx];
        box_curr=segs[file_idx][row,col];

        
        
        if box_curr[0]<0:
            print 'problem',score_curr,box_curr
            continue;

        seg_shape=(segs[file_idx].shape[0],segs[file_idx].shape[1])
        # print box_curr,segs[file_idx].shape

        # print 'score_curr',score_curr
        # print '(box_curr,row,col,im_size,scale_curr,seg_shape,stride,w)',(box_curr,row,col,im_size,scale_curr,seg_shape,stride,w)

        box_curr=getPredBoxesCanonicalFromBBoxTight(box_curr,row,col,im_size,scale_curr,seg_shape,stride,w)
        if min(box_curr)<0:
            print 'problem edge ',score_curr,box_curr
            continue;
        # print box_curr
        bboxes[idx_to_fill]=box_curr;
        scores_record[idx_to_fill]=score_curr;
        idx_to_fill+=1;
        if idx_to_fill==top:
            break;

    return bboxes,scores_record;


def getMatOverlap(pred_boxes_all,gt_boxes):

    mat_overlap=np.zeros((pred_boxes_all.shape[0],gt_boxes.shape[0]));
    for pred_idx in range(mat_overlap.shape[0]):
        for gt_idx in range(mat_overlap.shape[1]):
            mat_overlap[pred_idx,gt_idx]=getBBoxIOU(pred_boxes_all[pred_idx],gt_boxes[gt_idx]);

    # print mat_overlap;
    return mat_overlap


def saveMatOverlap((im_name,im_path_canonical,dir_meta_test,scales,dir_gt_boxes,out_file,stride,w,top,idx)):
    print idx
    pred_boxes,pred_scores=getTopBBoxAndScores(im_name,im_path_canonical,dir_meta_test,scales,stride,w,top)
    # print pred_boxes.shape,pred_scores.shape
    gt_file=os.path.join(dir_gt_boxes,im_name+'.npy');
    gt_boxes=np.load(gt_file);
    # print gt_boxes.shape
    gt_boxes_size=[box_curr[2]*box_curr[3] for box_curr in gt_boxes];
    # print gt_boxes[0]
    gt_boxes=np.array([convertBBoxFormatToStandard(gt_box) for gt_box in gt_boxes]);
    # print gt_boxes[0],gt_boxes_size[0]
    mat_overlap=getMatOverlap(pred_boxes,gt_boxes)
    # print mat_overlap.shape
    # print np.max(mat_overlap,axis=0);
    # print gt_boxes_size
    np.savez(out_file,mat_overlap=mat_overlap,gt_boxes_size=gt_boxes_size,pred_boxes=pred_boxes,pred_scores=pred_scores);

def script_saveMatOverlapsVal():
    dir_meta_validation='/disk2/mayExperiments/validation/rescaled_images'
    dir_gt_boxes='/disk2/mayExperiments/validation_anno';
    dir_meta_test='/disk3/maheen_data/headC_160_noFlow_bbox/'
    
    dir_canon_im='/disk2/mayExperiments/validation/rescaled_images/4'
    dir_validations=[os.path.join(dir_meta_validation,str(num)) for num in range(7)];
    dir_results=[os.path.join(dir_meta_test,str(num)) for num in range(7)];
    power_scale_range=(-2,1);
    power_step_size=0.5
    top=1000;
    stride=16;
    w=160;
    out_dir_overlap='/disk3/maheen_data/headC_160_noFlow_bbox/mat_overlaps_'+str(top)
    util.mkdir(out_dir_overlap);

    
    im_names=util.getFileNames(util.getFilesInFolder(dir_results[0],ext='_box.npy'),ext=False);
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

    power_range=np.arange(power_scale_range[0],power_scale_range[1]+power_step_size,power_step_size);
    scales=[2**val for val in power_range];

    args=[];
    for idx_im_name,im_name in enumerate(im_names_to_keep):
        im_path_canonical=os.path.join(dir_canon_im,im_name+'.jpg');
        out_file=os.path.join(out_dir_overlap,im_name+'.npz');
        
        if os.path.exists(out_file):
            continue;

        args.append((im_name,im_path_canonical,dir_meta_test,scales,dir_gt_boxes,out_file,stride,w,top,idx_im_name))

    # print len(args);

    p=multiprocessing.Pool(multiprocessing.cpu_count());
    p.map(saveMatOverlap,args)


def getNumCorrect(file_curr,num_to_pick,threshes_size,thresh_overlap=0.5):
    results=np.load(file_curr);
    mat_overlap=results['mat_overlap'];
    gt_boxes_size=results['gt_boxes_size'];
    overlap_rel=np.max(mat_overlap[:num_to_pick,:],axis=0);
    
    assert len(overlap_rel)==len(gt_boxes_size);
    gt_boxes_size=np.array(gt_boxes_size);
    arr_small=gt_boxes_size<threshes_size[0];
    arr_medium= np.logical_and(gt_boxes_size<=threshes_size[1],gt_boxes_size>=threshes_size[0]);
    arr_large= threshes_size[1]<=gt_boxes_size
    
    arrs=[arr_small,arr_medium,arr_large];
    
    assert sum([sum(arr_curr) for arr_curr in arrs])==len(gt_boxes_size);
    
    performace=[];
    
    for arr_curr in arrs:
        if sum(arr_curr)!=0:
            val_curr=overlap_rel[arr_curr];
            val_curr=sum(val_curr>=thresh_overlap);
            avg=val_curr/float(sum(arr_curr));
        else:
            avg=-1;
        performace.append(avg);

    # print overlap_rel
    # print gt_boxes_size
    # print threshes_size
    # print arrs
    return performace;


def getIdxCorrect((file_curr,thresh_overlap,idx_arg)):
    print idx_arg;
    results=np.load(file_curr);
    mat_overlap=results['mat_overlap'];
    gt_boxes_size=results['gt_boxes_size'];
    idx_correct=mat_overlap>=thresh_overlap;
    # print idx_correct.shape

    idx_correct_new=[];
    for idx in range(idx_correct.shape[1]):
        idx_rel=np.where(idx_correct[:,idx])[0];
        if len(idx_rel)==0:
            idx_correct_new.append(-1);
        else:
            idx_correct_new.append(np.min(idx_rel));
    idx_correct_new=np.expand_dims(idx_correct_new,1);
    gt_boxes_size=np.expand_dims(gt_boxes_size,1);
    # print idx_correct_new.shape,gt_boxes_size.shape

    toReturn=np.hstack((idx_correct_new,gt_boxes_size));

    return toReturn

def getIdxCorrectWithImageId((file_curr,thresh_overlap,idx_arg)):
    print idx_arg;
    img_idx=int(file_curr[file_curr.rindex('_')+1:file_curr.rindex('.')]);
    
    results=np.load(file_curr);
    mat_overlap=results['mat_overlap'];
    gt_boxes_size=results['gt_boxes_size'];
    idx_correct=mat_overlap>=thresh_overlap;
    # print idx_correct.shape

    idx_correct_new=[];
    for idx in range(idx_correct.shape[1]):
        idx_rel=np.where(idx_correct[:,idx])[0];
        if len(idx_rel)==0:
            idx_correct_new.append(-1);
        else:
            idx_correct_new.append(np.min(idx_rel));

    idx_correct_new=np.expand_dims(idx_correct_new,1);
    gt_boxes_size=np.expand_dims(gt_boxes_size,1);
    img_idx_all=img_idx*np.ones(idx_correct_new.shape)
    # print idx_correct_new.shape,gt_boxes_size.shape

    toReturn=np.hstack((idx_correct_new,gt_boxes_size,img_idx_all));

    return toReturn


def script_saveMetaInformation():
    top=1000;
    out_dir_overlap='/disk3/maheen_data/headC_160_noFlow_bbox/mat_overlaps_'+str(top)
    out_file=os.path.join(out_dir_overlap,'meta_info.npy');
    print out_file

    mats_overlap=util.getFilesInFolder(out_dir_overlap,ext='.npz');
    print len(mats_overlap);
    raw_input();
    # mats_overlap=mats_overlap[:10]
    
    thresh_overlap=0.5;
    args=[];
    for idx_mat_overlap,mat_overlap in enumerate(mats_overlap):
        args.append((mat_overlap,thresh_overlap,idx_mat_overlap));

    p=multiprocessing.Pool(multiprocessing.cpu_count());
    results=p.map(getIdxCorrect,args);
    results=np.vstack(results);

    print results.shape
    np.save(out_file,results);
    
def script_saveRecallProposalCurve():
    top=1000;
    out_dir_overlap='/disk3/maheen_data/headC_160_noFlow_bbox/mat_overlaps_'+str(top)
    
    meta_file=os.path.join(out_dir_overlap,'meta_info.npy');
    threshes_size=[32**2,96**2];
    meta_info=np.load(meta_file);
    print meta_info.shape
    out_file_plot=meta_file[:meta_file.rindex('.')]+'.png';
    
    rel_idx=[];
    rel_idx.append(meta_info[:,1]<threshes_size[0]);
    rel_idx.append(np.logical_and(meta_info[:,1]<=threshes_size[1],meta_info[:,1]>=threshes_size[0]));
    rel_idx.append(meta_info[:,1]>threshes_size[1]);
    rel_idx.append(meta_info[:,1]>=-1);
    assert np.sum(rel_idx[-1])==meta_info.shape[0]
    # assert sum([np.sum(rel_idx_curr) for rel_idx_curr in rel_idx])==meta_info.shape[0]

    counts_correct=[[],[],[],[]]
    for top_curr in range(1000):

        for idx_rel_idx_curr,rel_idx_curr in enumerate(rel_idx):
            overlaps=meta_info[rel_idx_curr,0];
            correct=np.sum(np.logical_and(overlaps>=0,overlaps<=top_curr));
            total=overlaps.shape[0];
            ratio=correct/float(total);
            counts_correct[idx_rel_idx_curr].append(ratio);
            # print overlaps.shape,np.sum(rel_idx_curr);
    
    # aucs=[np.mean(counts_curr) for counts_curr in counts_correct];
    # print aucs
    print [counts_correct[-1][idx-1] for idx in [10,100,1000]];
    xAndYs=[(range(1,1001),counts_curr) for counts_curr in counts_correct];
    legend=['small','medium','large','total'];
    visualize.plotSimple(xAndYs,out_file_plot,title='Average Recall',xlabel='Number of Proposals',ylabel='Average Recall',legend_entries=legend,loc=0);
    print out_file_plot.replace('/disk3','vision3.cs.ucdavis.edu:1001');


def main():
    script_saveRecallProposalCurve()
    # script_saveMetaInformation()
    # script_saveMatOverlapsVal()

    return
    
    curr_range=range(5,7);
    path_to_res='/disk3/maheen_data/headC_160_noFlow_bbox'
    path_to_text='/disk2/mayExperiments/validation/rescaled_images'
    out_files=[os.path.join(path_to_text,str(num)+'_leftover.txt') for num in curr_range];
    out_dirs=[os.path.join(path_to_res,str(num)) for num in curr_range];
    limit=-1
    gpu=1;
    path_to_test_file='/home/maheenrashid/Downloads/deep_proposals/torch_new/test_command_line_bigIm.th';
    model='/disk2/aprilExperiments/headC_160/noFlow_gaussian_all_actual/intermediate_resume_2/model_all_70000.dat';
    
    gpu=1;
    limit=-1;
    out_file_sh=os.path.join(path_to_res,'test_leftover_big.sh');
    script_writeTestSH(out_files,out_dirs,out_file_sh,path_to_test_file,model,limit,gpu,overwrite=False)
    print out_file_sh


    return
    path_to_im='/disk2/ms_coco/val2014'

    top=1000;
    
    pred_folder='/disk3/maheen_data/headC_160_noFlow_bbox/4'
    anno_dir='/disk2/mayExperiments/validation_anno';
    
    out_dir_meta_resize='/disk2/mayExperiments/validation/rescaled_images';

    pred_files=util.getFilesInFolder(pred_folder,'_box.npy');
    anno_files=util.getFilesInFolder(anno_dir,'.npy');
    anno_ims=util.getFileNames(anno_files,ext=False);

    anno_ims.sort();
    anno_ims_rel=anno_ims[:5000];
    ims_leftover=[];
    for anno_im in anno_ims_rel:
        file_curr=os.path.join(pred_folder,anno_im+'_box.npy');
        if not os.path.exists(file_curr):
            im_curr=os.path.join(path_to_im,anno_im+'.jpg');
            assert os.path.exists(im_curr);
            ims_leftover.append(im_curr);


    print len(ims_leftover)
    print ims_leftover[0]
    # ip.rescaleImAndSaveMeta(ims_leftover,out_dir_meta_resize)
    out_files_text=[]
    for scale_idx in range(7):
        dir_curr=os.path.join(out_dir_meta_resize,str(scale_idx));
        file_list=[];
        for im_curr in util.getFileNames(ims_leftover,ext=False):
            file_curr=os.path.join(dir_curr,im_curr+'.jpg')
            print file_curr
            assert os.path.exists(file_curr)
            file_list.append(file_curr);
        print len(file_list);
        out_file_text=dir_curr+'_leftover.txt';
        out_files_text.append(out_file_text);
        util.writeFile(out_file_text,file_list);

    for out_file_curr in out_files_text:
        print out_file_curr

if __name__=='__main__':
    main();