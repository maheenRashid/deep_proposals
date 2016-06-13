import numpy as np;
import os;
import util;
import visualize;
import data_preprocessing as dp;
import processScoreResults as psr;
import scipy.misc
import multiprocessing;
import scripts_resultProcessing as srp;

from utils import cython_bbox

def script_saveMatOverlapInfo():
    val_file='/disk2/januaryExperiments/pedro_data/coco-proposals/val.npy';
    val_data=np.load(val_file);
    print val_data.shape;
    print val_data[101,:];
    print 'hello';

    path_to_im='/disk2/ms_coco/val2014';
    path_to_gt='/disk2/mayExperiments/validation_anno';
    path_to_pred_us='/disk3/maheen_data/headC_160_noFlow_bbox/mat_overlaps_1000';

    im_pre='COCO_val2014_';
    im_ids=val_data[:,0];
    im_ids=np.unique(im_ids);
    im_ids=np.sort(im_ids);
    # im_ids=im_ids[:5000];
    print im_ids;

    im_files=[dp.addLeadingZeros(int(im_id),os.path.join(path_to_im,im_pre),'.jpg') for im_id in im_ids];
    im_names=util.getFileNames(im_files,ext=False);
    print im_names[0]
    out_dir='/disk3/maheen_data/pedro_val/mat_overlap'

    idx_count=0;
    problem=0;
    for im_id,im_name,im_file in zip(im_ids,im_names,im_files):
        print idx_count
        rel_rows=val_data[val_data[:,0]==im_id,:];
        assert np.unique(rel_rows[:,0]).size==1
        out_file=os.path.join(out_dir,im_name+'.npz');
        gt_file=os.path.join(path_to_gt,im_name+'.npy');
        comparison_file=os.path.join(path_to_pred_us,im_name+'.npz');

        if os.path.exists(gt_file) and os.path.exists(comparison_file):
            idx_count=idx_count+1;
            saveMatOverlapInfo(out_file,gt_file,rel_rows)
        else:
            problem=problem+1;

        if idx_count==5000:
            break;  


def saveMatOverlapInfo(out_file,gt_file,rel_rows):
    scores=rel_rows[:,-2];
    sort_idx=np.argsort(scores)[::-1];
    rel_rows=rel_rows[sort_idx,:];
    pred_boxes=rel_rows[:,1:5];
    # print pred_boxes.shape

    pred_boxes=np.array([psr.convertBBoxFormatToStandard(pred_box) for pred_box in pred_boxes])
    pred_scores=rel_rows[:,-2];

    gt_boxes=np.load(gt_file);
    gt_boxes_size=[box_curr[2]*box_curr[3] for box_curr in gt_boxes];
    
    gt_boxes=np.array([psr.convertBBoxFormatToStandard(gt_box) for gt_box in gt_boxes]);
    
    mat_overlap=psr.getMatOverlap(pred_boxes,gt_boxes)
    np.savez(out_file,mat_overlap=mat_overlap,gt_boxes_size=gt_boxes_size,pred_boxes=pred_boxes,pred_scores=pred_scores);


def getIdxCorrectSanityCheck(mat_overlap_file,gt_file,im_file):
    im=scipy.misc.imread(im_file);
    gt_boxes=np.load(gt_file);
    gt_boxes=np.array([psr.convertBBoxFormatToStandard(gt_box) for gt_box in gt_boxes]);

    mat_info=np.load(mat_overlap_file);
    pred_boxes=mat_info['pred_boxes'];
    
    min_arr=np.zeros((pred_boxes.shape[0],2));
    min_arr[:,0]=pred_boxes[:,1];
    pred_boxes[:,1]=np.max(min_arr,axis=1);

    min_arr=np.zeros((pred_boxes.shape[0],2));
    min_arr[:,0]=pred_boxes[:,0];
    pred_boxes[:,0]=np.max(min_arr,axis=1);

    max_r=im.shape[0]*np.ones((pred_boxes.shape[0],2));
    max_r[:,0]=pred_boxes[:,2];
    pred_boxes[:,2]=np.min(max_r,axis=1);

    max_r=im.shape[1]*np.ones((pred_boxes.shape[0],2));
    max_r[:,0]=pred_boxes[:,3];
    pred_boxes[:,3]=np.min(max_r,axis=1);

    mat_overlap_new=psr.getMatOverlap(pred_boxes,gt_boxes)

    correct=mat_overlap_new>=0.5
    idx_correct=[];
    for c in range(mat_overlap_new.shape[1]):
        idx_correct_curr=np.where(correct[:,c])[0]
        if idx_correct_curr.size==0:
            c_curr=-1;
        else:
            c_curr=np.min(idx_correct_curr);
        idx_correct.append(c_curr);

    return idx_correct

def fixMatOverlap((mat_overlap_file,gt_file,im_file,out_file,idx)):
    print idx;
    im=scipy.misc.imread(im_file);
    gt_boxes=np.load(gt_file);
    gt_boxes=np.array([psr.convertBBoxFormatToStandard(gt_box) for gt_box in gt_boxes]);

    mat_info=np.load(mat_overlap_file);

    pred_scores = mat_info['pred_scores']
    gt_boxes_size = mat_info['gt_boxes_size']
    mat_overlap = mat_info['mat_overlap']
    pred_boxes = mat_info['pred_boxes']

    # print mat_info.keys();
    pred_boxes=mat_info['pred_boxes'];
    
    min_arr=np.zeros((pred_boxes.shape[0],2));
    min_arr[:,0]=pred_boxes[:,1];
    pred_boxes[:,1]=np.max(min_arr,axis=1);

    min_arr=np.zeros((pred_boxes.shape[0],2));
    min_arr[:,0]=pred_boxes[:,0];
    pred_boxes[:,0]=np.max(min_arr,axis=1);

    max_r=im.shape[0]*np.ones((pred_boxes.shape[0],2));
    max_r[:,0]=pred_boxes[:,2];
    pred_boxes[:,2]=np.min(max_r,axis=1);

    max_r=im.shape[1]*np.ones((pred_boxes.shape[0],2));
    max_r[:,0]=pred_boxes[:,3];
    pred_boxes[:,3]=np.min(max_r,axis=1);

    # mat_overlap_new=psr.getMatOverlap(pred_boxes,gt_boxes)

    mat_overlap_new=cython_bbox.bbox_overlaps(np.array(pred_boxes,dtype=np.float),np.array(gt_boxes,dtype=np.float));
    
    np.savez(out_file,pred_scores = pred_scores,gt_boxes_size = gt_boxes_size,mat_overlap = mat_overlap_new,pred_boxes = pred_boxes)

def script_reSaveMatOverlap():

    dir_old='/disk3/maheen_data/pedro_val/mat_overlap';
    dir_new='/disk3/maheen_data/pedro_val/mat_overlap_check';

    util.mkdir(dir_new);
    
    path_to_im='/disk2/ms_coco/val2014';
    path_to_gt='/disk2/mayExperiments/validation_anno';

    mat_overlap_files=util.getFilesInFolder(dir_old,ext='.npz');
    im_names=util.getFileNames(mat_overlap_files,ext=False);

    args=[];
    for idx,(mat_overlap_file,im_name) in enumerate(zip(mat_overlap_files,im_names)):
        gt_file=os.path.join(path_to_gt,im_name+'.npy');
        im_file=os.path.join(path_to_im,im_name+'.jpg');
        out_file=os.path.join(dir_new,im_name+'.npz');
        # if os.path.exists(out_file):
        #     continue;

        args.append((mat_overlap_file,gt_file,im_file,out_file,idx));

    p = multiprocessing.Pool(32);
    p.map(fixMatOverlap, args);    


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


def visualizeScores(path_to_data,out_file):
    data=np.load(path_to_data);
    scores=data[:,-2];
    print np.unique(data[:,-1]);
    print np.min(scores);
    print np.max(scores);
    visualize.hist(scores,out_file,bins=100,normed=True,xlabel='Value',ylabel='Frequency',title='Predicted Scores')





def main():

    path_to_data='/disk2/januaryExperiments/pedro_data/coco-proposals/val.npy';
    out_file='/disk2/temp/pedro_score_hist.png';

    visualizeScores(path_to_data,out_file)


    return
    # script_reSaveMatOverlap()

    # path_to_gt='/disk2/mayExperiments/validation_anno';
    path_pedro='/disk3/maheen_data/pedro_val';
    mat_overlap_dir=os.path.join(path_pedro,'mat_overlap_check');


    # mat_overlap_files=util.getFilesInFolder(mat_overlap_dir,ext='.npz');
    # im_names = util.getFileNames(mat_overlap_files, ext=False);
    



    # return
    # path_us='/disk3/maheen_data/headC_160_noFlow_bbox';
    # mat_overlap_dir=os.path.join(path_us,'mat_overlaps_1000');

    params_dict={};

    params_dict['out_dir_overlap'] = mat_overlap_dir
    params_dict['out_file'] = os.path.join(path_pedro,'meta_info_with_idx.npy');
    params_dict['thresh_overlap'] = 0.5;

    params=srp.createParams('saveMetaMatOverlapInfo');
    params=params(**params_dict);
    
    srp.script_saveMetaMatOverlapInfoWithIndex(params)
    out_file_plot=params.out_file[:params.out_file.rindex('.')]+'.png';
    mat_overlap=np.load(params.out_file);
    img_idx=np.unique(mat_overlap[:,2]);
    print img_idx.shape;

    

    
    args=[]
    for idx_img_idx_curr,img_idx_curr in enumerate(img_idx):
        # print idx_img_idx_curr
        args.append((mat_overlap,img_idx_curr));

    p=multiprocessing.Pool(32);
    counts_correct_all=p.map(getCountsCorrectByIdx,args);


    counts_comb=[[],[],[],[]];
    for r in range(4):
        for counts_correct in counts_correct_all:
            if len(counts_correct[r])>0:
                counts_comb[r].append(counts_correct[r]);
    counts_comb=[np.array(curr) for curr in counts_comb];
    avgs=[np.mean(curr,axis=0) for curr in counts_comb];


    print [avgs[-1][idx-1] for idx in [10,100,1000]];
    xAndYs=[(range(1,1001),counts_curr) for counts_curr in avgs];
    legend=['small','medium','large','total'];
    visualize.plotSimple(xAndYs,out_file_plot,title='Average Recall',xlabel='Number of Proposals',ylabel='Average Recall',legend_entries=legend,loc=0);
    print out_file_plot.replace('/disk3','vision3.cs.ucdavis.edu:1001');



    
    
    return
    meta_old=os.path.join(path_pedro,'meta_info.npy');
    meta_old=np.load(meta_old);
    print meta_old.shape

    meta_new=os.path.join(path_us,'meta_info.npy');
    meta_new=np.load(meta_new);
    print meta_new.shape
    print np.sum(meta_old[:,0]<0);
    print np.sum(meta_new[:,0]<0);
    
    return



    for mat_overlap_file,im_name in zip(mat_overlap_files,im_names):

        mat_info = np.load(mat_overlap_file);
        pred_boxes = mat_info['pred_boxes'];
        mat_overlap = mat_info['mat_overlap'];

        idx_to_rem=np.where(np.min(pred_boxes,axis=1)<0)[0];
        if idx_to_rem.size==0:
            continue;
        im_file=os.path.join(path_to_im,im_name+'.jpg');
    
        gt_file=os.path.join(path_to_gt,im_name+'.npy');

        idx_correct = psr.getIdxCorrect((mat_overlap_file,0.5,0))
        idx_correct=idx_correct[:,0];
        idx_correct_sanity_check=getIdxCorrectSanityCheck(mat_overlap_file,gt_file,im_file);
        idx_correct_sanity_check=np.array(idx_correct_sanity_check,dtype=type(idx_correct[0]));
        if not np.array_equal(idx_correct, idx_correct_sanity_check):

            print idx_correct, idx_correct_sanity_check
            raw_input();

        # raw_input();
    # print idx_correct
    # print mat_overlap[idx_diff,:]
    # print mat_overlap_new[idx_diff,:];

    # print np.allclose(mat_overlap,mat_overlap_new);
        



    

if __name__=='__main__':
    main();