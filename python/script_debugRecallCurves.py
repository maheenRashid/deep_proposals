import numpy as np;
import os;
import scripts_resultProcessing as srp;
import processScoreResults as psr;
import util;
import cPickle as pickle;
import json;
import multiprocessing;
import sys;
sys.path.append('/home/maheenrashid/Downloads/deep_proposals/coco-master/PythonAPI')
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import visualize;

def script_saveToExploreBoxes():
    # dir_mat_overlap='/disk3/maheen_data/headC_160_noFlow_bbox/mat_overlaps_1000';
    dir_mat_overlap='/disk3/maheen_data/pedro_val/mat_overlap';
    scratch_dir='/disk3/maheen_data/scratch_dir'
    util.mkdir(scratch_dir);
    out_file=os.path.join(scratch_dir,'them_neg_box.p');
    
    
    overlap_files=util.getFilesInFolder(dir_mat_overlap,ext='.npz');
    print len(overlap_files)
    to_explore=[];
    
    for idx_overlap_file,overlap_file in enumerate(overlap_files):
        print idx_overlap_file
        meta_data=np.load(overlap_file);
        pred_boxes=meta_data['pred_boxes'];
        # print pred_boxes.shape
        min_boxes=np.min(pred_boxes,axis=1);
        num_neg=np.sum(min_boxes<0);
        if num_neg>0:
            to_explore.append((overlap_file,pred_boxes));

    print len(to_explore);

    pickle.dump(to_explore,open(out_file,'wb'));

def formatRows((row,idx)):
    # print row[0];
    print idx;
    dict_curr={unicode('image_id'):int(row[0]),
        unicode('category_id'):int(row[-1]),
        unicode('bbox'):[float(val) for val in row[1:5]],
        unicode('score'):float(row[-2])};
    return dict_curr;
    

def script_reformatResultsForCOCOEval():
    val_results_eg='/home/maheenrashid/Downloads/deep_proposals/coco-master/results/instances_val2014_fakebbox100_results.json'    
    data=json.load(open(val_results_eg,'rb'));

    pedro_data_org='/disk2/januaryExperiments/pedro_data/coco-proposals/val.npy';
    out_file='/home/maheenrashid/Downloads/deep_proposals/coco-master/results/instances_val2014_maheen_results.json'   
    data=np.load(pedro_data_org);
    img_ids=data[:,0];
    img_ids=np.unique(img_ids);
    img_ids=np.sort(img_ids);
    img_ids=img_ids[:5000]


    idx_keep=np.in1d(data[:,0], img_ids);
    data=data[idx_keep,:];

    args=[];
    for idx,row in enumerate(data):
        args.append((row,idx));
        
    p = multiprocessing.Pool(multiprocessing.cpu_count());
    data_formatted = p.map(formatRows,args);

    json.dump(data_formatted,open(out_file,'w'))

def script_recallCheck():

    coco_eval_file='/disk2/temp/recall_check.p';
    # note. only for top 50
    out_file=coco_eval_file+'ng';
    coco_eval=pickle.load(open(coco_eval_file,'rb'));
    print coco_eval['recall'].shape
    
    recall=coco_eval['recall'];
    labels=['all','small','medium','large'];
    xAndYs=[];
    for idx in range(len(labels)):
        rec_curr=recall[:,:,idx,:];
        print rec_curr.shape
        rec_curr= np.mean(rec_curr,axis=0);
        print rec_curr.shape;
        rec_curr=rec_curr.ravel();
        print rec_curr.shape;
        # raw_input();
        xAndYs.append((range(len(rec_curr)),rec_curr));

    visualize.plotSimple(xAndYs,out_file,title='Avg Recall',xlabel='Number of proposals',ylabel='Avg Recall',legend_entries=labels)
    print xAndYs[0][1][9],xAndYs[0][1][99],xAndYs[0][1][999]
    print [np.mean(a[1]) for a in xAndYs];


def loadAndPickN((mat_overlap_file,num_to_pick,idx)):
    print idx;
    mat_overlap=np.load(mat_overlap_file);
    pred_scores=mat_overlap['pred_scores'];
    pred_scores_scores=pred_scores[:,0];
    assert np.array_equal(pred_scores_scores,np.sort(pred_scores_scores)[::-1]);
    pred_scores=pred_scores[:num_to_pick,:];
    return pred_scores;

def script_saveBestImage():
    dir_overlaps = '/disk3/maheen_data/headC_160_noFlow_bbox/mat_overlaps_no_neg_1000';
    out_dir='/disk3/maheen_data/debugging_score_and_scale';
    util.mkdir(out_dir);
    num_to_pick=10;

    mat_overlaps = util.getFilesInFolder(dir_overlaps,'.npz');

    args=[];
    for idx_mat_overlap_file,mat_overlap_file in enumerate(mat_overlaps):
        args.append((mat_overlap_file,num_to_pick,idx_mat_overlap_file));

    p = multiprocessing.Pool(multiprocessing.cpu_count());
    pred_scores_all = p.map(loadAndPickN,args);
    
    pred_scores_all = np.vstack(pred_scores_all);
    
    best_image_all = pred_scores_all[:,1];

    out_hist = os.path.join(out_dir,'best_image_hist.png');
    visualize.hist(best_image_all,out_hist,bins=7,normed=True,xlabel='Value',ylabel='Frequency',title='Best Image Hist');
    print out_hist.replace('/disk3','vision3.cs.ucdavis.edu:1001');

def script_saveSegSavingInfoFiles():

    dir_overlaps = '/disk3/maheen_data/headC_160_noFlow_bbox/mat_overlaps_no_neg_1000';
    out_dir='/disk3/maheen_data/debugging_score_and_scale';
    img_dir_meta='/disk2/mayExperiments/validation/rescaled_images';
    out_dir_npy=os.path.join(out_dir,'npy_for_idx');
    out_file_test_pre=os.path.join(out_dir,'test_with_seg');
    # out_file_test_big=os.path.join(out_dir,'test_with_seg_big.txt');
    util.mkdir(out_dir_npy);
    num_to_pick=10;


    mat_overlaps = util.getFilesInFolder(dir_overlaps,'.npz');
    # mat_overlaps = mat_overlaps[:10];

    args=[];
    for idx_mat_overlap_file,mat_overlap_file in enumerate(mat_overlaps):
        args.append((mat_overlap_file,num_to_pick,idx_mat_overlap_file));

    
    p = multiprocessing.Pool(multiprocessing.cpu_count());
    pred_scores_all = p.map(loadAndPickN,args);
    print len(args);


    lines_to_write={};
    # lines_to_write_big=[];
    img_names=util.getFileNames(mat_overlaps,ext=False);
    for img_name,pred_scores in zip(img_names,pred_scores_all):
        img_num_uni=np.unique(pred_scores[:,1]);
        for img_num in img_num_uni:
            img_num=int(img_num);
            curr_im=os.path.join(img_dir_meta,str(img_num),img_name+'.jpg');
            # print curr_im
            assert os.path.exists(curr_im);
            out_dir_npy_curr = os.path.join(out_dir_npy,str(img_num));
            util.mkdir(out_dir_npy_curr);
            out_file = os.path.join(out_dir_npy_curr,img_name+'.npy');
            pred_scores_rel = pred_scores[pred_scores[:,1]==img_num,:];

            np.save(out_file,pred_scores_rel);

            if img_num in lines_to_write:
                lines_to_write[img_num].append(curr_im+' '+out_file);
            else:
                lines_to_write[img_num]=[curr_im+' '+out_file];
            

    
    for img_num in lines_to_write.keys():
        out_file_test=out_file_test_pre+'_'+str(img_num)+'.txt';
        print out_file_test,len(lines_to_write[img_num]);
        util.writeFile(out_file_test,lines_to_write[img_num]);


    


def main():

    # script_saveSegSavingInfoFiles()
    out_dir='/disk3/maheen_data/debugging_score_and_scale';
    out_dir_seg=os.path.join(out_dir,'seg');
    out_file_sh = os.path.join(out_dir,'test_seg_big.sh');
    path_to_test_file='/home/maheenrashid/Downloads/deep_proposals/torch_new/test_command_line_bigIm.th';
    limit=-1;
    gpu=1;
    model='/disk2/aprilExperiments/headC_160/noFlow_gaussian_all_actual/intermediate_resume_2/model_all_70000.dat';
    util.mkdir(out_dir_seg);

    out_file_test_pre=os.path.join(out_dir,'test_with_seg');
    
    out_files = [out_file_test_pre+'_'+str(num)+'.txt' for num in range(5,7)];
    out_dirs_seg = [os.path.join(out_dir_seg,str(num)) for num in range(5,7)]
    
    [util.mkdir(out_dir_curr) for out_dir_curr in out_dirs_seg]
    
    psr.script_writeTestSH(out_files,out_dirs_seg,out_file_sh,path_to_test_file,model,limit,gpu,overwrite=False,saveSeg=True)

    return
    scratch_dir='/disk3/maheen_data/scratch_dir'
    util.mkdir(scratch_dir);
    out_file=os.path.join(scratch_dir,'us_neg_box.p');

    to_explore=pickle.load(open(out_file,'rb'));

    for file_name,pred_boxes in to_explore:
        print file_name
        # print pred_boxes.shape
        meta_data=np.load(file_name);
        print meta_data.keys();
        mat_overlap=meta_data['mat_overlap'];
        pred_boxes=meta_data['pred_boxes'];
        gt_boxes_size=meta_data['gt_boxes_size'];

        print mat_overlap.shape,pred_boxes.shape,gt_boxes_size.shape


        for idx,pred_box in enumerate(pred_boxes):
            if np.min(pred_box)<0:
                print idx,pred_box,mat_overlap[idx];

        raw_input();
    

    
    

if __name__=='__main__':
    main();