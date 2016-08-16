import numpy as np;
import json;
import os
# import util
import json;
import data_preprocessing as dp;
import sys
sys.path.append('/home/maheenrashid/Downloads/debugging_jacob/python/')
import processOutput as po;
import util;
import visualize;
import scipy.misc;
import multiprocessing;
import random;

def getSubsetByCategory(anno,category_id,im_pre,im_post='.png',mask_post='_mask.png'):
    args=[];

    tuples_by_class=[];

    for idx_curr_anno,curr_anno in enumerate(anno):
        if idx_curr_anno%100==0:
            print idx_curr_anno;

        if curr_anno['iscrowd']!=0:
            continue
        
        if curr_anno['category_id']!=category_id:
            continue;

        id_no=curr_anno['image_id'];
        im_path=dp.addLeadingZeros(id_no,im_pre);

        out_file=im_path+'_'+str(idx_curr_anno);
        out_file_im=out_file+im_post;
        out_file_mask=out_file+mask_post;
        tuples_by_class.append((out_file_im,out_file_mask));

    return tuples_by_class

def script_saveBboxFiles(anno,out_dir,im_pre,idx_all,cat_id):
    
    bbox_dict={};

    idx=0;
    for idx in idx_all:
        if idx%100==0:
            print idx;
        # print anno[idx]
        assert anno[idx]['iscrowd']==0;
        assert anno[idx]['category_id']==cat_id;
    
        im_id=anno[idx]['image_id'];
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



def script_saveBboxFilesByImId(anno,out_dir,im_pre,img_idx_list,cat_id):
    
    bbox_dict={};

    idx_all=[];
    for idx in range(len(anno)):
        if anno[idx]['iscrowd']==0:
            cat_id_curr = anno[idx]['category_id'];
            im_id = anno[idx]['image_id'];
            if cat_id == cat_id_curr and im_id in img_idx_list:
                idx_all.append(idx);


    idx=0;
    for idx in idx_all:
        if idx%100==0:
            print idx;
        # print anno[idx]
        assert anno[idx]['iscrowd']==0;
        assert anno[idx]['category_id']==cat_id;
    
        im_id=anno[idx]['image_id'];
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



def script_saveHumanOnlyPos():
    path_to_anno='/disk2/ms_coco/annotations';    
    anno_file='instances_train2014.json';
    anno=json.load(open(os.path.join(path_to_anno,anno_file),'rb'))['annotations'];
    category_id=1; 
    im_pre='/disk2/aprilExperiments/positives_160/COCO_train2014_';
    out_file='/disk2/aprilExperiments/positives_160_human.txt'

    tuples=getSubsetByCategory(anno,category_id,im_pre)

    lines=[];
    for idx_line,(img_file,mask_file) in enumerate(tuples):
        if idx_line%100==0:
            print idx_line;
        
        if os.path.exists(img_file) and os.path.exists(mask_file):
            lines.append(img_file+' '+mask_file);
        
    print len(tuples),len(lines);

    util.writeFile(out_file,lines);


def script_saveHumanOnlyNeg():
    out_file='/disk2/aprilExperiments/positives_160_human.txt'
    out_dir='/disk2/aprilExperiments/negatives_npy_onlyHuman';
    util.mkdir(out_dir);
    im_pre='COCO_train2014_'

    lines=util.readLinesFromFile(out_file);
    img_files=[line[:line.index(' ')] for line in lines];

    img_names=util.getFileNames(img_files,ext=False);
    img_name=img_names[0];
    
    print img_name
    
    img_name_split=img_name.split('_');
    idx_all=[int(img_name.split('_')[-1]) for img_name in img_names];

    print len(img_names),len(idx_all),idx_all[0];
    cat_id=1;

    path_to_anno='/disk2/ms_coco/annotations';
    anno_file='instances_train2014.json';
    anno=json.load(open(os.path.join(path_to_anno,anno_file),'rb'))['annotations'];
    
    script_saveBboxFiles(anno,out_dir,im_pre,idx_all,cat_id)

def script_saveNonHumanOnlyNeg():
    neg_file='/disk2/marchExperiments/deep_proposals/negatives.txt';
    out_dir='/disk2/aprilExperiments/negatives_npy_onlyHuman'

    lines=util.readLinesFromFile(neg_file);
    npy_files=[line[line.index(' ')+1:] for line in lines];
    npy_file_names=util.getFileNames(npy_files);

    exists=0;
    for idx_npy_file_name,npy_file_name in enumerate(npy_file_names):
        if idx_npy_file_name%100==0:
            print idx_npy_file_name;

        file_curr=os.path.join(out_dir,npy_file_name);
        if os.path.exists(file_curr):
            exists+=1;
        else:
            zeros=np.zeros((0,4));
            np.save(file_curr,zeros);

    print exists,len(npy_file_names);
    
def script_writeHumanOnlyNegFile():
    neg_file_old='/disk2/marchExperiments/deep_proposals/negatives.txt'
    neg_file_new='/disk2/marchExperiments/deep_proposals/negatives_onlyHuman.txt'

    npy_dir_old='/disk2/marchExperiments/deep_proposals/negatives'
    npy_dir_new='/disk2/aprilExperiments/negatives_npy_onlyHuman'

    lines=util.readLinesFromFile(neg_file_old);
    lines_new=[line.replace(npy_dir_old,npy_dir_new) for line in lines];
    for line in lines_new:
        assert npy_dir_new in line;

    print len(lines),len(lines_new);
    print lines_new[0];
    util.writeFile(neg_file_new,lines_new);

def makeFloVizHTML(out_file_html,img_paths,dir_flo_viz):
    # out_file_html=os.path.join(out_dir,'flo_viz.html');
    img_paths_html=[];
    captions_html=[];
    for img_path,img_file_name in zip(img_paths,util.getFileNames(img_paths,ext=False)):
        out_file_flo_viz=os.path.join(dir_flo_viz,img_file_name+'.png');
        if img_path.startswith('/disk2'):
            img_path='/disk3'+img_path;

        img_paths_curr=[util.getRelPath(img_path,'/disk3'),util.getRelPath(out_file_flo_viz,'/disk3')];
        img_paths_html.append(img_paths_curr);
        captions_html.append([img_file_name,'flo']);

    visualize.writeHTML(out_file_html,img_paths_html,captions_html);


def script_saveFloAndVizFromTestFile(neg_file,out_dir,gpu=0):
    pos_lines=util.readLinesFromFile(neg_file);
    pos_img_files=[line[:line.index(' ')] for line in pos_lines];
    
    print len(pos_img_files);
    print pos_img_files[0];

    gpu=1;
    model_file='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction_test/examples/opticalflow/final.caffemodel'
    clusters_file='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction_test/examples/opticalflow/clusters.mat';

    dir_flo=os.path.join(out_dir,'flo');
    util.mkdir(dir_flo);
    dir_flo_viz=os.path.join(out_dir,'flo_viz');
    util.mkdir(dir_flo_viz);

    po.script_saveFlosAndViz(pos_img_files,dir_flo,dir_flo_viz,gpu,model_file,clusters_file,overwrite=True);


def saveFlowImage((flo_file,out_file,idx)):
    print idx,out_file
    if os.path.exists(out_file):
        print 'SKIP'
        return;

    if not os.path.exists(flo_file):
        print 'SKIP NO FLO',flo_file
        return;

    flo = util.readFlowFile(flo_file)
    
    mag=np.power(np.power(flo[:,:,0],2)+np.power(flo[:,:,1],2),0.5)
    
    flo_final=np.dstack((flo,mag));
    for dim in range(flo_final.shape[2]):
        flo_curr=flo_final[:,:,dim]
        min_val=np.min(flo_curr);
        flo_curr = flo_curr-min_val;
        max_val=np.max(flo_curr);
        scale_factor=255.0/max_val;
        flo_curr=flo_curr*scale_factor;
        flo_final[:,:,dim]=flo_curr;

    # for dim in range(flo_final.shape[2]):
    #     print dim,np.min(flo_final[:,:,dim]),np.max(flo_final[:,:,dim]),np.mean(flo_final[:,:,dim])

    scipy.misc.imsave(out_file,flo_final);



def script_saveFlowIm():
    pos_file='/disk2/aprilExperiments/positives_160_human.txt';
    # out_dir='/disk3/maheen_data/headC_160/noFlow_gaussian_human/pos_flos'

    out_dir='/disk3/maheen_data/headC_160/neg_flos'
    out_dir_flo_im=os.path.join(out_dir,'flo_im');
    util.mkdir(out_dir_flo_im);

    match_info_file=os.path.join(out_dir,'flo','match_info.txt');
    flo_dir=os.path.join(out_dir,'flo','flo_files');

    h5_files,img_files,img_sizes=po.parseInfoFile(match_info_file,lim=None)
    print len(img_files);
    flo_files=[os.path.join(flo_dir,file_name+'.flo') for file_name in util.getFileNames(img_files,ext=False)];
    out_files=[os.path.join(out_dir_flo_im,file_name+'.png') for file_name in util.getFileNames(img_files,ext=False)];

    args=[];
    for idx,(flo_file,out_file) in enumerate(zip(flo_files,out_files)):
        args.append((flo_file,out_file,idx))
    
    p=multiprocessing.Pool(multiprocessing.cpu_count())
    p.map(saveFlowImage,args)
    
def writeTrainFilesWithFlow(old_train_file,dir_flo_im,new_train_file,ext='.png'):
    lines=util.readLinesFromFile(old_train_file);
    img_files=[line[:line.index(' ')] for line in lines];
    file_names=util.getFileNames(img_files,ext=False);
    flo_im_files=[os.path.join(dir_flo_im,file_name+ext) for file_name in file_names];
    for flo_im_file in flo_im_files:
        assert os.path.exists(flo_im_file);

    lines_new=[line+' '+flo_im_curr for line,flo_im_curr in zip(lines,flo_im_files)];
    util.writeFile(new_train_file,lines_new);


def getMeanRGB((flo_path,idx)):
    print idx
    flo=scipy.misc.imread(flo_path);
    flo=np.reshape(flo,(flo.shape[0]*flo.shape[1],flo.shape[2]))
    means=np.mean(flo,axis=0);
    return means


def main():



    # out_file='/disk2/aprilExperiments/positives_160_human.txt'
    # out_dir='/disk2/aprilExperiments/negatives_npy_onlyHuman';
    # util.mkdir(out_dir);
    im_pre='COCO_val2014_'

    # lines=util.readLinesFromFile(out_file);
    # img_files=[line[:line.index(' ')] for line in lines];

    anno_dir='/disk3/maheen_data/headC_160_noFlow_justHuman/im/0';
    out_anno_dir='/disk3/maheen_data/val_anno_human_only_300';
    util.mkdir(out_anno_dir);
    img_files=util.getFilesInFolder(anno_dir,'.jpg');
    img_names=util.getFileNames(img_files,ext=False);
    
    # img_name_split=img_name.split('_');
    idx_all=[int(img_name.split('_')[-1]) for img_name in img_names];

    print len(img_names),len(idx_all),idx_all[0];
    cat_id=1;

    path_to_anno='/disk2/ms_coco/annotations';
    anno_file='instances_val2014.json';
    anno=json.load(open(os.path.join(path_to_anno,anno_file),'rb'))['annotations'];
    
    # script_saveBboxFiles(anno,out_anno_dir,im_pre,idx_all,cat_id)
    script_saveBboxFilesByImId(anno,out_anno_dir,im_pre,idx_all,cat_id)


    return
    # person 1
    
    # neg_file='/disk2/marchExperiments/deep_proposals/negatives_onlyHuman.txt'
    out_dir='/disk3/maheen_data/headC_160/neg_flos'
    # dir_flo_im=os.path.join(out_dir,'flo_im');
    # new_neg_file=os.path.join(out_dir,'negatives_onlyHuman_withFlow.txt');

    # neg_file='/disk2/aprilExperiments/positives_160_human.txt'
    # out_dir='/disk3/maheen_data/headC_160/noFlow_gaussian_human/pos_flos'
    dir_flo_im=os.path.join(out_dir,'flo_im');
    # new_neg_file=os.path.join(out_dir,'positives_onlyHuman_withFlow.txt');
    
    out_file_means=os.path.join(out_dir,'means.npy');

    flo_files=util.getFilesInFolder(dir_flo_im,ext='.png');
    # flo_files=flo_files[:100];

    p=multiprocessing.Pool(multiprocessing.cpu_count());
    means_all=p.map(getMeanRGB,zip(flo_files,range(len(flo_files))));

    means_all=np.array(means_all);
    means=np.mean(means_all,axis=0);
    print means;
    np.save(out_file_means,means);


    # writeTrainFilesWithFlow(neg_file,dir_flo_im,new_neg_file)

    # lines=util.readLinesFromFile(new_neg_file);

    # for line in lines[:10]:
    #     print line;
    # print '___';
    # print len(lines);


    # 

    # neg_lines=util.readLinesFromFile(pos_file);
    # for i in range(10):
    #     print neg_lines[i];
    # print '___';
    # print len(neg_lines);


    return
    pos_file='/disk2/aprilExperiments/positives_160_human.txt';
    out_dir='/disk3/maheen_data/headC_160/noFlow_gaussian_human/pos_flos'
    gpu=1;
    util.mkdir(out_dir);

    # script_saveFloAndVizFromTestFile(pos_file,out_dir,gpu=gpu)
    
    neg_file='/disk2/marchExperiments/deep_proposals/negatives_onlyHuman.txt';
    out_dir='/disk3/maheen_data/headC_160/neg_flos'
    gpu=0;
    util.mkdir(out_dir);
    # script_saveFloAndVizFromTestFile(neg_file,out_dir,gpu=gpu)

    
    return;
    path_to_im='/disk2/ms_coco/train2014';
    im_pre='COCO_train2014_';

    path_to_anno='/disk2/ms_coco/annotations';
    # anno_file='train_subset_just_anno.json';
    anno_file='instances_train2014_except_anno.json';
    anno=json.load(open(os.path.join(path_to_anno,anno_file),'rb'));
    print anno.keys();
    print anno['categories']
    for anno_curr in anno['categories']:
        print anno_curr['id'], anno_curr['name']

    return
    preds_file='/disk2/januaryExperiments/pedro_data/coco-proposals/train.npy';
    preds=np.load(preds_file);
    print preds.shape;
    print np.min(preds[:,-2]),np.max(preds[:,-2])
    print np.unique(preds[:,0]).shape   
    print np.min(preds[np.where(preds[:,0]==preds[0,0]),-2]),np.max(preds[np.where(preds[:,0]==preds[0,0]),-2])


if __name__=='__main__':
    main();