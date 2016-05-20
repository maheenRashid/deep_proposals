import numpy as np;
import json;
import os
import util
import json;
import data_preprocessing as dp;


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
    


def main():
    # person 1
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