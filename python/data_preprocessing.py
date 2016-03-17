import scipy.misc
import numpy as np;
import json;
import os;
from PIL import Image, ImageDraw
import cPickle as pickle;
import matplotlib.pyplot as plt;
import multiprocessing;
import glob;
import util;

def addLeadingZeros(id_no,im_pre=None,im_post=None,num_digits_total=12,):
    if im_pre is None:
        im_pre='';
    if im_post is None:
        im_post='';

    num_dig_curr=str(id_no);
    num_dig_remain=num_digits_total-len(num_dig_curr);
    num_dig_pre='0'*num_dig_remain;
    num_dig_final=num_dig_pre+num_dig_curr;
    res=im_pre+num_dig_final+im_post;
    return res;

def getMask(seg,shape,dim=1):
    mask = Image.new('L', (shape[1],shape[0]),0)
    for seg_curr in seg:
        ImageDraw.Draw(mask).polygon(seg_curr, outline=1, fill=1)
    mask = np.array(mask)
    # mask[mask==0]=bg
    # mask[mask==1]=fill;
    if dim==3:
        mask=np.dstack((mask,mask,mask));
    return mask

def getBB(seg):
    if not hasattr(seg, '__iter__'):
        seg =[seg];
    seg_flat=[a for b in seg for a in b];
    x=seg_flat[::2];
    y=seg_flat[1::2];
    bb=[min(x),min(y),max(x),max(y)];
    return bb;

def getDimInfo(bb):
    # print 'bb',bb;
    diffs=[bb[dim+2]-bb[dim] for dim in range(2)]
    max_dim=np.argmax(diffs);
    return max_dim,diffs

def getRescaleSizeIm(bb,max_dim_size,im_dim):
    # print 'bb in rescale',bb;
    max_dim,diffs=getDimInfo(bb);
    assert diffs[0]>0 and diffs[1]>0;
    # print max_dim,diffs,bb
    ratio=float(max_dim_size)/diffs[max_dim];
    new_size=[int(im_dim[i]*ratio) for i in range(len(im_dim))];
    new_size_bb=[int(bb[i]*ratio) for i in range(len(bb))];
    return new_size,new_size_bb,ratio

def getCropCoord(new_size_bb,crop_bb_org,negOnly=False):
    if negOnly:
        new_size_bb=[val-min(crop_bb_org[idx%2],0) for idx,val in enumerate(new_size_bb)];
    else:
        new_size_bb=[val-crop_bb_org[idx%2] for idx,val in enumerate(new_size_bb)];
    return new_size_bb;

def showIm(im,ImageFlag=True):
    if ImageFlag:
        im=np.array(im);
    plt.ion();
    fig=plt.figure();
    plt.imshow(im);
    plt.show();

def scaleValues(seg,im_size,max_dim_size,total_size):
    # print seg
    bb=getBB(seg);
    # print 'bb in scaleValues',bb;
    # get im resize size, resized bb size, and ratio for rescaling
    new_size,new_size_bb,ratio=getRescaleSizeIm(bb,max_dim_size,im_size);
    #get max_dim, and width height of resized bb
    _,diffs_rs=getDimInfo(new_size_bb);
    #figure out how much buffer to add so that image is centered in total_size x total size crop
    bufs=[(total_size-diff)/2.0 for diff in diffs_rs];
    #get the cropping indices
    crop_bb_org=[int(val+bufs[idx%2]) if idx>1 else int(val-bufs[idx%2]) for idx,val in enumerate(new_size_bb)]
    #get padding required for correct crop
    pad_req=[-1*val if val<0 else max(val-new_size[idx%2],0) for idx,val in enumerate(crop_bb_org)]
    #get new crop,bb and seg values
    crop_bb=getCropCoord(crop_bb_org,crop_bb_org,negOnly=True)
    seg_new=[];
    for seg_curr in seg:
        seg_curr=[int(val*ratio) for val in seg_curr];
        seg_curr=getCropCoord(seg_curr,crop_bb_org,negOnly=True)
        seg_curr=getCropCoord(seg_curr,crop_bb)
        seg_new.append(seg_curr);

    return new_size,pad_req,seg_new,crop_bb
    
def saveFullImAndBBCanonical((im_path,seg,max_dim_size,total_size,out_file_im,out_file_bb)):
    print out_file_im
    # print seg;
    im=Image.open(im_path);
    
    bb=getBB(seg);
    try:
        new_size,new_size_bb,ratio=getRescaleSizeIm(bb,max_dim_size,list(im.size));
    except AssertionError:
        print 'assertion error'
        return;
    _,diffs_rs=getDimInfo(new_size_bb);
    #figure out how much buffer to add so that image is centered in total_size x total size crop
    bufs=[(total_size-diff)/2.0 for diff in diffs_rs];
    #get the cropping indices
    crop_bb_org=[int(val+bufs[idx%2]) if idx>1 else int(val-bufs[idx%2]) for idx,val in enumerate(new_size_bb)]
    
    # print crop_bb_org
    im=im.resize(new_size);

    crop_bb_org=[max(val,0) if idx<2 else min(list(im.size)[idx-2],val) for idx,val in enumerate(crop_bb_org)]

    #resize
    im.save(out_file_im);
    np.save(open(out_file_bb,'wb'),crop_bb_org);
    # return seg;

    # return im,crop_bb_org;    

def saveMaskAndCrop((im_path,seg,max_dim_size,total_size,out_file_im,out_file_mask)):
    print out_file_im
    # print seg;
    im=Image.open(im_path);
    try:
        new_size,pad_req,seg,crop_bb=scaleValues(seg,list(im.size),max_dim_size,total_size);
    except AssertionError:
        print 'assertion error'
        return;
    #resize
    im=im.resize(new_size);
    #pad it
    im_np=np.array(im);
    if sum(pad_req)!=0:
        to_pad=[(pad_req[1],pad_req[3]),(pad_req[0],pad_req[2])]
        if len(im_np.shape)>2:
            to_pad.append((0,0));
        im_np=np.pad(im_np,tuple(to_pad),'edge');
    #crop it
    # print im_np.shape,crop_bb,im.size
    if len(im_np.shape)>2:
        im_np=im_np[crop_bb[1]:crop_bb[3],crop_bb[0]:crop_bb[2],:]
    else:
        im_np=im_np[crop_bb[1]:crop_bb[3],crop_bb[0]:crop_bb[2]]
    #get mask
    mask=getMask(seg,im_np.shape,dim=1)
    #save
    scipy.misc.imsave(open(out_file_mask,'wb'),mask);

    scipy.misc.imsave(open(out_file_im,'wb'),im_np);
    # return seg;


def genArgs(anno,im_pre,out_dir,max_dim_size,total_size,im_post='.png',mask_post='_mask.png'):
    args=[];
    for idx_curr_anno,curr_anno in enumerate(anno):
        if curr_anno['iscrowd']==0:
            id_no=curr_anno['image_id'];
            seg_all=curr_anno['segmentation'];
            im_path=addLeadingZeros(id_no,im_pre,'.jpg');
            # print curr_anno;

            # for idx_seg,seg in enumerate(seg_all):
            out_file=os.path.join(out_dir,im_path[im_path.rindex('/')+1:im_path.rindex('.')]+'_'+str(idx_curr_anno));
            out_file_im=out_file+im_post;
            out_file_mask=out_file+mask_post;

            if not os.path.exists(out_file_im):
                # print 'yes';
                # print idx_curr_anno,curr_anno
                args.append((im_path,seg_all,max_dim_size,total_size,out_file_im,out_file_mask))
    return args;

def script_saveMaskBig(anno,im_pre,out_dir):
    for idx_curr_anno,curr_anno in enumerate(anno):
        if curr_anno['iscrowd']==0:
            
            id_no=curr_anno['image_id'];
            seg_all=curr_anno['segmentation'];
            
            im_path=addLeadingZeros(id_no,im_pre,'.jpg');

            out_file=os.path.join(out_dir,im_path[im_path.rindex('/')+1:im_path.rindex('.')]+'_'+str(idx_curr_anno));
            out_file=out_file+'_mask.png';  
            print idx_curr_anno,len(anno),out_file;
            
            if not os.path.exists(out_file):
                im=scipy.misc.imread(open(im_path,'rb'));
                mask=getMask(seg_all,im.shape,dim=1);
                print mask.shape,im.shape
            # 
                scipy.misc.imsave(open(out_file,'wb'),mask);
            # raw_input();


def writeDirContentsToFile(dir,out_file,postfix):
    with open(out_file,'wb') as f:
        for file_curr in glob.iglob(dir+'/*'+postfix):
            print file_curr
            # if file_curr.endswith(postfix):
                # print file_curr
            f.write(os.path.join(dir,file_curr)+'\n');

def script_writeUCFOnlyFile():

    path_to_UCF='/disk2/video_data/UCF-101';
    path_to_text='/disk2/februaryExperiments/training_jacob/caffe_files/train.txt';
    path_to_text_ucf='/disk2/februaryExperiments/training_jacob/caffe_files/train_ucf.txt';
    start_string='/disk2/februaryExperiments/training_jacob/training_data/v_';
    
    # lines=np.array(lines);

    dirs=[os.path.join(path_to_UCF,dir_curr) for dir_curr in os.listdir(path_to_UCF)]
    dirs_ac=[dir_curr for dir_curr_meta in dirs for dir_curr in os.listdir(dir_curr_meta) if dir_curr.endswith('.avi')];
    print dirs
    print len(dirs_ac);
    print dirs_ac[:10];


    with open(path_to_text,'rb') as f:
        lines=f.readlines();
    rel_lines=[line for line in lines if line.startswith(start_string)];

    with open(path_to_text_ucf,'wb') as f:
        for line in rel_lines:
            f.write(line);

def writeTrainingDataFiles(dir_content_file,pre_dir,img_dir,out_file_text,ignore_amount=-2,postfix='.jpg'):
    start_idx=len(pre_dir);
    files=util.readLinesFromFile(dir_content_file);
    lines_to_write=[];

    for idx_file_curr,file_curr in enumerate(files):
        if idx_file_curr%1000==0:
            print idx_file_curr
        file_name=file_curr[start_idx+1:];
        file_name=file_name.split('_');
        file_name='_'.join(file_name[:ignore_amount]);
        file_name=file_name+postfix;
        file_name=os.path.join(img_dir,file_name);
        lines_to_write.append(file_name+' '+file_curr);
    util.writeFile(out_file_text,lines_to_write);

def script_saveBboxFiles(anno,out_dir,im_pre):
    
    bbox_dict={};

    idx=0;
    for idx in range(len(anno)):
        if idx%100==0:
            print idx;

        if anno[idx]['iscrowd']==0:
            im_id=anno[idx]['image_id'];
            bbox=anno[idx]['bbox'];
            
            im_path=addLeadingZeros(im_id,os.path.join(out_dir,im_pre),'.npy');
            if im_path in bbox_dict:
                # print im_path,bbox_dict[im_path],
                bbox_dict[im_path].append(bbox);
                # print bbox_dict[im_path];
            else:
                bbox_dict[im_path]=[bbox];

    for im_path in bbox_dict.keys():
        bbox_curr=bbox_dict[im_path];
        bbox_curr=np.array(bbox_curr);
        # print im_path
        np.save(im_path,bbox_curr);


def main():
    path_to_im='/disk2/ms_coco/train2014'
    path_to_anno='/disk2/ms_coco/annotations';
    out_dir='/disk2/marchExperiments/deep_proposals/negatives';
    path_to_im='/disk2/ms_coco/train2014';
    im_pre='COCO_train2014_';

    out_file='/disk2/marchExperiments/deep_proposals/negatives.txt';
    lines=[];
    for file_curr in os.listdir(out_dir):
        if file_curr.endswith('.npy'):
            file_name=file_curr[:file_curr.rindex('.')];
            file_im=os.path.join(path_to_im,file_name+'.jpg');
            file_npy=os.path.join(out_dir,file_curr);
            lines.append(file_im+' '+file_npy);

    print len(lines);
    # print lines[:100]
    util.writeFile(out_file,lines);



    return
    path_to_im='/disk2/ms_coco/train2014';
    im_pre='COCO_train2014_';

    path_to_anno='/disk2/ms_coco/annotations';
    train_anno=os.path.join(path_to_anno,'instances_train2014.json');

    out_dir='/disk2/februaryExperiments/deep_proposals/im_resize';
    if not os.path.exists(out_dir):
        os.mkdir(out_dir);

    max_dim_size=128;
    total_size=240;
    im_post='.jpg';
    mask_post='.npy';
    
    anno=json.load(open(train_anno,'rb'))['annotations']
    # anno=anno[:1000];
    # for arg in genArgs(anno,os.path.join(path_to_im,im_pre),out_dir,max_dim_size,total_size,im_post,mask_post):
    #     saveFullImAndBBCanonical(arg);

    args=genArgs(anno,os.path.join(path_to_im,im_pre),out_dir,max_dim_size,total_size,im_post,mask_post)
    p=multiprocessing.Pool();
    p.map(saveFullImAndBBCanonical,args)
    
    # out_file='/disk2/februaryExperiments/deep_proposals/contents_im_resize.txt';
    # postfix='.jpg';
    # writeDirContentsToFile(out_dir,out_file,postfix);
    

    return

    
    
    
    script_saveMaskBig(anno,os.path.join(path_to_im,im_pre),out_dir);

    return
    print len(anno)
    p=multiprocessing.Pool();
    p.map(saveMaskAndCrop,genArgs(anno,os.path.join(path_to_im,im_pre),out_dir,max_dim_size,total_size))
    
    # script_saveMaskBig(
    return
    # im_path='temp/im.jpg';
    # seg_path='temp/seg.p';
    # out_file_im='temp/im_crop.png';
    # out_file_mask='temp/im_crop_mask.png';

    # max_dim_size=128;
    # total_size=240;
    # seg=pickle.load(open(seg_path,'rb'));
    # # saveMaskAndCrop((im_path,seg,max_dim_size,total_size,out_file_im,out_file_mask));
    # im=scipy.misc.imread(out_file_im);
    # mask=scipy.misc.imread(out_file_mask);
    # showIm(im,False);
    # showIm(mask,False);
    # print np.min(im),np.max(im);
    # print np.min(mask),np.max(mask);
    # raw_input();
    # # return
    

    # return
    path_to_im='/disk2/ms_coco/train2014';
    im_pre='COCO_train2014_';
    path_to_prop_data='/disk2/januaryExperiments/pedro_data/coco-proposals';
    train_prop=os.path.join(path_to_prop_data,'train.npy');
    train_prop=os.path.join(path_to_prop_data,'train_subset.npy');

    path_to_anno='/disk2/ms_coco/annotations';
    train_anno=os.path.join(path_to_anno,'instances_train2014.json');
    train_anno=os.path.join(path_to_anno,'train_subset_just_anno.json');

    out_dir='/disk2/temp';

    #read json
    data=json.load(open(train_anno,'rb'))
    curr_anno=data['annotations'][0];
    id_no=curr_anno['image_id'];
    seg=curr_anno['segmentation'];

    #print image path
    im_path=addLeadingZeros(id_no,im_pre,'.jpg');
    print im_path;

    #save seg
    seg=seg[0];
    pickle.dump(seg,open(os.path.join(out_dir,'seg.p'),'wb'));


    
    
if __name__=='__main__':
    main();