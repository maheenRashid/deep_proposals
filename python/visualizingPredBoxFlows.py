import util;
import numpy as np;
import processScoreResults as psr;
import scipy.misc;
import os;
import scipy.special;
import cv2;
import visualize;
from collections import namedtuple
import multiprocessing;
import cPickle as pickle;

def createParams(type_Experiment):
    if type_Experiment == 'saveSegVizAll':
        list_params = ['path_to_im_meta',
                        'path_to_im_canon',
                        'path_to_predScores_meta',
                        'path_to_seg_meta',
                        'path_to_mat_overlap',
                        'path_to_score_mat_meta',
                        'img_names_txt',
                        'scale_idx_range',
                        'stride',
                        'w',
                        'alpha',
                        'num_to_pick',
                        'out_dir_overlay_meta',
                        'power_scale_range',
                        'power_step_size',
                        'out_file_html',
                        'height_width',
                        'num_threads',
                        'out_dir_im_dict',
                        'overwrite'];
                        
        params = namedtuple('Params_saveSegVizAll',list_params);
    elif type_Experiment == 'saveFloVizAll':
        list_params = ['mat_overlap_dir',
                        'gt_dir',
                        'thresh_overlap',
                        'flo_viz_dir',
                        'results_dir_meta',
                        'path_to_im_canon',
                        'out_dir_overlay_flo',
                        'gt_color',
                        'pred_color',
                        'alpha_overlay',
                        'w',
                        'stride',
                        'power_scale_range',
                        'power_step_size',
                        'num_threads'];
        params = namedtuple('Params_saveFloVizAll',list_params);               
    else:
        params = None

    return params;


def getPredScoresDict(path_to_predScores_meta,path_to_score_mat_meta,img_name,scale_idx_range):
    pred_scores_dict={};
    for scale_idx_curr in scale_idx_range:
        pred_scores_file_curr=os.path.join(path_to_predScores_meta,str(scale_idx_curr),img_name+'.npy');
        if os.path.exists(pred_scores_file_curr):
            pred_score_mat_file_curr=os.path.join(path_to_score_mat_meta,str(scale_idx_curr),img_name+'.npy');
            assert os.path.exists(pred_score_mat_file_curr);
            pred_score_mat=np.load(pred_score_mat_file_curr);
            pred_scores_dict[scale_idx_curr]=(np.load(pred_scores_file_curr),pred_score_mat[0][0].shape);
    return pred_scores_dict

def getBBoxBigDict(pred_scores_dict,im_size,scales,stride,w):
    bboxes_dict={};
    for curr_key in pred_scores_dict.keys():    
        pred_scores_dict_curr=pred_scores_dict[curr_key];
        scale_curr= scales[curr_key];
        bbox_all=[];
        for curr_row in pred_scores_dict_curr[0]:
            row=curr_row[2];
            col=curr_row[3];
            bbox_containing=psr.getBoxContainingCanonical(row,col,im_size,scale_curr,pred_scores_dict_curr[1],stride,w)
            bbox_containing=[int(val) for val in bbox_containing];
            bbox_all.append(bbox_containing);
        bbox_all=np.vstack(bbox_all);
        bboxes_dict[curr_key]=bbox_all;
    return bboxes_dict;

def sigmoidMask(mask):
    mask_sig=scipy.special.expit(mask);
    return mask_sig;

def getSegsDict(bboxes_dict,path_to_seg_meta,img_name):
    segs_dict={};
    for curr_key in bboxes_dict.keys():
        bbox_rel=bboxes_dict[curr_key];
        seg_path=os.path.join(path_to_seg_meta,str(curr_key),img_name+'_seg.npy');
        segs_rel=np.load(seg_path);
        # segs_rel = sigmoidMask(segs_rel);
        bbox_curr=bbox_rel[0];
        size_r=bbox_curr[2]-bbox_curr[0];
        size_c=bbox_curr[3]-bbox_curr[1];
        segs_all=[];
        for idx in range(segs_rel.shape[0]):
            seg_curr=segs_rel[idx];
            seg_rel = cv2.resize(seg_curr, (size_c,size_r))
            seg_rel=np.expand_dims(seg_rel,axis=0);
            segs_all.append(seg_rel);
            
        segs_all=np.concatenate(segs_all,axis=0);
        segs_dict[curr_key]=segs_all;
    return segs_dict

def getSegOverlay(im,box_curr,seg_curr,alpha=0.5):
    to_crop=[0-box_curr[0],0-box_curr[1],box_curr[2]-im.shape[0],box_curr[3]-im.shape[1]];
    to_crop=[max(val,0) for val in to_crop];
    box_start=[max(0,box_curr[0]),max(0,box_curr[1])];
    seg_crop=seg_curr[to_crop[0]:seg_curr.shape[0]-to_crop[2],to_crop[1]:seg_curr.shape[1]-to_crop[3]];
    heatmap=visualize.getHeatMap(seg_crop);
    if len(im.shape)<3:
        im=np.dstack((im,im,im));
    heatmap_big=128*np.ones((im.shape))
    # print heatmap_big.shape,im.shape,heatmap.shape,box_start
    box_end=[min(heatmap_big.shape[0],box_start[0]+heatmap.shape[0]),min(heatmap_big.shape[1],box_start[1]+heatmap.shape[1])]
    heatmap_big[box_start[0]:box_end[0],box_start[1]:box_end[1],:]=heatmap[:box_end[0]-box_start[0],:box_end[1]-box_start[1],:]
    if heatmap_big.shape!=im.shape:
        # print 'resizing',heatmap_big.shape,im.shape,
        heatmap_big=cv2.imresize(heatmap_big,(im.shape[1],im.shape[0]));
        # print heatmap_big.shape
    img_fuse=visualize.fuseAndSave(im,heatmap_big,alpha);
    return img_fuse;

def getPredBoxesDict(mat_overlap_file,num_to_pick):
    mat_overlap = np.load(mat_overlap_file);
    pred_boxes = mat_overlap['pred_boxes'];
    pred_scores = mat_overlap['pred_scores'];
    pred_boxes = pred_boxes[:num_to_pick,:];
    pred_scores= pred_scores[:num_to_pick,:];

    scale_uni=np.unique(pred_scores[:,1]);
    pred_boxes_dict={};
    for scale_idx in scale_uni:
        pred_boxes_rel=pred_boxes[pred_scores[:,1]==scale_idx,:];
        pred_boxes_dict[scale_idx]=pred_boxes_rel;
    return pred_boxes_dict


def saveSegViz(im,alpha,img_name,out_dir_overlay_meta,pred_scores_dict,pred_boxes_dict,bboxes_dict,segs_dict,overwrite):
    img_list_all = [];
    for curr_key in bboxes_dict:
        bboxes_curr=bboxes_dict[curr_key];
        segs_curr=segs_dict[curr_key];
        bboxes_pred_curr = pred_boxes_dict[curr_key];
        pred_scores_curr = pred_scores_dict[curr_key];
        pred_scores_curr = pred_scores_curr[0][:,0]
        out_dir_overlay_curr = os.path.join(out_dir_overlay_meta,str(curr_key));
        for idx in range(bboxes_curr.shape[0]):
            out_file=os.path.join(out_dir_overlay_curr,img_name+'_'+str(idx)+'.png');
            if not os.path.exists(out_file) or overwrite:
                box_curr = bboxes_curr[idx];
                seg_curr = segs_curr[idx];
                box_pred_curr = bboxes_pred_curr[idx];
                pred_score_curr = pred_scores_curr[idx];
                im_overlay_seg=getSegOverlay(im,box_curr,seg_curr,alpha);
                visualize.plotBBox(im_overlay_seg,[box_pred_curr],out_file,labels=[pred_score_curr]);
            img_list_all.append(out_file);

    return img_list_all;

def getFloOverlay(im,flos_boxes,alpha=0.5):

    if len(im.shape)<3:
        im=np.dstack((im,im,im));
    
    heatmap_big=np.zeros((im.shape))
    heatmap_count=np.zeros((im.shape[0],im.shape[1]));

    for flo_curr,box_curr in flos_boxes:
        box_size=[box_curr[2]-box_curr[0],box_curr[3]-box_curr[1]];
        flo_curr=cv2.resize(flo_curr,(box_size[1],box_size[0]));
        to_crop=[0-box_curr[0],0-box_curr[1],box_curr[2]-im.shape[0],box_curr[3]-im.shape[1]];
        to_crop=[max(val,0) for val in to_crop];
        
        flo_crop=flo_curr[to_crop[0]:flo_curr.shape[0]-to_crop[2],to_crop[1]:flo_curr.shape[1]-to_crop[3]];
        flo_crop=flo_crop[:,:,2];

        heatmap_curr=visualize.getHeatMap(flo_crop);

        box_start=[max(0,box_curr[0]),max(0,box_curr[1])];
        box_end=[min(heatmap_big.shape[0],box_start[0]+heatmap_curr.shape[0]),min(heatmap_big.shape[1],box_start[1]+heatmap_curr.shape[1])]        
        
        heatmap_big[box_start[0]:box_end[0],box_start[1]:box_end[1],:]=heatmap_big[box_start[0]:box_end[0],box_start[1]:box_end[1],:]+heatmap_curr[:box_end[0]-box_start[0],:box_end[1]-box_start[1],:];
        heatmap_count[box_start[0]:box_end[0],box_start[1]:box_end[1]]=heatmap_count[box_start[0]:box_end[0],box_start[1]:box_end[1]]+1;
    
    heatmap_big[heatmap_count==0,:]=128;
    heatmap_count[heatmap_count==0]=1;
    heatmap_big=heatmap_big/np.dstack((heatmap_count,heatmap_count,heatmap_count));
    img_fuse=visualize.fuseAndSave(im,heatmap_big,alpha);
    return img_fuse;


def saveImDict(img_name,im_size,path_to_predScores_meta,path_to_seg_meta,mat_overlap_file,
            path_to_score_mat_meta,scale_idx_range,stride,w,alpha,num_to_pick,scales,out_file):
    
    pred_scores_dict=getPredScoresDict(path_to_predScores_meta,path_to_score_mat_meta,img_name,scale_idx_range);
    pred_boxes_dict = getPredBoxesDict(mat_overlap_file,num_to_pick);
    bboxes_dict=getBBoxBigDict(pred_scores_dict,im_size,scales,stride,w);
    segs_dict=getSegsDict(bboxes_dict,path_to_seg_meta,img_name);
    pickle.dump([pred_scores_dict,pred_boxes_dict,bboxes_dict,segs_dict],open(out_file,'wb'));
    return [pred_scores_dict,pred_boxes_dict,bboxes_dict,segs_dict]
    

def saveSegVizImage((img_name,path_to_im_meta,path_to_im_canon,path_to_predScores_meta,path_to_seg_meta,path_to_mat_overlap,
            path_to_score_mat_meta,scale_idx_range,stride,w,alpha,num_to_pick,out_dir_overlay_meta,power_scale_range,power_step_size,out_dir_im_dict,overwrite,idx_img_name)):
    
    print idx_img_name
    
    power_range=np.arange(power_scale_range[0],power_scale_range[1]+power_step_size,power_step_size);
    scales=[2**val for val in power_range];

    im_path=os.path.join(path_to_im_canon,img_name+'.jpg');
    im=scipy.misc.imread(im_path);
    im_size=im.shape;
    mat_overlap_file = os.path.join(path_to_mat_overlap,img_name+'.npz');
    out_file_dict=os.path.join(out_dir_im_dict,img_name+'.p');

    if overwrite or not os.path.exists(out_file_dict):
        dict_list=saveImDict(img_name,im_size,path_to_predScores_meta,path_to_seg_meta,mat_overlap_file,
            path_to_score_mat_meta,scale_idx_range,stride,w,alpha,num_to_pick,scales,out_file_dict);
    else:
        dict_list=pickle.load(open(out_file_dict,'rb'));
    
    [pred_scores_dict,pred_boxes_dict,bboxes_dict,segs_dict] = dict_list;
    
    img_list = saveSegViz(im,alpha,img_name,out_dir_overlay_meta,pred_scores_dict,pred_boxes_dict,bboxes_dict,segs_dict,overwrite)
    img_list = [im_path]+img_list;
    return img_list;
    # for img_curr in img_list:
    #     print img_curr;

def script_saveSegVizAll(params):

    path_to_im_meta = params.path_to_im_meta;
    path_to_im_canon = params.path_to_im_canon;
    path_to_predScores_meta = params.path_to_predScores_meta;
    path_to_seg_meta = params.path_to_seg_meta;
    path_to_mat_overlap = params.path_to_mat_overlap;
    path_to_score_mat_meta = params.path_to_score_mat_meta;
    img_names_txt = params.img_names_txt;
    scale_idx_range = params.scale_idx_range;
    stride = params.stride;
    w = params.w;
    alpha = params.alpha;
    num_to_pick = params.num_to_pick;
    out_dir_overlay_meta = params.out_dir_overlay_meta;
    power_scale_range = params.power_scale_range
    power_step_size = params.power_step_size
    out_file_html = params.out_file_html    
    height_width = params.height_width;
    num_threads = params.num_threads;
    out_dir_im_dict = params.out_dir_im_dict;
    overwrite = params.overwrite;

    util.mkdir(out_dir_overlay_meta);   
    for scale_curr in scale_idx_range:
        util.mkdir(os.path.join(out_dir_overlay_meta,str(scale_curr)));
    
    img_names=util.readLinesFromFile(img_names_txt);

    args=[];
    for idx_img_name,img_name in enumerate(img_names):
        arg_curr = (img_name,path_to_im_meta,path_to_im_canon,path_to_predScores_meta,path_to_seg_meta,path_to_mat_overlap,
            path_to_score_mat_meta,scale_idx_range,stride,w,alpha,num_to_pick,out_dir_overlay_meta,power_scale_range,power_step_size,out_dir_im_dict,overwrite,idx_img_name)
        args.append(arg_curr);

    # args=args[:10];
    # args= [args[idx] for idx in [422]]    
    # img_list_all=[];
    # for arg in args:
    #     img_list=saveSegVizImage(arg);
    #     img_list_all.append(img_list);
        # for img_curr in img_list:
        #     print img_curr;
        # break;

    p=multiprocessing.Pool(num_threads);
    img_list_all = p.map(saveSegVizImage,args);

    img_paths_html=[];
    captions_html=[];
    for img_list in img_list_all:
        img_row=[];
        caption_row=[];
        org_img=img_list[0];
        # print org_img
        img_list=img_list[1:];
        org_img_begin = '/'.join(org_img.split('/')[:2]);
        org_img = util.getRelPath(org_img,org_img_begin);
        # print org_img
        img_row.append(org_img);
        caption_row.append(util.getFileNames([org_img])[0]);
        for img_curr in img_list:
            img_begin = '/'.join(img_curr.split('/')[:2]);
            img_folder = img_curr.split('/')[-2];
            img_row.append(util.getRelPath(img_curr,img_begin));
            caption_row.append(img_folder);
        img_paths_html.append(img_row);
        captions_html.append(caption_row);

    visualize.writeHTML(out_file_html,img_paths_html,captions_html,height=height_width[0],width=height_width[1]);
    print out_file_html


def getPredBoxesAndInfoRel(gt_boxes,meta_info,thresh_overlap):
    mat_overlap=meta_info['mat_overlap'];
    boxes_include_idx=np.sum(mat_overlap>=thresh_overlap,axis=1)>0;

    pred_boxes= meta_info['pred_boxes'];
    pred_info = meta_info['pred_scores'];


    pred_info_rel=pred_info[boxes_include_idx,:];
    pred_boxes_rel=pred_boxes[boxes_include_idx,:];
    return pred_boxes_rel,pred_info_rel    

def getFloAndBoxes(pred_boxes_rel,pred_info_rel,im_size,score_mat_path,flo_rel_path,scales,w,stride):
    flos_boxes=[];
    for pred_info_curr in pred_info_rel:
        scale_idx=int(pred_info_curr[1]);
        row = pred_info_curr[2];
        col = pred_info_curr[3];
        scale_curr=scales[scale_idx];

        score_mat_file=os.path.join(score_mat_path[0],str(scale_idx),score_mat_path[1]);
        score_mat=np.load(score_mat_file)[0][0];
        
        bbox_containing=psr.getBoxContainingCanonical(row,col,im_size,scale_curr,score_mat.shape,stride,w)
        bbox_containing=[int(val) for val in bbox_containing];
        
        flo_rel_file=os.path.join(flo_rel_path[0],str(scale_idx),flo_rel_path[1]+'_'+str(int(row))+'_'+str(int(col))+'.png');
        flo_curr=scipy.misc.imread(flo_rel_file);
        
        flos_boxes.append((flo_curr,bbox_containing));
    return flos_boxes

def saveFloOverlay((mat_overlap_file,gt_file,im_file,thresh_overlap,score_mat_path,flo_rel_path,scales,w,stride,alpha_overlay,pred_color,gt_color,out_file)):
    im=scipy.misc.imread(im_file)
    im_size=im.shape;
    gt_boxes=np.load(gt_file);
    gt_boxes=np.array([psr.convertBBoxFormatToStandard(gt_box) for gt_box in gt_boxes]);
    meta_info=np.load(mat_overlap_file);
    pred_boxes_rel,pred_info_rel = getPredBoxesAndInfoRel(gt_boxes,meta_info,thresh_overlap)
    flos_boxes = getFloAndBoxes(pred_boxes_rel,pred_info_rel,im_size,score_mat_path,flo_rel_path,scales,w,stride)
    im_flo=getFloOverlay(im,flos_boxes,alpha_overlay);
    boxes_all=np.vstack((pred_boxes_rel,gt_boxes));
    colors=[pred_color]*pred_boxes_rel.shape[0]+[gt_color]*gt_boxes.shape[0];
    visualize.plotBBox(im_flo,boxes_all,out_file,colors);

def script_saveFloVizAll(params):
    mat_overlap_dir = params.mat_overlap_dir
    gt_dir = params.gt_dir
    thresh_overlap = params.thresh_overlap
    flo_viz_dir = params.flo_viz_dir
    results_dir_meta = params.results_dir_meta
    path_to_im_canon = params.path_to_im_canon
    out_dir_overlay_flo = params.out_dir_overlay_flo
    gt_color = params.gt_color
    pred_color = params.pred_color
    alpha_overlay = params.alpha_overlay
    w = params.w
    stride = params.stride
    power_scale_range = params.power_scale_range
    power_step_size = params.power_step_size
    num_threads = params.num_threads

    util.mkdir(out_dir_overlay_flo);
    power_range=np.arange(power_scale_range[0],power_scale_range[1]+power_step_size,power_step_size);
    scales=[2**val for val in power_range];
    im_names=util.getFileNames(util.getFilesInFolder(mat_overlap_dir,'.npz'),ext=False);

    args=[]
    for im_name in im_names:
        mat_overlap_file=os.path.join(mat_overlap_dir,im_name+'.npz');
        flo_viz_pre=os.path.join(flo_viz_dir,im_name);
        gt_file=os.path.join(gt_dir,im_name+'.npy');
        im_file=os.path.join(path_to_im_canon,im_name+'.jpg');

        score_mat_path = [results_dir_meta,im_name+'.npy']
        flo_rel_path = [flo_viz_dir,os.path.join('flo_im',im_name)]
        out_file=os.path.join(out_dir_overlay_flo,im_name+'.png');

        arg=(mat_overlap_file,gt_file,im_file,thresh_overlap,score_mat_path,flo_rel_path,scales,w,stride,alpha_overlay,pred_color,gt_color,out_file);
        args.append(arg);

    p=multiprocessing.Pool(num_threads);
    p.map(saveFloOverlay,args);
    
 
def main():
    
    out_file_html='/disk3/maheen_data/headC_160_withFlow_justHuman/overlay_flo_50_viz.html'
    folders=['/disk3/maheen_data/headC_160_noFlow_justHuman/gt_overlap_50_viz','/disk3/maheen_data/headC_160_withFlow_justHuman/gt_overlap_50_viz','/disk3/maheen_data/headC_160_withFlow_justHuman/overlay_flo_50_viz'];
    img_names=util.getFileNames(util.getFilesInFolder(folders[0],ext='.png'),ext=True);
    height=500;
    width=500;
    captions=['no flo','with flo','boxes with flo']
    visualize.writeHTMLForDifferentFolders(out_file_html,folders,captions,img_names,height=height,width=width);
    print out_file_html;
    return

    params_dict={};
    params_dict ['mat_overlap_dir'] = '/disk3/maheen_data/headC_160_withFlow_justHuman/mat_overlaps_1000';
    params_dict ['gt_dir'] = '/disk3/maheen_data/val_anno_human_only';
    params_dict ['thresh_overlap'] = 0.5;
    params_dict ['flo_viz_dir'] = '/disk3/maheen_data/headC_160_withFlow_justHuman/im_with_padding/flo';
    params_dict ['results_dir_meta'] = '/disk3/maheen_data/headC_160_withFlow_justHuman/results';
    params_dict ['path_to_im_canon'] = '/disk3/maheen_data/headC_160_withFlow_justHuman/im/4';
    params_dict ['out_dir_overlay_flo'] = '/disk3/maheen_data/headC_160_withFlow_justHuman/overlay_flo_50_viz';
    params_dict ['gt_color'] = (255,0,0);
    params_dict ['pred_color'] = (255,255,255);
    params_dict ['alpha_overlay'] = 0.4;
    params_dict ['w'] = 160;
    params_dict ['stride'] = 16;
    params_dict ['power_scale_range'] = (-2,1);
    params_dict ['power_step_size'] = 0.5;
    params_dict ['num_threads'] =  multiprocessing.cpu_count();

    params=createParams('saveFloVizAll');
    params=params(**params_dict);

    script_saveFloVizAll(params)
    pickle.dump(params._asdict(),open(os.path.join(params.out_dir_overlay_flo,'params_saveFloVizAll.p'),'wb'));


    

    
    # util.mkdir(out_dir_overlay_meta);   
    # for scale_curr in scale_idx_range:
    #     util.mkdir(os.path.join(out_dir_overlay_meta,str(scale_curr)));
    # power_scale_range=(-2,1);
    # power_step_size=0.5
    # power_range=np.arange(power_scale_range[0],power_scale_range[1]+power_step_size,power_step_size);
    # scales=[2**val for val in power_range];

    # img_names=util.readLinesFromFile(img_names_txt);
    # img_name = img_names[0];
    # im_path=os.path.join(path_to_im_canon,img_name+'.jpg');
    # im=scipy.misc.imread(im_path);
    # im_size=im.shape;
    # mat_overlap_file = os.path.join(path_to_mat_overlap,img_name+'.npz');
    # pred_scores_dict=getPredScoresDict(path_to_predScores_meta,path_to_score_mat_meta,img_name,scale_idx_range);
    # pred_boxes_dict = getPredBoxesDict(mat_overlap_file,num_to_pick);
    # bboxes_dict=getBBoxBigDict(pred_scores_dict,im_size,scales,stride,w);
    # segs_dict=getSegsDict(bboxes_dict,path_to_seg_meta,img_name);
    # img_list = saveSegViz(im,alpha,img_name,out_dir_overlay_meta,pred_scores_dict,pred_boxes_dict,bboxes_dict,segs_dict)
    # img_list = [im_path]+img_list;

    # for img_curr in img_list:
    #     print img_curr;






# def fuseAndSave(img,heatmap,alpha,out_file_curr=None):









if __name__=='__main__':
    main();