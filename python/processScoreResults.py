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


def parseTorchCommand(path_to_test_file,model,testFile,outDir,splitModel=False,limit=-1,gpu=1,overwrite=False):
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
	
	command.extend(['-gpu',str(gpu)]);
	command=' '.join(command);
	return command;


def script_writeTestSH(out_files,out_dirs,out_file_sh,path_to_test_file,model,limit,gpu,overwrite=False):
	commands_all=[];
	for test_file,out_dir in zip(out_files,out_dirs):
		# print test_file,out_dir
		command=parseTorchCommand(path_to_test_file,model,test_file,out_dir,limit=limit,gpu=gpu,overwrite=overwrite);
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

def getPredBoxesCanonicalFromBBoxTight(box_curr,row,col,im_size,scale_curr,seg_shape,stride,w):
	if len(im_size)>2:
		img_size_curr=(im_size[0]*scale_curr,im_size[1]*scale_curr,im_size[2]);
	else:
		img_size_curr=(im_size[0]*scale_curr,im_size[1]*scale_curr);
	# print 'img_size_curr',img_size_curr,'row',row,'col,',col
	bbox_containing=getBBox(img_size_curr,seg_shape,row,col,stride,w,keepNeg=True)
	# print 'bbox_containing',bbox_containing
	pred_box=getBBoxTightCanonical(bbox_containing,box_curr)
	# print 'pred_box',pred_box
	pred_box=np.array(pred_box)/float(scale_curr);
	# print 'pred_box',pred_box
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

def main():

	dir_meta_validation='/disk2/mayExperiments/validation/rescaled_images'
	dir_gt_boxes='/disk2/mayExperiments/validation_anno';
	dir_meta_test='/disk3/maheen_data/headC_160_noFlow/'
	dir_meta_test='/disk3/maheen_data/headC_160_noFlow_bbox/'
	util.mkdir(dir_meta_test);

	dir_canon_im='/disk2/mayExperiments/validation/rescaled_images/4'
	# dir_validations=[os.path.join(dir_meta_validation,str(num)) for num in range(6)];
	dir_validations=[os.path.join(dir_meta_validation,str(num)) for num in [5,6]];

	out_files=[dir_curr+'.txt' for dir_curr in dir_validations];
	# script_saveValidationTexts(dir_validations,out_files);

	path_to_test_file='/home/maheenrashid/Downloads/deep_proposals/torch_new/test_command_line_bigIm.th';
	model='/disk2/aprilExperiments/headC_160/noFlow_gaussian_all_actual/intermediate_resume_2/model_all_70000.dat';
	
	# out_file_sh=os.path.join(dir_meta_test,'test_20.sh');
	gpu=1;
	limit=-1;
	out_file_sh=os.path.join(dir_meta_test,'test_'+str(limit)+'.sh');

	power_scale_range=(-2,1);
	power_step_size=0.5
	thresh_seg=0.2;
	stride=16;
	w=160;
	overwrite=False;
	top=100;

	out_dirs=[os.path.join(dir_meta_test,util.getFileNames([out_file],ext=False)[0]) for out_file in out_files];
	script_writeTestSH(out_files,out_dirs,out_file_sh,path_to_test_file,model,limit,gpu,overwrite);
	print out_file_sh
	# subprocess.call('sh '+out_file_sh,shell=True);
	return

	# get scales
	power_range=np.arange(power_scale_range[0],power_scale_range[1]+power_step_size,power_step_size);
	scales=[2**val for val in power_range];
	scales=scales[:-1]


	# get top n boxes for image.
	im_list=util.readLinesFromFile(out_files[0])[:limit];
	im_names=util.getFileNames(im_list,ext=False);

	im_name=im_names[1];
	im_path=im_list[1];
	print im_name,im_path,
	im_path_canonical=os.path.join(dir_canon_im,im_name+'.jpg')
	im_size=scipy.misc.imread(im_path_canonical);
	im_size=im_size.shape;
	print im_size;


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

		print 'score_curr',score_curr
		# print '(box_curr,row,col,im_size,scale_curr,seg_shape,stride,w)',(box_curr,row,col,im_size,scale_curr,seg_shape,stride,w)

		box_curr=getPredBoxesCanonicalFromBBoxTight(box_curr,row,col,im_size,scale_curr,seg_shape,stride,w)
		# print box_curr
		bboxes[idx_to_fill]=box_curr;
		scores_record[idx_to_fill]=score_curr;
		idx_to_fill+=1;
		if idx_to_fill==top:
			break;


	# print len(bboxes);
	# bboxes=nms.non_max_suppression_fast(bboxes, 0.99)
	# print bboxes.shape
	# bboxes=bboxes[:1000];
	visualize.plotBBox(im_path_canonical,bboxes,'/disk2/temp/check_pred.png');

	



	# for pred_info_curr in pred_info:
	# 	print pred_info_curr

	# pred_info_curr=pred_info[4];
	# seg=np.load(pred_info_curr[1]);
	# seg_shape=(seg.shape[0],seg.shape[1])
	# scale_curr=pred_info_curr[2];
	# boxes=[];
	# for row in range(2):
	# # seg_shape[0]):
	# 	for col in range(2):
	# 	# seg_shape[1]):
	# 		box_curr=[0,0,159,159];
	# 		print row,col,box_curr,
	# 		box_new=getPredBoxesCanonicalFromBBoxTight(box_curr,row,col,im_size,scale_curr,seg_shape,stride,w)
	# 		print box_new
	# 		boxes.append(box_new);

	# visualize.plotBBox(im_path_canonical,boxes,'/disk2/temp/check_pred.png');


	# return


	return

	# get predictions and scales for all images
	pred_info=[]

	for scale_idx in range(len(scales)):
		out_dir_curr=os.path.join(dir_meta_test,str(scale_idx));
		
		scale_curr=scales[scale_idx];
		# print out_dir_curr,scale_curr
		file_seg=os.path.join(out_dir_curr,im_name+'_seg.npy');
		file_score=os.path.join(out_dir_curr,im_name+'.npy');
		if os.path.exists(file_seg):
			assert os.path.exists(file_score);
	
			pred_info.append([file_score,file_seg,scale_curr]);


	# get gt boxes
	gt_file=os.path.join(dir_gt_boxes,im_name+'.npy');
	assert os.path.exists(gt_file);
	gt_boxes=np.load(gt_file);
	gt_boxes=np.array([convertBBoxFormatToStandard(gt_box) for gt_box in gt_boxes]);

	# get pred boxes canonical
	pred_boxes_all=np.zeros((0,4));
	confidence_all=np.zeros((0,));
	for pred_info_curr in pred_info:
		pred_boxes,confidence=getPredBoxesCanonical(pred_info_curr[0],pred_info_curr[1],pred_info_curr[2],im_size,stride,w,thresh_seg);
		pred_boxes_all=np.append(pred_boxes_all,pred_boxes,axis=0);
		confidence_all=np.append(confidence_all,confidence,axis=0);

	print pred_boxes_all

	# get mat overlap
	mat_overlap=np.zeros((pred_boxes_all.shape[0],gt_boxes.shape[0]));
	for pred_idx in range(mat_overlap.shape[0]):
		for gt_idx in range(mat_overlap.shape[1]):
			mat_overlap[pred_idx,gt_idx]=getBBoxIOU(pred_boxes_all[pred_idx],gt_boxes[gt_idx]);

	print mat_overlap;






if __name__=='__main__':
	main();