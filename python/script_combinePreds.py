import numpy as np
import util;
import os;
import visualize;
from PIL import Image,ImageDraw;
import matplotlib.pyplot as plt;

def getPaddedSize(max_r,max_c,stride=16,w=160):
	r_pad=((max_r-1)*stride)+160;
	c_pad=((max_c-1)*stride)+160;
	return (r_pad,c_pad)


def getPixStart(r_idx,c_idx,im_size_org,max_rows,max_cols,stride=16,w=160):

	
	p_size=getPaddedSize(max_rows,max_cols,stride,w);
	diffs=(p_size[0]-im_size_org[0],p_size[1]-im_size_org[1]);
	
	seg_no=[r_idx,c_idx];
	pix_start=[(idx)*16 for idx in seg_no];
	print pix_start
	pix_start_org=[idx-(diff/2) for idx,diff in zip(pix_start,diffs)];

	return pix_start_org;

def getScoresList(out_dir):
	file_list=util.getEndingFiles(out_dir,'_score.npy');

	score_list={};
	for file_curr in file_list:
		if file_curr.startswith('all'):
			continue;
		score_curr=np.load(os.path.join(out_dir,file_curr))[0];
		r_curr=int(file_curr[:file_curr.index('_')]);
		c_curr=int(file_curr[file_curr.index('_')+1:file_curr.rindex('_')]);
		score_list[(r_curr,c_curr)]=score_curr;
	return score_list;

def getHeatMap(arr,max_val=255):
    cmap = plt.get_cmap('jet')
    rgba_img = cmap(arr)
    rgb_img = np.delete(rgba_img, 3, 2)
    rgb_img = rgb_img*max_val
    # print rgb_img.shape,rgba_img.shape
    return rgb_img

def main():
	# out_dir='/disk2/aprilExperiments/headC_160/figuring_test/im_pred';
	# seg_file='1_1_seg.png'
	# seg_curr=np.load(os.path.join(out_dir,seg_file))[0];
	# print seg_curr[:,0];
	# seg_curr=np.load(os.path.join(out_dir,'1_2_seg.png'))[0];
	# print seg_curr[:,0];
	# # min_value=np.min(seg_curr);
	# # print min_value
	# # bin_curr=np.sum(seg_curr==min_value,axis=0);
	# # print bin_curr.shape;
	# # print bin_curr
	# # idx_emp=np.where(bin_curr==seg_curr.shape[0]);
	# # print len(idx_emp);
	# # print idx_emp
	# # print np.max(idx_emp);

	# # bin_curr=np.sum(seg_curr==0,axis=1);
	# # print bin_curr.shape;
	# # idx_emp=np.where(bin_curr==seg_curr.shape[1]);
	# # print len(idx_emp);
	# # print np.max(idx_emp);
	# # print idx_emp
	

	# return
	im_path='/disk2/ms_coco/train2014/COCO_train2014_000000460565.jpg';

	out_dir='/disk2/aprilExperiments/headC_160/figuring_test/im_pred_score';
	im_size_org=(427,640);
	score_list=getScoresList(out_dir);

	# out_dir_new='/disk2/aprilExperiments/headC_160/figuring_test/im_pred_score';
	score_all=np.load(os.path.join(out_dir,'all_score.npy'))[0][0];
	print score_all.shape

	for i in range(score_all.shape[0]):
		for j in range(score_all.shape[1]):
			score_bef=score_list[(i+1,j+1)];
			score_now=score_all[i,j];
			print score_bef,score_now,i,j
			print np.abs(score_bef-score_now)
			assert np.abs(score_bef-score_now)<0.00001;

	visualize.saveMatAsImage(score_all,os.path.join(out_dir,'new_mat.png'))


	return
	scores=score_list.values();
		
	idx_max=np.argmax(scores);

	idx_sort=np.argsort(scores)[::-1];
	# print idx_sort
	# return
	# [::-1];
	# print idx_sort[0],idx_max
	print scores
	idx_max=idx_sort[19];
	print scores[idx_max]

	print idx_max,scores[idx_max];
	# return
	keys=score_list.keys();
	[r,c]=zip(*keys);
	max_r=max(r);
	max_c=max(c);

	max_idx=keys[idx_max];
	
	pix_starts=[];
	good_ones=[max_idx];
	im=Image.open(im_path);
	draw = ImageDraw.Draw(im)

	for r_idx,c_idx in good_ones:
		print r_idx,c_idx;
		pix_start=getPixStart(r_idx-1,c_idx-1,im_size_org,max_r,max_c);
		pix_starts.append(pix_start);


		seg_curr=os.path.join(out_dir,str(r_idx)+'_'+str(c_idx)+'_seg.npy');
		seg_curr=np.load(seg_curr)[0];
		heatmap=getHeatMap(seg_curr);
		im=np.array(im);
		im=im*0.5;
		heatmap=heatmap*0.5;
		print pix_start
		# pix_start=[pix_start_curr-160 for pix_start_curr in pix_start]
		im_rel=im[pix_start[0]:pix_start[0]+160,pix_start[1]:pix_start[1]+160]
		im[pix_start[0]:pix_start[0]+160,pix_start[1]:pix_start[1]+160]=im_rel+heatmap[:min(heatmap.shape[0],im_rel.shape[0]),:min(heatmap.shape[1],im_rel.shape[1])];
	

	im=Image.fromarray(np.uint8(im));
	for pix_start in pix_starts:
		draw.rectangle([pix_start[1],pix_start[0],pix_start[1]+160,pix_start[0]+160]);

	im.save(os.path.join(out_dir,'check.png'));




	return
	for r_idx in range(0,max_rows,9):
		row_curr=[];
		scores_row=[];
		for c_idx in range(0,max_cols,9):
			seg_curr=np.load(os.path.join(out_dir,str(r_idx+1)+'_'+str(c_idx+1)+'_seg.png'))[0];
			seg_curr[seg_curr<0]=0;	
			score_curr=np.load(os.path.join(out_dir,str(r_idx+1)+'_'+str(c_idx+1)+'_score.png'))[0];
			print seg_curr.shape,score_curr;
			row_curr.append(seg_curr);
			scores_row.append(score_curr);
		
		scores.append(scores_row);
		
		row_curr=np.hstack(tuple(row_curr));
		if r_idx==0:
			img_yet=row_curr;
		else:
			img_yet=np.vstack((img_yet,row_curr));

	print img_yet.shape

	visualize.saveMatAsImage(img_yet,os.path.join(out_dir,'full_img.png'))




if __name__=='__main__':
	main();