import numpy as np;
import cv2
import multiprocessing;
import scipy.misc;
import util;
import os;
import visualize;

def resizeAndSave((img_path,mask_path,scale,crop_size,out_file_im,out_file_mask,idx)):
	print idx;

	assert scale<1;

	im=scipy.misc.imread(img_path);
	mask=scipy.misc.imread(mask_path);

	im_new=scipy.misc.imresize(im, scale,interp='bilinear');
	mask_new=scipy.misc.imresize(mask, scale,interp='nearest');

	crop_coors=[];
	for dim_curr in range(2):
		old_dim=im_new.shape[dim_curr];
		new_dim=crop_size[dim_curr];
		diff=(old_dim-new_dim)/2;
		crop_coor=[diff,diff+new_dim];
		crop_coors.append(crop_coor);

	if len(im_new.shape)>2:
		im_new=im_new[crop_coors[0][0]:crop_coors[0][1],crop_coors[1][0]:crop_coors[1][1],:]
	else:
		im_new=im_new[crop_coors[0][0]:crop_coors[0][1],crop_coors[1][0]:crop_coors[1][1]]
	
	if len(mask_new.shape)>2:
		mask_new=mask_new[crop_coors[0][0]:crop_coors[0][1],crop_coors[1][0]:crop_coors[1][1],:]
	else:
		mask_new=mask_new[crop_coors[0][0]:crop_coors[0][1],crop_coors[1][0]:crop_coors[1][1]]	
	
	assert im_new.shape[:2]==mask_new.shape[:2];

	coords=np.where(mask_new==1);
	dims=[];
	mins=[];
	for idx in range(len(coords)):
		dims.append(np.max(coords[idx])-np.min(coords[idx]));
		mins.append(np.min(coords[idx]))

	# print np.max(dims);
	# assert np.max(dims)==96;
	# assert np.min(mins)==40;

	scipy.misc.imsave(out_file_im,im_new);
	scipy.misc.imsave(out_file_mask,mask_new);




def main():
	pos_file='/disk2/februaryExperiments/deep_proposals/positive_data.txt';
	out_dir='/disk2/aprilExperiments/positives';
	util.mkdir(out_dir);

	scale=96/128.0;
	crop_size=[176,176];

	lines=util.readLinesFromFile(pos_file);
	# lines=lines[:6];

	print len(lines);
	
	args=[]
	for idx_line,line in enumerate(lines):
		[img_path,mask_path]=line.split(' ');
		out_file_im=os.path.join(out_dir,img_path[img_path.rindex('/')+1:]);
		out_file_mask=os.path.join(out_dir,mask_path[mask_path.rindex('/')+1:]);
		if os.path.exists(out_file_im):
			continue;
		else:
			args.append((img_path,mask_path,scale,crop_size,out_file_im,out_file_mask,idx_line));

	print len(args);
	# for arg in args:
	# 	resizeAndSave(arg);
	p=multiprocessing.Pool(multiprocessing.cpu_count());
	p.map(resizeAndSave,args);



if __name__=='__main__':
	main();
