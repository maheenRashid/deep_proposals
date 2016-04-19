import numpy as np;
import scipy.misc;
from PIL import Image, ImageDraw
import os;
import util;
import visualize;
# import matplotlib.pyplot as plt

def drawCropAndBBox(im,bbox,crop_box):
	draw = ImageDraw.Draw(im)
	for box_no in range(bbox.shape[0]):
		draw.rectangle([(bbox[box_no,0],bbox[box_no,1]),(bbox[box_no,0]+bbox[box_no,2],bbox[box_no,1]+bbox[box_no,3])],outline=(255,255,255))
	draw.rectangle([(crop_box[0],crop_box[1]),(crop_box[2],crop_box[3])],outline=(0,0,255))
	del draw
	return im

def script_makeNegImages():

	in_dir='/disk2/aprilExperiments/testing_neg_fixed_test/'
	out_dir=os.path.join(in_dir,'visualizing_negs');
	util.mkdir(out_dir);


	# return
	im_paths=[os.path.join(in_dir,file_curr) for file_curr in os.listdir(in_dir) if file_curr.endswith('.png')];
	for im_path in im_paths:
		print im_path
		bbox=np.load(im_path.replace('.png','_bbox.npy'));
		crop_box=np.load(im_path.replace('.png','_crop.npy'));

		# print im_path.replace('.png','_bbox.npy')
		# print im_path.replace('.png','_crop.npy')
		# break;

		im = Image.open(im_path);
		im=drawCropAndBBox(im,bbox,crop_box)
		
		# write to stdout
		im.save(os.path.join(out_dir,im_path[im_path.rindex('/')+1:]), "PNG");
		# break;
	visualize.writeHTMLForFolder(out_dir,ext='png',height=300,width=300);


def getCorrespondingTolerance(pos_box,max_dim,tolerance):
	# bbox is xmin,ymin,xmax,ymax
	dims=[pos_box[2],pos_box[3]];
	max_dim_pos=max(dims)
	max_dim_pos_idx=dims.index(max_dim_pos);
	scale=float(max_dim)/max_dim_pos;
	print (max_dim_pos,scale)
	new_tolerance=tolerance/scale;

	center_box=[pos_box[idx]+dim_curr/2.0 for idx,dim_curr in enumerate(dims)];
	box_to_plot=[center_box[0]-tolerance,center_box[1]-tolerance,2*tolerance,2*tolerance];
	
	return box_to_plot,scale,new_tolerance

def scaleTest(pos_box,crop_box,tolerances):
	area_pos=pos_box[2]*pos_box[3];
	print (crop_box[2]-crop_box[0]),(crop_box[3]-crop_box[1])
	area_crop=(crop_box[2]-crop_box[0])*(crop_box[3]-crop_box[1]);
	scale_diff=area_crop/float(area_pos);
	# print area_pos,area_crop,scale_diff;
	isValid=True;
	
	if tolerances[0]<scale_diff<tolerances[1]:
		isValid=False;
	
	print (area_pos,area_crop,scale_diff,isValid);


def main():
	in_dir='/disk2/aprilExperiments/testing_neg_torch';
	# check_dir=os.path.join(in_dir,'check_crops_invalid');
	# out_file_html='visualize_crops.html';
	visualize.writeHTMLForFolder(in_dir,'.png',200,200);

	return

	in_dir='/disk2/aprilExperiments/testing_neg_fixed_test';
	img_path=os.path.join(in_dir,'1.png');
	crop_path_pos=img_path.replace('.png','_crop_pos.npy');
	crop_path_neg=img_path.replace('.png','_crop_neg.npy');

	bbox_path=img_path.replace('.png','_bbox.npy');

	im=scipy.misc.imread(img_path);
	bbox=np.load(bbox_path);
	crop_box_pos=np.load(crop_path_pos);
	crop_box_neg=np.load(crop_path_neg);

	# crop_box_pos=np.array(crop_path_pos,dty);
	# crop_box_neg=np.array(crop_path_neg,dty);


	out_file_neg_box=img_path.replace('.png','_box_neg.png');
	out_file_pos_box=img_path.replace('.png','_box_pos.png');
	im=Image.open(img_path);
	# draw=Image.ImageDraw(im);

	# draw = ImageDraw.Draw(im)
	# for crop_box in crop_box_neg:
	# 	# draw.rectangle([(bbox[box_no,0],bbox[box_no,1]),(bbox[box_no,0]+bbox[box_no,2],bbox[box_no,1]+bbox[box_no,3])],outline=(255,255,255))
	# 	draw.rectangle([(crop_box[0],crop_box[1]),(crop_box[2],crop_box[3])],outline=(0,0,255))

	# box_no=0;
	# draw.rectangle([(bbox[box_no,0],bbox[box_no,1]),(bbox[box_no,0]+bbox[box_no,2],bbox[box_no,1]+bbox[box_no,3])],outline=(255,255,255))
	# del draw

	# print out_file_neg_box
	# im.save(out_file_neg_box, "PNG");

	

	
	to_plot=[];
	print crop_box_pos.shape

	# min_x_req=bbox[0,0]-224;
	# min_y_req=bbox[0,1]-224;
	# max_x_req=bbox[0,0]+bbox[0,2]+224;
	# max_y_req=bbox[0,1]+bbox[0,3]+224;

	center_x=bbox[0,0]+bbox[0,2]/2;
	center_y=bbox[0,1]+bbox[0,3]/2;
	min_x_req=center_x-224;
	min_y_req=center_y-224;
	max_x_req=center_x+224;
	max_y_req=center_y+224;


	for idx_crop_box_curr,crop_box_curr in enumerate(crop_box_pos):
		# if idx_crop_box_curr%100==0:
		# 	print idx_crop_box_curr;
		[min_x,min_y,max_x,max_y]=list(crop_box_curr);
		if min_x>=min_x_req and min_y>=min_y_req and max_x<=max_x_req and max_y<=max_y_req:
			to_plot.append(idx_crop_box_curr);


	print len(to_plot)

	# draw = ImageDraw.Draw(im);
	crop_check_dir=os.path.join(in_dir,'check_crops_invalid');
	util.mkdir(crop_check_dir);
	print crop_check_dir;
	# return
	im_np=np.array(im);

	for idx_idx_box,idx_box in enumerate(range(len(crop_box_neg))):
		if idx_idx_box%1000==0:
			print idx_idx_box;
		crop_box=crop_box_neg[idx_box];
		im_curr=im_np[crop_box[1]:crop_box[3],crop_box[0]:crop_box[2],:];
		out_file=os.path.join(crop_check_dir,str(idx_idx_box)+'.png');
		scipy.misc.imsave(out_file,im_curr);

	# 	draw.rectangle([(crop_box[0],crop_box[1]),(crop_box[2],crop_box[3])],outline=(0,0,255))

	# box_no=0;
	# draw.rectangle([(bbox[box_no,0],bbox[box_no,1]),(bbox[box_no,0]+bbox[box_no,2],bbox[box_no,1]+bbox[box_no,3])],outline=(255,255,255))
	# del draw
	
	# print out_file_pos_box
	# im.save(out_file_pos_box, "PNG");


	return
	im=scipy.misc.imread(img_path);
	bbox=np.load(bbox_path);
	crop_box_pos=np.load(crop_path_pos);
	crop_box_neg=np.load(crop_path_neg);

	print bbox.shape
	ys=bbox[:,1];
	print ys.shape
	print np.argmax(ys);
	print bbox[np.argmax(ys)]

	# return
	crop_box_pos=np.array(crop_box_pos,dtype='int');
	crop_box_neg=np.array(crop_box_neg,dtype='int');
	
	print crop_box_pos.shape
	print crop_box_neg.shape
	new_im=np.zeros((im.shape[0],im.shape[1]));
	for crop_box_curr in crop_box_neg:
		# print crop_box_curr
		new_im[crop_box_curr[1]:crop_box_curr[3],crop_box_curr[0]:crop_box_curr[2]]=new_im[crop_box_curr[1]:crop_box_curr[3],crop_box_curr[0]:crop_box_curr[2]]+1;

	im_new=new_im;
	visualize.saveMatAsImage(im_new,os.path.join(in_dir,'heat_map_neg.png'));

	new_im=np.zeros((im.shape[0],im.shape[1]));
	for crop_box_curr in crop_box_pos:
		# print crop_box_curr
		new_im[crop_box_curr[1]:crop_box_curr[3],crop_box_curr[0]:crop_box_curr[2]]=new_im[crop_box_curr[1]:crop_box_curr[3],crop_box_curr[0]:crop_box_curr[2]]+1;

	im_new=new_im;
	print im_new.shape
	visualize.saveMatAsImage(im_new,os.path.join(in_dir,'heat_map_pos.png'));





	

	return
	script_makeNegImages();
	return

	in_dir='/disk2/aprilExperiments/testing_neg';
	img_path=os.path.join(in_dir,'71.png');
	crop_path=img_path.replace('.png','_crop.npy');
	bbox_path=img_path.replace('.png','_bbox.npy');

	max_dim=128;
	tolerance=32;

	bbox=np.load(bbox_path);
	crop_box=np.load(crop_path);
	im = Image.open(img_path);
	
	box_to_plot,scale,new_tolerance=getCorrespondingTolerance(bbox[0],max_dim,tolerance);

	
    # print (center_box,center_crop)
    



	center_box=[bbox[0,0]+bbox[0,2]/2.0,bbox[0,1]+bbox[0,3]/2.0];
	center_crop=[crop_box[0]+(crop_box[2]-crop_box[0])/2.0,crop_box[1]+(crop_box[3]-crop_box[1])/2.0];
	print center_box,center_crop
	dist_centers=np.sqrt(np.sum(np.power(np.array(center_box)-np.array(center_crop),2)));
	print (new_tolerance,dist_centers)

	if dist_centers<new_tolerance:
		scaleTest(bbox[0],crop_box,[0.5,2]);
	# print dist_centers,new_tolerance

	

	# bbox=np.vstack((bbox,box_to_plot));
	# print bbox
	
	# im=drawCropAndBBox(im,bbox,crop_box);
	# im=np.array(im);


	# plt.ion();
	# plt.figure();
	# plt.imshow(im);
	# plt.show();
	# raw_input();



if __name__=='__main__':
	main();