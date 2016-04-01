import numpy as np;
import util;
import visualize;
import os;
import scipy.misc;
def reshapeMat(im,means):
	if im.shape[0]==3:
		im_layers=[];
		for im_dim in range(3):
			im_curr=im[im_dim,:,:];
			# im_curr=im[:,:,im_dim];
			im_curr=im_curr+means[im_dim];
			im_curr_d=im_curr.reshape(im.shape[1],im.shape[2]);
			im_layers.append(im_curr_d);
			# print im_curr_d.shape
			# print np.min(im_curr),np.max(im_curr)
		im=np.dstack(tuple(im_layers[:]));
	else:
		im=im.reshape((im.shape[1],im.shape[2]));
	return im;
		
def script_visualizeSegResults(pred_file,gt_output_file,gt_data_output_file,out_file_html,rel_path,means,out_dir):
	data=np.load(gt_data_output_file);
	gt= np.load(gt_output_file);
	out = np.load(pred_file);

	print data.shape;
	im_paths=[];captions=[];
	
	for im_no in range(data.shape[0]):
		print im_no;
		data_path=os.path.join(out_dir,str(im_no)+'_data.png');
		gt_path=os.path.join(out_dir,str(im_no)+'_gt.png');
		pred_path=os.path.join(out_dir,str(im_no)+'_pred.png');
		# scipy.misc.imsave(data_path,reshapeMat(data[im_no],means));
		visualize.saveMatAsImage(reshapeMat(data[im_no],means)/255,data_path);
		visualize.saveMatAsImage(reshapeMat(gt[im_no],means),gt_path);
		visualize.saveMatAsImage(reshapeMat(out[im_no],means),pred_path);
		im_paths.append([data_path.replace(rel_path[0],rel_path[1]),gt_path.replace(rel_path[0],rel_path[1]),pred_path.replace(rel_path[0],rel_path[1])]);
		captions.append(['im','mask_gt','mask_pred']);
		# if im_no==10:
		# 	break;

	visualize.writeHTML(out_file_html,im_paths,captions,height=224,width=224);
		# break;

def script_visualizeScoreResults(test_output_file,gt_output_file,gt_data_output_file,out_file_html,rel_path,means,out_dir):
	data=np.load(gt_data_output_file);
	gt_label=np.load(gt_output_file);
	pred_label=np.load(test_output_file);
	print data.shape,gt_label.shape,pred_label.shape

	# print data.shape;
	im_paths=[];captions=[];
	correct=0;
	for im_no in range(data.shape[0]):
		data_path=os.path.join(out_dir,str(im_no)+'_data.png');
		# gt_path=os.path.join(out_dir,str(im_no)+'_gt.png');
		# pred_path=os.path.join(out_dir,str(im_no)+'_pred.png');
		# scipy.misc.imsave(data_path,reshapeMat(data[im_no],means));

		visualize.saveMatAsImage(reshapeMat(data[im_no],means)/255,data_path);
		pred_label_curr=pred_label[im_no,0];
		gt_label_curr=gt_label[im_no,0];
		# visualize.saveMatAsImage(reshapeMat(gt[im_no],means),gt_path);
		# visualize.saveMatAsImage(reshapeMat(out[im_no],means),pred_path);
		im_paths.append([data_path.replace(rel_path[0],rel_path[1])]);
		if (pred_label_curr*gt_label_curr)>=0:
			correct=correct+1;

		captions.append(['Pred '+str(pred_label_curr)+' GT '+str(gt_label_curr)]);
		# if im_no==10:
		# 	break;
	print correct
	visualize.writeHTML(out_file_html,im_paths,captions,height=224,width=224);


def main():
	out_dir='/disk2/marchExperiments/deep_proposals/checkNegScaling';
	out_file_html=os.path.join(out_dir,'visualize.html');
	rel_path=['/disk2','../../../..']
	img_paths=[];
	captions=[];

	for i in range(1,11):
		img_row_curr=[];
		img_curr=os.path.join(out_dir,'img_'+str(i)+'.png');
		img_row_curr.append(img_curr.replace(rel_path[0],rel_path[1]));
		for j in range(1,3):
			img_curr=os.path.join(out_dir,'img_crop_'+str(j)+'_'+str(i)+'.png');
			img_row_curr.append(img_curr.replace(rel_path[0],rel_path[1]));
		img_paths.append(img_row_curr);
		captions.append(['','','']);
	visualize.writeHTML(out_file_html,img_paths,captions);
			




	return
	text_input='/disk2/februaryExperiments/deep_proposals/positive_data.txt';
	test_output_file='/disk2/februaryExperiments/deep_proposals/model_no_seg_test.npy';
	gt_output_file='/disk2/februaryExperiments/deep_proposals/model_no_seg_gt.npy';
	gt_data_output_file='/disk2/februaryExperiments/deep_proposals/model_no_seg_gt_data.npy';
	# out_dir='/disk2/februaryExperiments/deep_proposals/model_no_score_test';
	out_dir='/disk2/marchExperiments/deep_proposals/model_no_seg_test';
	
	test_output_file='/disk2/marchExperiments/deep_proposals/debugging_nan/pred.npy';
	gt_output_file='/disk2/marchExperiments/deep_proposals/debugging_nan/label.npy';
	gt_data_output_file='/disk2/marchExperiments/deep_proposals/debugging_nan/im.npy';


	out_dir='/disk2/marchExperiments/deep_proposals/model_seg_tan_test';
	util.mkdir(out_dir);
	
	out_file_html=os.path.join(out_dir,'visualize.html');
	
	rel_path=['/disk2','../../../..'];

	means=[122,117,104]

	script_visualizeSegResults(test_output_file,gt_output_file,gt_data_output_file,out_file_html,rel_path,means,out_dir)

	# script_visualizeSegResults(gt_data_output_file)

	# script_visualizeScoreResults(test_output_file,gt_output_file,gt_data_output_file,out_file_html,rel_path,means,out_dir)

	# lines=util.readLinesFromFile(text_input);
	# lines=lines[:100];

	# out=np.load(test_output_file);
	# gt=np.load(gt_output_file);
	



	return

	out_files=[];
	gt_files=[];
	mask_files=[];

	print out.shape,gt.shape
	for im_no in range(out.shape[0]):
		print im_no
		im=out[im_no];
		im=im.reshape((im.shape[1],im.shape[2]));
		# print im.shape
		out_file=os.path.join(out_dir,str(im_no+1)+'.png');
		visualize.saveMatAsImage(im,out_file);
		out_files.append(out_file);

		im=gt[im_no];
		im=im.reshape((im.shape[1],im.shape[2]));
		# print im.shape
		out_file=os.path.join(out_dir,str(im_no+1)+'_mask.png');
		visualize.saveMatAsImage(im,out_file);
		gt_files.append(out_file);


	out_file_html=os.path.join(out_dir,'visualize.html');
	rel_path=['/disk2','../../../..'];
	im_paths=[];captions=[];
	for idx,line in enumerate(lines):
		im_path=line[:line.index(' ')];
		# mask_path=line[line.index(' '):];
		# mask=scipy.misc.imread(mask_path);

		im_path=im_path.replace(rel_path[0],rel_path[1]);
		im_paths.append([im_path,out_files[idx].replace(rel_path[0],rel_path[1]),
			gt_files[idx].replace(rel_path[0],rel_path[1])]);
		captions.append(['im','pred','gt']);

	visualize.writeHTML(out_file_html,im_paths,captions,height=224,width=224);

if __name__=='__main__':
	main();

