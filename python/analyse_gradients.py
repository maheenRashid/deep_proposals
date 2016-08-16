
import numpy as np;
import os;
import util;
from collections import OrderedDict;
import visualize;
def getGradientMags(np_files):

	# print np_files;

	gradients=[];
	for np_file in np_files:
		grad=np.load(np_file);
		# print grad.shape;
		grad_mag=np.linalg.norm(grad,axis=1);
		gradients.extend(list(grad_mag));

	return gradients;

def writeCommandsForTrainDebug(params,path_to_train_file,out_file_commands_pre,model_num):
	model_num=[str(model_num_curr) for model_num_curr in model_num];
	print model_num;

	commands=[];

	for path_to_folder,out_dir,segConstant in params:
		for model_num_curr in model_num:
			command='th '+path_to_train_file+' -model '+os.path.join(path_to_folder,'model_all_'+model_num_curr+'.dat')+' -outDir '+os.path.join(out_dir,str(model_num_curr))+' -segConstant '+segConstant;
			commands.append(command);
			print command;

	print len(commands)
	idx_split=util.getIdxRange(len(commands),3);
	for idx_idx,begin_idx in enumerate(idx_split[:-1]):
		end_idx=idx_split[idx_idx+1];
		commands_curr=commands[begin_idx:end_idx];
		out_file_curr=out_file_commands_pre+str(idx_idx)+'.sh';
		util.writeFile(out_file_curr,commands_curr);
		print 'sh '+out_file_curr;


def getMagInfo(file_pairs,alt=True,range_to_choose=None):
	grads_all=[];
	weights_all=[];
	ratios_all=[];
	for file_grad_curr,file_weight_curr  in file_pairs:
		grads=np.load(file_grad_curr);
		weights=np.load(file_weight_curr);
		if alt==True:
			grads=grads[::2];
			weights=weights[::2];
		if range_to_choose is not None:
			grads=[grads[x] for x in range_to_choose];
			weights=[weights[x] for x in range_to_choose];

		ratios=[grad_curr/weight_curr for grad_curr,weight_curr in zip(grads,weights)];
		grads_all.append(grads);
		weights_all.append(weights);
		ratios_all.append(ratios);
	grads_all=np.array(grads_all);
	weights_all=np.array(weights_all);
	ratios_all=np.array(ratios_all);
	
	grads_mean=np.mean(grads_all,axis=0);
	weights_mean=np.mean(weights_all,axis=0);
	ratios_mean=np.mean(ratios_all,axis=0);

	means=[grads_mean,weights_mean,ratios_mean];
	totals=[grads_all,weights_all,ratios_all];
	return means,totals

	# print grads_all.shape,weights_all.shape,ratios_all.shape,grads_mean.shape,weights_mean.shape,ratios_mean.shape


def compareMagInfoLayerTime(dirs,out_dirs,model_num,num_iters,file_pre_grad,file_pre_weight,out_file_html):
	img_paths=[];
	captions=[];

	for dir_curr,out_dir_curr in zip(dirs,out_dirs):
		dict_for_plotting={'grads_mag':OrderedDict(),'weights_mag':OrderedDict(),'ratios':OrderedDict()};

		dict_for_plotting=OrderedDict(dict_for_plotting.items());
		for model_num_curr in model_num:
			file_pairs=[(os.path.join(dir_curr,model_num_curr,file_pre_grad+str(iter_curr)+'.npy'),
					os.path.join(dir_curr,model_num_curr,file_pre_weight+str(iter_curr)+'.npy')) for iter_curr in range(1,num_iters)];
			means,_=getMagInfo(file_pairs,alt=True);

			dict_for_plotting['grads_mag'][model_num_curr]=means[0];
			dict_for_plotting['weights_mag'][model_num_curr]=means[1];
			dict_for_plotting['ratios'][model_num_curr]=means[2];

		img_paths_curr=[];
		captions_curr=[];
		for key_curr in dict_for_plotting.keys():
			out_file_curr=os.path.join(out_dir_curr,key_curr+'.png');
			data=dict_for_plotting[key_curr];
			xAndYs=data.values();
			legend_entries=data.keys();
			xAndYs=[(range(len(x_curr)),x_curr) for x_curr in xAndYs];
			visualize.plotSimple(xAndYs,out_file_curr,title=key_curr,xlabel='layer',ylabel='magnitude',legend_entries=legend_entries,outside=True);
			print out_file_curr.replace('/disk3','vision3.cs.ucdavis.edu:1001');
			img_paths_curr.append(util.getRelPath(out_file_curr,'/disk3'));
			# print dir_curr.split('/');
			captions_curr.append(dir_curr.split('/')[-2]+' '+dir_curr.split('/')[-1]+' '+key_curr);

		img_paths.append(img_paths_curr);
		captions.append(captions_curr);

	visualize.writeHTML(out_file_html,img_paths,captions,height=200,width=200);
	print out_file_html.replace('/disk3','vision3.cs.ucdavis.edu:1001');


def compareMagInfoLayerTime(dirs,out_dirs,model_num,num_iters,file_pre_grad,file_pre_weight,out_file_html):
	img_paths=[];
	captions=[];

	for dir_curr,out_dir_curr in zip(dirs,out_dirs):
		dict_for_plotting={'grads_mag':OrderedDict(),'weights_mag':OrderedDict(),'ratios':OrderedDict()};

		dict_for_plotting=OrderedDict(dict_for_plotting.items());
		for model_num_curr in model_num:
			print model_num_curr;
			file_pairs=[(os.path.join(dir_curr,model_num_curr,file_pre_grad+str(iter_curr)+'.npy'),
					os.path.join(dir_curr,model_num_curr,file_pre_weight+str(iter_curr)+'.npy')) for iter_curr in num_iters];
			means,_=getMagInfo(file_pairs,alt=True);

			dict_for_plotting['grads_mag'][model_num_curr]=means[0];
			dict_for_plotting['weights_mag'][model_num_curr]=means[1];
			dict_for_plotting['ratios'][model_num_curr]=means[2];

		img_paths_curr=[];
		captions_curr=[];
		for key_curr in dict_for_plotting.keys():
			out_file_curr=os.path.join(out_dir_curr,key_curr+'.png');
			data=dict_for_plotting[key_curr];
			xAndYs=data.values();
			legend_entries=data.keys();
			xAndYs=[(range(len(x_curr)),x_curr) for x_curr in xAndYs];
			visualize.plotSimple(xAndYs,out_file_curr,title=key_curr,xlabel='layer',ylabel='magnitude',legend_entries=legend_entries,outside=True);
			print out_file_curr.replace('/disk3','vision3.cs.ucdavis.edu:1001');
			img_paths_curr.append(util.getRelPath(out_file_curr,'/disk3'));
			# print dir_curr.split('/');
			captions_curr.append(dir_curr.split('/')[-2]+' '+dir_curr.split('/')[-1]+' '+key_curr);

		img_paths.append(img_paths_curr);
		captions.append(captions_curr);

	visualize.writeHTML(out_file_html,img_paths,captions,height=500,width=500);
	print out_file_html.replace('/disk3','vision3.cs.ucdavis.edu:1001');

def compareGradInfoLayer(dirs,out_dirs,model_num,num_iters,file_pre_grad,layer_range,plot_title):
	img_paths=[];
	for dir_curr,out_dir_curr in zip(dirs,out_dirs):
		# dict_for_plotting={plot_title:OrderedDict()};

		dict_for_plotting=OrderedDict();
		for model_num_curr in model_num:
			file_pairs=[(os.path.join(dir_curr,model_num_curr,file_pre_grad+str(iter_curr)+'.npy'),
					os.path.join(dir_curr,model_num_curr,file_pre_grad+str(iter_curr)+'.npy')) for iter_curr in num_iters];
			means,_=getMagInfo(file_pairs,alt=True,range_to_choose=layer_range);
			dict_for_plotting[model_num_curr]=means[0];

		# for key_curr in dict_for_plotting.keys():
		out_file_curr=os.path.join(out_dir_curr,plot_title+'.png');
		xAndYs=dict_for_plotting.values();
		legend_entries=dict_for_plotting.keys();
		xAndYs=[(layer_range,x_curr) for x_curr in xAndYs];
		visualize.plotSimple(xAndYs,out_file_curr,title=plot_title,xlabel='layer',ylabel='magnitude',legend_entries=legend_entries,outside=True);
		img_paths.append(out_file_curr);

	return img_paths;			
	

def main():


	# dirs=['/disk3/maheen_data/headC_160_withFlow_human_xavier_unit_floStumpPretrained_fullTraining/gradient_checks'];
	# # dirs=[os.path.join(dir_curr,dir_in) for dir_curr in dirs for dir_in in ['correct_16']];
	# range_flow=range(1);
	# out_dirs=[os.path.join(dir_curr,'plots') for dir_curr in dirs];
	# [util.mkdir(out_dir_curr) for out_dir_curr in out_dirs];
	
	# model_num=range(5000,45000,5000);
	# model_num.append(45000);
	# print model_num
	
	# model_num=[str(model_num_curr) for model_num_curr in model_num]	
	# num_iters=range(1,5);
	# # num_iters=range(2,3);
	# file_pre_weight='weight_mag_n_';
	# file_pre_grad='grad_mag_n_';	


	# out_file_html=os.path.join(dirs[0],'comparison_grads_weights_ratios_n.html');

	# compareMagInfoLayerTime(dirs,out_dirs,model_num,num_iters,file_pre_grad,file_pre_weight,out_file_html)
	
	# out_file_html=os.path.join(dirs[0],'comparison_grads_seg_no_seg_16.html');
	
	# layer_range=[26,27,28,31];
	# num_iters=range(1,5,2);
	# img_paths_seg_flow=compareGradInfoLayer([dirs[i] for i in range_flow],[out_dirs[i] for i in range_flow],model_num,num_iters,file_pre_grad,layer_range,'seg_grad_mag')
	# layer_range=range(26,32);
	# num_iters=range(2,5,2);
	# img_paths_score_flow=compareGradInfoLayer([dirs[i] for i in range_flow],[out_dirs[i] for i in range_flow],model_num,num_iters,file_pre_grad,layer_range,'score_grad_mag')

	# img_paths=[img_paths_seg_flow,img_paths_score_flow];
	# img_paths=[[util.getRelPath(path_curr,'/disk3') for path_curr in list_curr] for list_curr in img_paths];
	# captions=[];
	# for list_curr in img_paths:
	# 	captions_curr=[];
	# 	for path in list_curr:
	# 		path_split=path.split('/');
	# 		caption=path_split[-4]+' '+path_split[-3];
	# 		captions_curr.append(caption);
	# 		print caption
	# 	captions.append(captions_curr);
	# visualize.writeHTML(out_file_html,img_paths,captions,height=500,width=500);
	# print out_file_html.replace('/disk3','vision3.cs.ucdavis.edu:1001');



	# return
	# model_dir='/disk3/maheen_data/headC_160/withFlow_human_xavier_unit_floStumpPretrained_fullTraining/intermediate'
	# out_dir_meta='/disk3/maheen_data/headC_160_withFlow_human_xavier_unit_floStumpPretrained_fullTraining'
	# out_dir=os.path.join(out_dir_meta,'gradient_checks')
	# util.mkdir(out_dir);
	# params=[(model_dir,out_dir,'40')];
	# out_file_commands_pre=os.path.join(out_dir_meta,'debug_commands_');		
	# path_to_train_file='/home/maheenrashid/Downloads/deep_proposals/torch_new/headC_160_withFlow_debug.th'


	# model_num=range(5000,45000,5000);
	# model_num.append(45000);
	# print model_num
	# # return
	# writeCommandsForTrainDebug(params,path_to_train_file,out_file_commands_pre,model_num)


	# return

	dirs=['/disk3/maheen_data/headC_160/withFlow_human_debug'];
	dirs=[os.path.join(dir_curr,dir_in) for dir_curr in dirs for dir_in in ['correct']];
	range_flow=range(1);
	# range_noflow=range(2,len(dirs));

	# dirs=['/disk3/maheen_data/headC_160/withFlow_human_debug','/disk3/maheen_data/headC_160/noFlow_human_debug'];
	# dirs=[os.path.join(dir_curr,dir_in) for dir_curr in dirs for dir_in in ['correct_16','incorrect']];
	# range_flow=range(2);
	# range_noflow=range(2,len(dirs));

	out_dirs=[os.path.join(dir_curr,'plots') for dir_curr in dirs];
	[util.mkdir(out_dir_curr) for out_dir_curr in out_dirs];

	model_num=range(5000,100000,20000);
	model_num.append(100000);
	
	# model_num=range(2000,32000,6000);
	# model_num.append(32000);
	
	model_num=[str(model_num_curr) for model_num_curr in model_num]	
	print model_num
	# num_iters=range(1,21);
	num_iters=range(2,3);
	file_pre_weight='weight_mag_n_';
	file_pre_grad='grad_mag_n_';	


	out_file_html=os.path.join('/disk3/maheen_data/headC_160/withFlow_human_debug','comparison_grads_weights_ratios.html');

	compareMagInfoLayerTime(dirs,out_dirs,model_num,num_iters,file_pre_grad,file_pre_weight,out_file_html)
	# compareMagInfoLayerTime(dirs,out_dirs,model_num,num_iters,file_pre_grad,file_pre_weight,out_file_html)
	out_file_html=os.path.join('/disk3/maheen_data/headC_160/withFlow_human_debug','comparison_grads_seg_no_seg.html');
	

	layer_range=[26,27,28,31];
	# layer_range=[27,28];

	num_iters=range(3,21,2);
	img_paths_seg_flow=compareGradInfoLayer([dirs[i] for i in range_flow],[out_dirs[i] for i in range_flow],model_num,num_iters,file_pre_grad,layer_range,'seg_grad_mag')
	layer_range=range(26,32);
	num_iters=range(2,21,2);
	img_paths_score_flow=compareGradInfoLayer([dirs[i] for i in range_flow],[out_dirs[i] for i in range_flow],model_num,num_iters,file_pre_grad,layer_range,'score_grad_mag')

	# layer_range=[13,14,17]
	# num_iters=range(1,21,2);
	# img_paths_seg_noflow=compareGradInfoLayer([dirs[i] for i in range_noflow],[out_dirs[i] for i in range_noflow],model_num,num_iters,file_pre_grad,layer_range,'seg_grad_mag')
	# layer_range=range(13,18);
	# num_iters=range(2,21,2);
	# img_paths_score_noflow=compareGradInfoLayer([dirs[i] for i in range_noflow],[out_dirs[i] for i in range_noflow],model_num,num_iters,file_pre_grad,layer_range,'score_grad_mag')

	# layer_range=[26,27,28,31];
	# num_iters=range(1,21,2);
	# img_paths_seg_flow=compareGradInfoLayer(dirs[1:],out_dirs[1:],model_num,num_iters,file_pre_grad,layer_range,'seg_grad_mag')
	# layer_range=range(26,32);
	# num_iters=range(2,21,2);
	# num_iters=range(2,7,2);
	# img_paths_score_flow=compareGradInfoLayer(dirs[1:],out_dirs[1:],model_num,num_iters,file_pre_grad,layer_range,'score_grad_mag')
	# img_paths=[img_paths_seg_flow,img_paths_score_flow,img_paths_seg_noflow,img_paths_score_noflow];

	img_paths=[img_paths_seg_flow,img_paths_score_flow];
	img_paths=[[util.getRelPath(path_curr,'/disk3') for path_curr in list_curr] for list_curr in img_paths];
	captions=[];
	for list_curr in img_paths:
		captions_curr=[];
		for path in list_curr:
			path_split=path.split('/');
			caption=path_split[-4]+' '+path_split[-3];
			captions_curr.append(caption);
			print caption
		captions.append(captions_curr);
	visualize.writeHTML(out_file_html,img_paths,captions,height=500,width=500);
	print out_file_html.replace('/disk3','vision3.cs.ucdavis.edu:1001');




	return
	params=[('/disk3/maheen_data/headC_160/withFlow_xavier_16_score/intermediate','/disk3/maheen_data/headC_160/withFlow_human_debug/correct_16','40')];
	out_file_commands_pre='/disk3/maheen_data/headC_160/withFlow_human_debug/debug_commands_16_';		
	path_to_train_file='/home/maheenrashid/Downloads/deep_proposals/torch_new/headC_160_withFlow_debug.th'

	# params=[('/disk3/maheen_data/headC_160/noFlow_gaussian_human_softmax/intermediate_res','/disk3/maheen_data/headC_160/noFlow_human_debug/correct','40'),
	# 		('/disk3/maheen_data/headC_160/noFlow_gaussian_human/intermediate','/disk3/maheen_data/headC_160/noFlow_human_debug/incorrect','56')];
	# [util.mkdir(params_curr[1]) for params_curr in params];
	# out_file_commands_pre='/disk3/maheen_data/headC_160/noFlow_human_debug/debug_commands_';
	# path_to_train_file='/home/maheenrashid/Downloads/deep_proposals/torch_new/headC_160_noFlow_debug.th'	

	model_num=range(2000,32000,6000);
	model_num.append(32000);
	print model_num
	return
	writeCommandsForTrainDebug(params,path_to_train_file,out_file_commands_pre,model_num)



	# maheen_data/headC_160/withFlow_xavier_16_score/intermediate/

	return
	dirs=['/disk3/maheen_data/headC_160/withFlow_human_debug','/disk3/maheen_data/headC_160/noFlow_human_debug'];
	# out_file_html=os.path.join(dirs[0],'comparison_dloss_seg_score.html');
	dirs=[os.path.join(dir_curr,dir_in) for dir_curr in dirs for dir_in in ['correct']];
	out_dirs=[os.path.join(dir_curr,'plots') for dir_curr in dirs];
	file_seg = 'loss_seg.npy';
	file_score = 'loss_score.npy';
	num_iters=range(2,21);
	model_num=range(5000,100000,20000);
	model_num.append(100000);
	model_num=[str(model_num_curr) for model_num_curr in model_num]	

	img_paths=[];
	captions=[];

	for dir_curr,out_dir_curr in zip(dirs,out_dirs):
		dict_for_plotting={'loss_seg_all':OrderedDict(),'loss_score_all':OrderedDict(),'loss_ratio_all':OrderedDict()};
		for model_num_curr in model_num:
			file_curr_seg=os.path.join(dir_curr,model_num_curr,file_seg);
			file_curr_score=os.path.join(dir_curr,model_num_curr,file_score);

			score_all=np.load(file_curr_score);
			score_all=score_all[[0]+range(1,len(score_all),2)];
			score_all=score_all*32


			seg_all=np.load(file_curr_seg);
			seg_all=seg_all[range(0,len(seg_all),2)];

			ratios=seg_all/score_all;
			print dir_curr,model_num_curr
			print np.mean(score_all),np.mean(seg_all),np.mean(ratios);

		# 	break;
		# break;			
				# if num_iter_curr==2:
				# 	score_all.append(np.load(file_curr_score));
				# 	seg_all.append(np.load(file_curr_seg));
				# elif num_iter_curr%2==0:
				# 	score_all.append(np.load(file_curr_score));
				# else:
				# 	seg_all.append(np.load(file_curr_seg));


				

				# seg_curr=np.load(file_curr_seg);
				# seg_curr=np.unique(np.ravel(seg_curr));
				# score_curr=list(np.load(file_curr_score));
				# score_curr=list(np.unique(np.ravel(score_curr)));
				# seg_all.extend(seg_curr);
				# score_all.extend(score_curr);

	# 		seg_all=list(set(seg_all));
	# 		score_all=list(set(score_all));
	# 		seg_all.sort();
	# 		score_all.sort();
			
	# 		dict_for_plotting['seg_all'][model_num_curr]=seg_all;
	# 		dict_for_plotting['score_all'][model_num_curr]=score_all;

	# 	img_paths_curr=[];
	# 	captions_curr=[];
	# 	for key_curr in dict_for_plotting.keys():
	# 		out_file_curr=os.path.join(out_dir_curr,key_curr+'.png');
	# 		data=dict_for_plotting[key_curr];
	# 		xAndYs=data.values();
	# 		legend_entries=data.keys();
	# 		xAndYs=[(range(len(x_curr)),x_curr) for x_curr in xAndYs];
	# 		visualize.plotSimple(xAndYs,out_file_curr,title=key_curr,xlabel='sorted idx',ylabel='values',legend_entries=legend_entries,outside=True);
	# 		print out_file_curr.replace('/disk3','vision3.cs.ucdavis.edu:1001');
	# 		img_paths_curr.append(util.getRelPath(out_file_curr,'/disk3'));
	# 		# print dir_curr.split('/');
	# 		captions_curr.append(dir_curr.split('/')[-2]+' '+dir_curr.split('/')[-1]+' '+key_curr);

	# 	img_paths.append(img_paths_curr);
	# 	captions.append(captions_curr);

	# visualize.writeHTML(out_file_html,img_paths,captions,height=200,width=200);
	# print out_file_html.replace('/disk3','vision3.cs.ucdavis.edu:1001');			

	

	return
	dirs=['/disk3/maheen_data/headC_160/withFlow_human_debug','/disk3/maheen_data/headC_160/noFlow_human_debug'];
	out_file_html=os.path.join(dirs[0],'comparison_dloss_seg_score.html');
	dirs=[os.path.join(dir_curr,dir_in) for dir_curr in dirs for dir_in in ['correct']];
	out_dirs=[os.path.join(dir_curr,'plots') for dir_curr in dirs];
	file_seg_pre='dloss_seg_';
	file_score_pre='dloss_score_';
	num_iters=range(2,21);
	model_num=range(25000,100000,20000);
	model_num.append(100000);
	model_num=[str(model_num_curr) for model_num_curr in model_num]	
	
	img_paths=[];
	captions=[];
	
	for dir_curr,out_dir_curr in zip(dirs,out_dirs):
		dict_for_plotting={'seg_all':OrderedDict(),'score_all':OrderedDict()};
		for model_num_curr in model_num:
			seg_all=[];
			score_all=[];
			for num_iter_curr in num_iters:
				file_curr_seg=os.path.join(dir_curr,model_num_curr,file_seg_pre+str(num_iter_curr)+'.npy');
				file_curr_score=os.path.join(dir_curr,model_num_curr,file_score_pre+str(num_iter_curr)+'.npy');

				seg_curr=np.load(file_curr_seg);
				seg_curr=np.unique(np.ravel(seg_curr));
				score_curr=list(np.load(file_curr_score));
				score_curr=list(np.unique(np.ravel(score_curr)));
				seg_all.extend(seg_curr);
				score_all.extend(score_curr);

			seg_all=list(set(seg_all));
			score_all=list(set(score_all));
			seg_all.sort();
			score_all.sort();
			
			dict_for_plotting['seg_all'][model_num_curr]=seg_all;
			dict_for_plotting['score_all'][model_num_curr]=score_all;

		img_paths_curr=[];
		captions_curr=[];
		for key_curr in dict_for_plotting.keys():
			out_file_curr=os.path.join(out_dir_curr,key_curr+'.png');
			data=dict_for_plotting[key_curr];
			xAndYs=data.values();
			legend_entries=data.keys();
			xAndYs=[(range(len(x_curr)),x_curr) for x_curr in xAndYs];
			visualize.plotSimple(xAndYs,out_file_curr,title=key_curr,xlabel='sorted idx',ylabel='values',legend_entries=legend_entries,outside=True);
			print out_file_curr.replace('/disk3','vision3.cs.ucdavis.edu:1001');
			img_paths_curr.append(util.getRelPath(out_file_curr,'/disk3'));
			# print dir_curr.split('/');
			captions_curr.append(dir_curr.split('/')[-2]+' '+dir_curr.split('/')[-1]+' '+key_curr);

		img_paths.append(img_paths_curr);
		captions.append(captions_curr);

	visualize.writeHTML(out_file_html,img_paths,captions,height=200,width=200);
	print out_file_html.replace('/disk3','vision3.cs.ucdavis.edu:1001');			


	# dloss_seg=np.load(file_curr_seg);
	# dloss_seg=np.mean(dloss_seg,axis=0);
	# print dloss_seg.shape;
	# dloss_seg=dloss_seg[0];
	# print dloss_seg.shape;

	# dloss_score=np.load(file_curr_score);
	# print dloss_score.shape;
	# print np.min(dloss_score);
	# print np.max(dloss_score);
	# print np.min(dloss_seg);
	# print np.max(dloss_seg);


	# # print dloss_seg[0],np.min(dloss_seg),np.max(dloss_seg);
	# print file_curr;

		

	return
	dirs=['/disk3/maheen_data/headC_160/withFlow_human_debug','/disk3/maheen_data/headC_160/noFlow_human_debug'];
	out_file_html=os.path.join('/disk3/maheen_data/headC_160/withFlow_human_debug','comparison_grads_weights_ratios.html');
	
	# dirs=['/disk3/maheen_data/headC_160/withFlow_human_debug'];
	out_file_html=os.path.join('/disk3/maheen_data/headC_160/withFlow_human_debug','comparison_grads_seg_no_seg.html');
	

	dirs=[os.path.join(dir_curr,dir_in) for dir_curr in dirs for dir_in in ['correct','incorrect']];
	out_dirs=[os.path.join(dir_curr,'plots') for dir_curr in dirs];
	[util.mkdir(out_dir_curr) for out_dir_curr in out_dirs];

	model_num=range(5000,100000,20000);
	model_num.append(100000);
	model_num=[str(model_num_curr) for model_num_curr in model_num]	
	num_iters=range(1,21);
	file_pre_weight='weight_mag_';
	file_pre_grad='grad_mag_';	

	# file_curr=os.path.join(dirs[0],model_num[-1],file_pre_grad+'1.npy');
	# grads=np.load(file_curr);
	# print grads.shape;
	# grads=grads[::2];
	# print grads.shape;
	# print grads[26:]

	# compareMagInfoLayerTime(dirs,out_dirs,model_num,num_iters,file_pre_grad,file_pre_weight,out_file_html)
	# layer_range=range(26,32);
	layer_range=[26,27,28,31];
	num_iters=range(1,21,2);
	img_paths_seg_flow=compareGradInfoLayer(dirs[:2],out_dirs[:2],model_num,num_iters,file_pre_grad,layer_range,'seg_grad_mag')
	layer_range=range(26,32);
	num_iters=range(2,21,2);
	img_paths_score_flow=compareGradInfoLayer(dirs[:2],out_dirs[:2],model_num,num_iters,file_pre_grad,layer_range,'score_grad_mag')

	# layer_range=range(13,18)
	layer_range=[13,14,17]
	num_iters=range(1,21,2);
	img_paths_seg_noflow=compareGradInfoLayer(dirs[2:],out_dirs[2:],model_num,num_iters,file_pre_grad,layer_range,'seg_grad_mag')
	layer_range=range(13,18);
	num_iters=range(2,21,2);
	img_paths_score_noflow=compareGradInfoLayer(dirs[2:],out_dirs[2:],model_num,num_iters,file_pre_grad,layer_range,'score_grad_mag')

	img_paths=[img_paths_seg_flow,img_paths_score_flow,img_paths_seg_noflow,img_paths_score_noflow];
	img_paths=[[util.getRelPath(path_curr,'/disk3') for path_curr in list_curr] for list_curr in img_paths];
	# print img_paths
	captions=[];
	# path='../../../../../../..//maheen_data/headC_160/noFlow_human_debug/incorrect/plots/score_grad_mag.png'
	for list_curr in img_paths:
		captions_curr=[];
		for path in list_curr:
			path_split=path.split('/');
			caption=path_split[-4]+' '+path_split[-3];
			captions_curr.append(caption);
			print caption
		captions.append(captions_curr);
	visualize.writeHTML(out_file_html,img_paths,captions,height=300,width=300);
	print out_file_html.replace('/disk3','vision3.cs.ucdavis.edu:1001');

	# out_files=compareGradInfoLayer(dirs,out_dirs,model_num,num_iters,file_pre_grad,layer_range)

		# break;
			

			
	return
	params=[('/disk3/maheen_data/headC_160/withFlow_gaussian_human_softmax/intermediate','/disk3/maheen_data/headC_160/withFlow_human_debug/correct','40'),
					('/disk3/maheen_data/headC_160/withFlow_gaussian_human/intermediate','/disk3/maheen_data/headC_160/withFlow_human_debug/incorrect','56')];
	out_file_commands_pre='/disk3/maheen_data/headC_160/withFlow_human_debug/debug_commands_';		
	path_to_train_file='/home/maheenrashid/Downloads/deep_proposals/torch_new/headC_160_withFlow_debug.th'

	# params=[('/disk3/maheen_data/headC_160/noFlow_gaussian_human_softmax/intermediate_res','/disk3/maheen_data/headC_160/noFlow_human_debug/correct','40'),
	# 		('/disk3/maheen_data/headC_160/noFlow_gaussian_human/intermediate','/disk3/maheen_data/headC_160/noFlow_human_debug/incorrect','56')];
	# [util.mkdir(params_curr[1]) for params_curr in params];
	# out_file_commands_pre='/disk3/maheen_data/headC_160/noFlow_human_debug/debug_commands_';
	# path_to_train_file='/home/maheenrashid/Downloads/deep_proposals/torch_new/headC_160_noFlow_debug.th'	

	model_num=range(5000,100000,20000);
	model_num.append(100000);
	writeCommandsForTrainDebug(params,path_to_train_file,out_file_commands_pre,model_num)
	
	return
	print 'hello';
	dir_debug_with_flow='/disk3/maheen_data/headC_160/withFlow_human_debug';
	dir_debug_no_flow='/disk3/maheen_data/headC_160/noFlow_human_debug';
	score_dir='score_gradient_start';
	seg_dir='seg_gradient_start';

	dirs=[os.path.join(dir_debug_no_flow,score_dir),os.path.join(dir_debug_with_flow,score_dir),
		os.path.join(dir_debug_no_flow,seg_dir),os.path.join(dir_debug_with_flow,seg_dir)]

	for dir_curr in dirs:
		np_files=util.getFilesInFolder(dir_curr,'.npy');
		np_nums=[int(file_curr[file_curr.rindex('_')+1:file_curr.rindex('.')]) for file_curr in np_files];
		sort_idx=np.argsort(np_nums);
		np_files=np.array(np_files)[sort_idx];

		gradients=getGradientMags(np_files);
		print dir_curr;
		print len(gradients),np.mean(gradients),min(gradients),max(gradients);
	# dir_curr=os.path.join(dir_debug_with_flow,score_dir);
	

	



if __name__=='__main__':
	main();

