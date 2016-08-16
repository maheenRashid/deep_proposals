import util;
import numpy as np;
import os;
import shutil;
import multiprocessing;
import time;


def copyfile_wrapper((in_file,out_file,counter)):
	if counter%100==0:
		print counter;
	shutil.copyfile(in_file,out_file);

def copyFilesForTransfer(col_data,dirs_col):

	new_col_data=[];
	for idx,col_curr in enumerate(col_data):
		dir_curr=dirs_col[idx];
		util.mkdir(dir_curr);
		file_names=util.getFileNames(col_curr,ext=True);
		out_files=[os.path.join(dir_curr,file_name) for file_name in file_names];

		new_col_data.append(out_files);

		args=[];
		counter=0;
		for in_file,out_file in zip(col_curr,out_files):
			if not os.path.exists(out_file):
				args.append((in_file,out_file,counter));
				counter+=1;

		
		print len(args);
		t=time.time();
		p=multiprocessing.Pool(multiprocessing.cpu_count())
		p.map(copyfile_wrapper,args);
		print time.time()-t;

		

	# assert len(new_col_data)==len(col_data);
	
	# for idx,new_col_data_curr in enumerate(new_col_data):
	# 	print idx,len(new_col_data_curr),len(col_data[idx])
	
	return new_col_data
	
def modify_path(pos_data,pos_paths):
	pos_data_new=[];

	for pos_path_curr in pos_data:
		pos_files=[file_curr[file_curr.rindex('/')+1:] for file_curr in pos_path_curr.split(' ')]
		pos_files=[os.path.join(pos_paths[idx],file_curr) for idx,file_curr in enumerate(pos_files)];
		pos_files=' '.join(pos_files);
		pos_data_new.append(pos_files);

	return pos_data_new;

def main():
	
	neg_file='/disk3/maheen_data/headC_160/neg_flos/negatives_onlyHuman_withFlow.txt'
	pos_file='/disk3/maheen_data/headC_160/noFlow_gaussian_human/pos_flos/positives_onlyHuman_withFlow.txt'

	out_dir_meta='/disk3/maheen_data/transfer_to_vision1';
	out_dir_pos=os.path.join(out_dir_meta,'pos_data_human');
	out_dir_neg=os.path.join(out_dir_meta,'neg_data_human');
	util.mkdir(out_dir_meta);
	util.mkdir(out_dir_pos);
	util.mkdir(out_dir_neg);

	neg_data=util.readLinesFromFile(neg_file);
	pos_data=util.readLinesFromFile(pos_file);
	
	pos_file_out=os.path.join(out_dir_meta,'pos_data.txt')
	neg_file_out=os.path.join(out_dir_meta,'neg_data.txt');
	
	neg_path_meta='/home/SSD3/maheen-data/flo_pos_neg_data/neg_data_human';
	pos_path_meta='/home/SSD3/maheen-data/flo_pos_neg_data/pos_data_human';

	neg_paths=[os.path.join(neg_path_meta,dir_curr) for dir_curr in ['im','np','flo']];
	pos_paths=[os.path.join(pos_path_meta,dir_curr) for dir_curr in ['im','mask','flo']];

	pos_paths_new=modify_path(pos_data,pos_paths);
	neg_paths_new=modify_path(neg_data,neg_paths);
	
	util.writeFile(neg_file_out,neg_paths_new);
	util.writeFile(pos_file_out,pos_paths_new);


	# pos_paths_old=pos_data[0];
	# pos_paths_old=pos_paths_old.split(' ');
	# pos_paths_old=[path[:path.rindex('/')] for path in pos_paths_old];
	# print pos_paths_old;


	return
	# pos_data=pos_data[:101];
	# neg_data=neg_data[:101];
	
	pos_data=util.getColumnsFromLines(pos_data);
	neg_data=util.getColumnsFromLines(neg_data);

	pos_dirs=['im','mask','flo'];
	neg_dirs=['im','np','flo'];
	
	pos_dirs=[os.path.join(out_dir_pos,dir_curr) for dir_curr in pos_dirs];
	neg_dirs=[os.path.join(out_dir_neg,dir_curr) for dir_curr in neg_dirs];

	pos_data = copyFilesForTransfer(pos_data,pos_dirs);
	neg_data = copyFilesForTransfer(neg_data,neg_dirs);
	
	# shutil.copy_file(in_file,out_file);

	# print len(neg_data);
	# for i in range(len(neg_data)):
	# 	print neg_data[i][0];

	# print len(pos_data)
	# for i in range(len(pos_data)):
	# 	print pos_data[i][0];
	




if __name__=='__main__':
	main();
