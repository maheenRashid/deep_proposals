import numpy as np;
import os;
import util;

def main():
	print 'hello';
	path_to_debug='/disk3/maheen_data/headC_160/noFlow_gaussian_human_softmax/debug';
	# iter_to_explore=[40,41];
	files=util.getFileNames(util.getFilesInFolder(path_to_debug,'36.npy'),ext=False);
	score_dict={};
	for file_curr in files:
		file_curr_split=file_curr.split('_');
		type_output=file_curr_split[-2];
		score_dict[type_output]=np.load(os.path.join(path_to_debug,file_curr+'.npy'));

	for key_curr in score_dict.keys():
		print key_curr,score_dict[key_curr].shape;

	keys_to_explore=['inputLabel','dloss','output'];
	for key_curr in keys_to_explore:
		print key_curr;
		print score_dict[key_curr]

if __name__=='__main__':
	main();

