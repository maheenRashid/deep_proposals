import util;
# import matplotlib
# import numpy as np;
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt;
# import os;
import visualize;
import sys;

def getNumFollowing(line,start_str,terminal_str):
	idx_str=line.index(start_str);
	idx_str=idx_str+len(start_str);
	num=line[idx_str:];
	if terminal_str is not None:
		num=num[:num.index(terminal_str)];
	num=float(num);
	return num;

def main(argv):
	print 'hello';
	
	# log_file='/disk3/maheen_data/headC_160/noFlow_gaussian_human/log.txt';
	# out_file_pre='/disk3/maheen_data/headC_160/noFlow_gaussian_human/loss';
	log_file=argv[1];
	out_file_pre=argv[2];
	
	start_str='minibatches processed: ';
	score_str='loss score = ';
	seg_str='loss seg = ';

	lines=util.readLinesFromFile(log_file);
	lines_rel=[line for line in lines if line.startswith(start_str)];
	scores_seg=[];
	scores_score=[];
	iterations=[];
	for line in lines_rel:
		iterations.append(getNumFollowing(line,start_str,','));
		scores_seg.append(getNumFollowing(line,seg_str,','));
		scores_score.append(getNumFollowing(line,score_str,None));

	out_file_seg=out_file_pre+'_seg.png';
	out_file_score=out_file_pre+'_score.png';

	visualize.plotSimple([(iterations,scores_score)],out_file_score,title='Score Loss at '+str(iterations[-1]),xlabel='Iterations',ylabel='Loss')
	visualize.plotSimple([(iterations,scores_seg)],out_file_seg,title='Seg Loss at '+str(iterations[-1]),xlabel='Iterations',ylabel='Loss')

	# ,legend_entries=None,loc=0,outside=False)


# 	minibatches processed:  17620, loss seg = 30.920782, loss score = 0.212187
# minibatches processed:  17624, loss seg = 26.446381, loss score = 0.091342



if __name__=='__main__':
	main(sys.argv);
