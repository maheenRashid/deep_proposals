import util;
# import matplotlib
# import numpy as np;
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt;
# import os;
import visualize;
import sys;
import argparse;


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
	
	parser=argparse.ArgumentParser(description='Visualize Loss');
	parser.add_argument('-log_file', dest='log_file',type=str, nargs='+',
	                    help='log file(s) to parse')
	parser.add_argument('-out_file_pre', dest='out_file_pre',type=str,
                     help='loss file pre path. will be appended with _seg and _score')

	parser.add_argument('-val',dest='val',action='store_true',help='to plot val loss');

	args = parser.parse_args(argv)
	
	# print args;

	# # log_file='/disk3/maheen_data/headC_160/noFlow_gaussian_human/log.txt';
	# # out_file_pre='/disk3/maheen_data/headC_160/noFlow_gaussian_human/loss';
	# getopt.getopt(args, options[, long_options])

	log_file = args.log_file
	# argv[1];
	out_file_pre = args.out_file_pre

	# argv[2];
	out_file_seg=out_file_pre+'_seg.png';
	out_file_score=out_file_pre+'_score.png';

	start_str='minibatches processed: ';
	
	lines_all=[];
	last_idx_all=[];
	for log_file_curr in log_file:
		lines=util.readLinesFromFile(log_file_curr);
		lines_all.append(lines);
		lines_rev=lines[::-1];
		for line_curr in lines_rev:
			if line_curr.startswith(start_str):
				last_idx_all.append(getNumFollowing(line_curr,start_str,','));
				break;

	assert len(lines_all)==len(last_idx_all);

	
	scores_seg=[];
	scores_score=[];
	iterations=[];
	scores_seg_val=[];
	scores_score_val=[];
	iterations_val=[];

	for lines_idx,lines in enumerate(lines_all):
		
		score_str=', loss score = ';
		seg_str=', loss seg = ';

		if lines_idx==0:
			to_add=0;
		else:
			to_add=last_idx_all[lines_idx-1];
		
		lines_rel=[line for line in lines if line.startswith(start_str) and score_str in line];

		for line in lines_rel:
			iterations.append(getNumFollowing(line,start_str,',')+to_add);
			scores_seg.append(getNumFollowing(line,seg_str,','));
			scores_score.append(getNumFollowing(line,score_str,None));

		if args.val==True:
			score_str=', val loss score = ';
			seg_str=', val loss seg = ';

			lines_rel=[line for line in lines if line.startswith(start_str) and score_str in line];

			for line in lines_rel:
				iterations_val.append(getNumFollowing(line,start_str,',')+to_add);
				scores_seg_val.append(getNumFollowing(line,seg_str,','));
				scores_score_val.append(getNumFollowing(line,score_str,None));


	if args.val==False:
		visualize.plotSimple([(iterations,scores_score)],out_file_score,title='Score Loss at '+str(iterations[-1]),xlabel='Iterations',ylabel='Loss')
		visualize.plotSimple([(iterations,scores_seg)],out_file_seg,title='Seg Loss at '+str(iterations[-1]),xlabel='Iterations',ylabel='Loss')
	else:
		visualize.plotSimple([(iterations,scores_score),(iterations_val,scores_score_val)],out_file_score,title='Score Loss at '+str(iterations[-1]),xlabel='Iterations',ylabel='Loss',legend_entries=['Train','Val'])
		visualize.plotSimple([(iterations,scores_seg),(iterations_val,scores_seg_val)],out_file_seg,title='Seg Loss at '+str(iterations[-1]),xlabel='Iterations',ylabel='Loss',legend_entries=['Train','Val'])




# 	minibatches processed:  17620, loss seg = 30.920782, loss score = 0.212187
# minibatches processed:  17624, loss seg = 26.446381, loss score = 0.091342



if __name__=='__main__':
	main(sys.argv[1:]);
