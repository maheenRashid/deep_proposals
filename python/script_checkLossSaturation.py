
import numpy as np;
import visualize;
def smoothOutLoss(loss,step_size):
	range_check=range(0,loss.size,step_size);
	range_check.append(loss.size);
	# print range_check
	avg_loss=[];
	for idx_idx,idx_begin in enumerate(range_check[:-1]):
		idx_end=range_check[idx_idx+1];
		loss_curr=loss[idx_begin:idx_end];
		avg_curr=np.mean(loss_curr);
		avg_loss.append(avg_curr);
	# 	print idx_begin,idx_end
	# print len(avg_loss);
	return avg_loss;


def main():
	loss_seg_path='/disk3/maheen_data/headC_160/withFlow_gaussian_human_softmax/final/loss_all_final_seg.npy';
	loss_score_path='/disk3/maheen_data/headC_160/withFlow_gaussian_human_softmax/final/loss_all_final_score.npy';

	loss_seg_path_res = '/disk3/maheen_data/headC_160/withFlow_gaussian_human_softmax_res/final/loss_all_final_seg.npy';
	loss_score_path_res = '/disk3/maheen_data/headC_160/withFlow_gaussian_human_softmax_res/final/loss_all_final_score.npy';

	out_file_seg='/disk3/maheen_data/headC_160/withFlow_gaussian_human_softmax/final/loss_all_final_seg.png';
	out_file_score='/disk3/maheen_data/headC_160/withFlow_gaussian_human_softmax/final/loss_all_final_score.png';


	step_size=100;
	loss_seg=np.load(loss_seg_path);
	loss_score=np.load(loss_score_path);
	
	loss_seg=np.concatenate((loss_seg,np.load(loss_seg_path_res)));
	loss_score=np.concatenate((loss_score,np.load(loss_score_path_res)));
	print loss_seg.shape
	print loss_score.shape
	avg_seg=smoothOutLoss(loss_seg,step_size);
	avg_score=smoothOutLoss(loss_score,step_size);
	avg_score=avg_score[100:];
	avg_seg=avg_seg[100:];
	visualize.plotSimple([(range(len(avg_seg)),avg_seg)],out_file_seg,title='seg loss avg')
	visualize.plotSimple([(range(len(avg_score)),avg_score)],out_file_score,title='score loss avg')

	print out_file_score.replace('/disk3','vision3.cs.ucdavis.edu:1001');



if __name__=='__main__':
	main();