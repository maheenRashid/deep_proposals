import numpy as np;
import json;
import os
import util

def main():
	preds_file='/disk2/januaryExperiments/pedro_data/coco-proposals/train.npy';
	preds=np.load(preds_file);
	print preds.shape;
	print np.min(preds[:,-2]),np.max(preds[:,-2])
	print np.unique(preds[:,0]).shape	
	print np.min(preds[np.where(preds[:,0]==preds[0,0]),-2]),np.max(preds[np.where(preds[:,0]==preds[0,0]),-2])

if __name__=='__main__':
	main();