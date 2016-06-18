from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
import cPickle as pickle;

# results=pickle.load(open('/disk2/temp/recall_check.p','rb'));
# print results.keys();
# recall=results['recall'];
# print recall.shape

# # rec= recall[9,...];
# rec=np.mean(recall,axis=0);
# print rec.shape
# for i in range(rec.shape[1]):
# 	print rec[:,i,9],rec[:,i,99],rec[:,i,-1]



# def temp():

pylab.rcParams['figure.figsize'] = (10.0, 8.0)


annType = ['segm','bbox']
annType = annType[1]      #specify type here
print 'Running demo for *%s* results.'%(annType)

# initialize COCO ground truth api
dataDirGT='/disk2/ms_coco'
dataType='val2014'
annFile = '%s/annotations/instances_%s.json'%(dataDirGT,dataType)
cocoGt=COCO(annFile)

dataDir='..'
resFile='%s/results/instances_%s_fake%s100_results.json'
resFile='/home/maheenrashid/Downloads/deep_proposals/coco-master/results/instances_val2014_maheen_results.json'
# resFile = resFile%(dataDir, dataType, annType)
print len(resFile);
cocoDt=cocoGt.loadRes(resFile)


imgIds=sorted(cocoGt.getImgIds())
imgIds=imgIds[:5000]
print cocoDt[0],cocoDt.keys();
raw_input();

# print imgIds

cocoEval = COCOeval(cocoGt,cocoDt)
cocoEval.params.imgIds  = imgIds
cocoEval.params.useSegm = 0
cocoEval.params.useCats = 0;
cocoEval.params.maxDets = range(1000);
cocoEval.params.recThrs = [1];

# imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  useSegm    - [1] if true evaluate against ground-truth segments
    #  useCats    - [1] if true use category labels for evaluation    # Note: if useSegm=0 the evaluation is run on bounding boxes.
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.


# cocEval.params.
cocoEval.evaluate()
cocoEval.accumulate()
print cocoEval.eval.keys();

print cocoEval.eval['recall'].shape
pickle.dump(cocoEval.eval,open('/disk2/temp/recall_check_5000.p','wb'));

# for k in cocoEval.eval.keys():
# 	print k,cocoEval.eval[k].shape

# cocoEval.summarize()