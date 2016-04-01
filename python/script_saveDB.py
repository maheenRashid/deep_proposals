import h5py
import numpy as np;
import util;
import scipy.misc;

def reshapeForDB(im_curr,canoncial_shape,addDim):

	if im_curr.shape[:2] != (canoncial_shape[0],canoncial_shape[1]):
		print im_curr.shape,canoncial_shape
		im_curr = scipy.misc.imresize(im_curr,canoncial_shape);

	if addDim:
		im_curr=np.expand_dims(im_curr,3);

	# print im_curr.shape
	
	im_curr=np.transpose(im_curr,(2,0,1));
	return im_curr;

def main():
	pos_data_txt='/disk2/februaryExperiments/deep_proposals/positive_data.txt';
	num_to_keep=100;
	out_file_db='/disk2/februaryExperiments/deep_proposals/positive_data_'+str(num_to_keep)+'.hdf5';

	lines=util.readLinesFromFile(pos_data_txt);
	ims=[];
	masks=[];
	for line in lines:
		idx_space=line.index(' ');
		im=line[:idx_space];
		mask=line[idx_space+1:];
		ims.append(im);
		masks.append(mask);
	
	if num_to_keep is None:
		num_to_keep=len(ims);

	ims=ims[:num_to_keep];
	masks=masks[:num_to_keep];

	f=h5py.File(out_file_db,'w')
	
	canoncial_shape=(240,240);	
	dset = f.create_dataset('images', (len(ims),3,canoncial_shape[0],canoncial_shape[1]), dtype='i')
	dset = f.create_dataset('masks', (len(masks),1,canoncial_shape[0],canoncial_shape[1]), dtype='i')


	for idx in range(len(ims)):
		print idx
		img=scipy.misc.imread(ims[idx]);
		# print img.shape
		if len(img.shape)<3:
			img=np.dstack((img,img,img))
		# print img.shape
		f['/images'][idx,...]=reshapeForDB(img,canoncial_shape,addDim=False);		
		f['/masks'][idx,...]=reshapeForDB(scipy.misc.imread(masks[idx]),canoncial_shape,addDim=True);
		# print np.min(im_curr),np.max(im_curr);
		# print im_curr.shape

		# mask_curr=scipy.misc.imread(masks[idx]);
		# mask_curr=np.expand_dims(mask_curr,3)
		# mask_curr=np.transpose(mask_curr,(2,0,1));
		# print np.min(mask_curr),np.max(mask_curr);
		# print mask_curr.shape

		# f['/images'][idx,...]=im_curr;
		# f['/masks'][idx,...]=mask_curr;

	f.close();
	print len(ims);
	print ims[:3];
	print len(masks);
	print masks[:3];



if __name__=='__main__':
	main();