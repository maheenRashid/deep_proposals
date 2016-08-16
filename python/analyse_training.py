import numpy as np;
import util;
import os;
import random;
import visualize;
import scipy.misc;
import subprocess;
def shortenTrainingData(train_txt,train_txt_new,ratio_txt,val_txt_new=None):
    # pos_human='/disk3/maheen_data/headC_160/noFlow_gaussian_human/pos_flos/positives_onlyHuman_withFlow.txt';
    # neg_human='/disk3/maheen_data/headC_160/neg_flos/negatives_onlyHuman_withFlow.txt';

    # pos_human_small='/disk3/maheen_data/headC_160/noFlow_gaussian_human/pos_flos/positives_onlyHuman_withFlow_oneHundreth.txt';
    # neg_human_small='/disk3/maheen_data/headC_160/neg_flos/negatives_onlyHuman_withFlow_oneHundreth.txt';

    # ratio_txt=100;
    # shortenTrainingData(pos_human,pos_human_small,ratio_txt);
    # shortenTrainingData(neg_human,neg_human_small,ratio_txt);

    train_data=util.readLinesFromFile(train_txt);
    # print ratio_txt
    if ratio_txt<1:
        ratio_txt=int(len(train_data)*ratio_txt);
        # print ratio_txt;

    random.shuffle(train_data);
    train_data_new=train_data[:ratio_txt];
    print len(train_data),len(train_data_new);
    util.writeFile(train_txt_new,train_data_new);

    if val_txt_new is not None:
        val_data=train_data[ratio_txt:];
        print len(val_data);
        util.writeFile(val_txt_new,val_data);

def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    return data;    
    # plt.imshow(data); plt.axis('off')


def convertFileToFloOnly(neg_flo,out_file_neg):
    neg_flo=util.readLinesFromFile(neg_flo);
    neg_only_flo=[];
    for neg_flo_curr in neg_flo:
        neg_flo_curr=neg_flo_curr.split(' ');
        neg_only_flo.append(neg_flo_curr[-1]+' '+neg_flo_curr[1]);

    assert len(neg_only_flo)==len(neg_flo);
    util.writeFile(out_file_neg,neg_only_flo);

def printCommandToTrain(path_to_th, model, outDir, 
                        iterations=100000,
                        learningRate=0.001,
                        saveAfter=5000,
                        batchSize=32,
                        testAfter=0,
                        dispAfter=4,
                        pos_file='/disk2/aprilExperiments/positives_160.txt',
                        neg_file='/disk2/marchExperiments/deep_proposals/negatives.txt',
                        pos_val_file='/disk2/aprilExperiments/positives_160.txt',
                        neg_val_file='/disk2/marchExperiments/deep_proposals/negatives.txt',
                        saveModel=False,
                        gpu=1):
    command = ['th'];
    command.append(path_to_th);
    command.extend(['-model',model]);
    command.extend(['-outDir',outDir]);
    command.extend(['-iterations',str(iterations)]);
    command.extend(['-dispAfter',str(dispAfter)]);
    command.extend(['-iterations',str(iterations)]);
    command.extend(['-learningRate',str(learningRate)]);
    command.extend(['-saveAfter',str(saveAfter)]);
    command.extend(['-batchSize',str(batchSize)]);
    command.extend(['-testAfter',str(testAfter)]);
    command.extend(['-dispAfter',str(dispAfter)]);
    command.extend(['-pos_file',pos_file]);
    command.extend(['-neg_file',neg_file]);
    command.extend(['-pos_val_file',pos_val_file]);
    command.extend(['-neg_val_file',neg_val_file]);
    if saveModel:
        command.append('-saveModel');
    command.extend(['-gpu',str(gpu)]);
    command.extend(['2>&1','|','tee',os.path.join(outDir,'log.txt')]);
    command=' '.join(command);
    print command+'\n';
    return command;
    

def main():

    out_dir_training_files='/disk3/maheen_data/flo_only_training_files';
    # model='/disk2/aprilExperiments/headC_160/withFlow_gaussian_xavier_fixed_unitCombo_floStumpPretrained.dat'
    # out_dir='/disk3/maheen_data/headC_160/withFlow_human_xavier_unit_floStumpPretrained_fullTraining'
    model='/disk3/maheen_data/headC_160/withFlow_human_xavier_unit_floStumpPretrained_fullTraining/intermediate/model_all_45000.dat'
    out_dir='/disk3/maheen_data/headC_160/withFlow_human_xavier_unit_floStumpPretrained_fullTraining_res'
    util.mkdir(out_dir);

    out_file_pos_train=os.path.join(out_dir_training_files,'pos_human_flo_train.txt');
    out_file_neg_train=os.path.join(out_dir_training_files,'neg_human_flo_train.txt');
    
    out_file_pos_val=os.path.join(out_dir_training_files,'pos_human_flo_val.txt');
    out_file_neg_val=os.path.join(out_dir_training_files,'neg_human_flo_val.txt');
    
    out_file_neg='/disk3/maheen_data/headC_160/neg_flos/negatives_onlyHuman_withFlow.txt'
    out_file_pos='/disk3/maheen_data/headC_160/noFlow_gaussian_human/pos_flos/positives_onlyHuman_withFlow.txt'


    # shortenTrainingData(out_file_pos,out_file_pos_train,0.9,val_txt_new=out_file_pos_val)
    # shortenTrainingData(out_file_neg,out_file_neg_train,0.9,val_txt_new=out_file_neg_val)

    learningRate=0.00001;
    path_to_th='/home/maheenrashid/Downloads/deep_proposals/torch_new/headC_160_withFlow_cl.th';
    iterations=55000;

    out_log_curr=os.path.join(out_dir,'log.txt');
    out_im_pre=os.path.join(out_dir,'loss');

    command_curr='python script_visualizeLoss.py -log_file '+'/disk3/maheen_data/headC_160/withFlow_human_xavier_unit_floStumpPretrained_fullTraining/log.txt'+' '+out_log_curr+' -out_file_pre '+out_im_pre+' -val ';
    print command_curr;
    os.system(command_curr);
    return

    command=printCommandToTrain(path_to_th,model,out_dir,iterations=iterations,learningRate=learningRate,pos_file=out_file_pos_train,
            pos_val_file=out_file_pos_val,neg_file=out_file_neg_train,neg_val_file=out_file_neg_val,testAfter=40);

    # util.writeFile(os.path.join(out_dir,'train_command.sh'),[command]);

    return
    out_dir_training_files='/disk3/maheen_data/flo_only_training_files';
    num=64;

    out_file_neg='/disk3/maheen_data/headC_160/neg_flos/negatives_onlyHuman_withFlow.txt'
    out_file_pos='/disk3/maheen_data/headC_160/noFlow_gaussian_human/pos_flos/positives_onlyHuman_withFlow.txt'


    out_file_pos_mini=os.path.join(out_dir_training_files,'pos_human_withflo_'+str(num)+'.txt')
    out_file_neg_mini=os.path.join(out_dir_training_files,'neg_human_withflo_'+str(num)+'.txt');

    # shortenTrainingData(out_file_pos,out_file_pos_mini,num);
    # shortenTrainingData(out_file_neg,out_file_neg_mini,num);


    path_to_th='/home/maheenrashid/Downloads/deep_proposals/torch_new/headC_160_withFlow_cl.th';
    model='/disk2/aprilExperiments/headC_160/withFlow_gaussian_xavier_fixed.dat';
    out_dir='/disk3/maheen_data/headC_160/withFlow_human_xavier_unit_lr_search';
    out_file_html=os.path.join(out_dir,'comparison.html');
    out_file_sh = os.path.join(out_dir,'lr_search.sh');
    util.mkdir(out_dir);

    iterations=1000;
    commands=[];
    img_paths=[];
    captions=[];
    for learningRate in [0.001,0.0001,0.00001,0.000001, 0.0000001]:
        out_dir_curr=os.path.join(out_dir,str(learningRate));
        util.mkdir(out_dir_curr);
        out_log_curr=os.path.join(out_dir_curr,'log.txt');
        out_im_pre=os.path.join(out_dir_curr,'loss');

        command_curr='python script_visualizeLoss.py -log_file '+out_log_curr+' -out_file_pre '+out_im_pre;
        print command_curr;
        os.system(command_curr);

        img_paths_curr=[util.getRelPath(out_im_pre+post,'/disk3') for post in ['_score.png','_seg.png']];
        captions_curr=[str(learningRate)+' '+file_curr for file_curr in util.getFileNames(img_paths_curr)];
        captions.append(captions_curr);
        img_paths.append(img_paths_curr);

        # command=printCommandToTrain(path_to_th,model,out_dir_curr,iterations=iterations,learningRate=learningRate,pos_file=out_file_pos_mini,
        #     pos_val_file=out_file_pos_mini,neg_file=out_file_neg_mini,neg_val_file=out_file_neg_mini);

        # commands.append(command);
        # break;

    # util.writeFile(out_file_sh,commands);
    # print 'sh '+out_file_sh

    visualize.writeHTML(out_file_html,img_paths,captions);


    return

    out_dir_training_files='/disk3/maheen_data/flo_only_training_files';
    out_dir='/disk3/maheen_data/headC_160/onlyFlow_human_xavier_fix_full';
    out_file_html=os.path.join(out_dir,'comparison_loss.html'); 
    out_file_sh=os.path.join(out_dir,'lr_commands.sh');

    util.mkdir(out_dir_training_files);
    out_file_pos=os.path.join(out_dir_training_files,'pos_human_only_flo.txt');
    out_file_neg=os.path.join(out_dir_training_files,'neg_human_only_flo.txt');

    out_file_pos_train=os.path.join(out_dir_training_files,'pos_human_only_flo_train.txt');
    out_file_neg_train=os.path.join(out_dir_training_files,'neg_human_only_flo_train.txt');
    
    out_file_pos_val=os.path.join(out_dir_training_files,'pos_human_only_flo_val.txt');
    out_file_neg_val=os.path.join(out_dir_training_files,'neg_human_only_flo_val.txt');
    
    # shortenTrainingData(out_file_pos,out_file_pos_train,0.9,val_txt_new=out_file_pos_val)
    # shortenTrainingData(out_file_neg,out_file_neg_train,0.9,val_txt_new=out_file_neg_val)

    path_to_th='/home/maheenrashid/Downloads/deep_proposals/torch_new/headC_160_noFlow_cl.th';
    model='/disk3/maheen_data/headC_160/onlyFlow_human_xavier_fix_full/1e-05/intermediate/model_all_35000.dat';

    # out_file_html=os.path.join(out_dir,'loss_comparison.html');

    iterations=65000;
    testAfter=40;
    util.mkdir(out_dir);
    commands=[];
    img_paths=[];
    captions=[];
    for learningRate in [0.000001]:
        out_dir_curr=os.path.join(out_dir,str(learningRate*10)+'_res');
        util.mkdir(out_dir_curr);
        # command_curr='python script_visualizeLoss.py '+os.path.join(out_dir_curr,'log.txt')+' '+os.path.join(out_dir_curr,'loss');
        # os.system(command_curr);
        # img_paths.append([util.getRelPath(os.path.join(out_dir_curr,'loss'+post_curr),'/disk3') for post_curr in ['_seg.png','_score.png']]);
        # captions.append([str(learningRate)+' '+file_name for file_name in util.getFileNames(img_paths[-1])]);

        command=printCommandToTrain(path_to_th,model,out_dir_curr,iterations=iterations,learningRate=learningRate,pos_file=out_file_pos_train,
            pos_val_file=out_file_pos_val,neg_file=out_file_neg_train,neg_val_file=out_file_neg_val,testAfter=testAfter);
        print command
    #     commands.append(command);

    # util.writeFile(out_file_sh,commands);
    # print out_file_sh


    return
    out_dir_training_files='/disk3/maheen_data/flo_only_training_files';
    out_dir='/disk3/maheen_data/headC_160/onlyFlow_human_xavier_fix_full';
    out_file_html=os.path.join(out_dir,'comparison_loss.html'); 
    out_file_sh=os.path.join(out_dir,'lr_commands.sh');

    util.mkdir(out_dir_training_files);
    out_file_pos=os.path.join(out_dir_training_files,'pos_human_only_flo.txt');
    out_file_neg=os.path.join(out_dir_training_files,'neg_human_only_flo.txt');

    out_file_pos_train=os.path.join(out_dir_training_files,'pos_human_only_flo_train.txt');
    out_file_neg_train=os.path.join(out_dir_training_files,'neg_human_only_flo_train.txt');
    
    out_file_pos_val=os.path.join(out_dir_training_files,'pos_human_only_flo_val.txt');
    out_file_neg_val=os.path.join(out_dir_training_files,'neg_human_only_flo_val.txt');
    
    # shortenTrainingData(out_file_pos,out_file_pos_train,0.9,val_txt_new=out_file_pos_val)
    # shortenTrainingData(out_file_neg,out_file_neg_train,0.9,val_txt_new=out_file_neg_val)

    path_to_th='/home/maheenrashid/Downloads/deep_proposals/torch_new/headC_160_noFlow_cl.th';
    model='/disk2/aprilExperiments/headC_160/noFlow_gaussian_xavier_fixed.dat';

    out_file_html=os.path.join(out_dir,'loss_comparison.html');

    iterations=100000;
    testAfter=40;
    util.mkdir(out_dir);
    commands=[];
    img_paths=[];
    captions=[];
    for learningRate in [0.00001,0.000001]:
        out_dir_curr=os.path.join(out_dir,str(learningRate));
        util.mkdir(out_dir_curr);
        command_curr='python script_visualizeLoss.py '+os.path.join(out_dir_curr,'log.txt')+' '+os.path.join(out_dir_curr,'loss');
        os.system(command_curr);
        img_paths.append([util.getRelPath(os.path.join(out_dir_curr,'loss'+post_curr),'/disk3') for post_curr in ['_seg.png','_score.png']]);
        captions.append([str(learningRate)+' '+file_name for file_name in util.getFileNames(img_paths[-1])]);

    #     command=printCommandToTrain(path_to_th,model,out_dir_curr,iterations=iterations,learningRate=learningRate,pos_file=out_file_pos_train,
    #         pos_val_file=out_file_pos_val,neg_file=out_file_neg_train,neg_val_file=out_file_neg_val,testAfter=testAfter);
    #     commands.append(command);

    # util.writeFile(out_file_sh,commands);
    # print out_file_sh

    visualize.writeHTML(out_file_html,img_paths,captions,500,500);


    return
    num=64;

    out_file_pos_mini=os.path.join(out_dir_training_files,'pos_human_only_flo_'+str(num)+'.txt')
    out_file_neg_mini=os.path.join(out_dir_training_files,'neg_human_only_flo_'+str(num)+'.txt');

    # shortenTrainingData(out_file_pos,out_file_pos_mini,num);
    # shortenTrainingData(out_file_neg,out_file_neg_mini,num);


    path_to_th='/home/maheenrashid/Downloads/deep_proposals/torch_new/headC_160_noFlow_cl.th';
    model='/disk2/aprilExperiments/headC_160/noFlow_gaussian_xavier_fixed_unitCombo.dat';

    iterations=1000;
    util.mkdir(out_dir);
    commands=[];
    img_paths=[];
    captions=[];
    for learningRate in [0.001,0.0001,0.00001,0.000001, 0.0000001]:
        out_dir_curr=os.path.join(out_dir,str(learningRate));
        util.mkdir(out_dir_curr);
        out_log_curr=os.path.join(out_dir_curr,'log.txt');
        out_im_pre=os.path.join(out_dir_curr,'loss');

        # command_curr='python script_visualizeLoss.py '+out_log_curr+' '+out_im_pre;
        # print command_curr;
        # os.system(command_curr);

        # img_paths_curr=[util.getRelPath(out_im_pre+post,'/disk3') for post in ['_score.png','_seg.png']];
        # captions_curr=[str(learningRate)+' '+file_curr for file_curr in util.getFileNames(img_paths_curr)];
        # captions.append(captions_curr);
        # img_paths.append(img_paths_curr);

        command=printCommandToTrain(path_to_th,model,out_dir_curr,iterations=iterations,learningRate=learningRate,pos_file=out_file_pos_mini,
            pos_val_file=out_file_pos_mini,neg_file=out_file_neg_mini,neg_val_file=out_file_neg_mini);
        print command;
        commands.append(command);

    # util.writeFile(out_file_sh,commands);
    # print out_file_sh

    # visualize.writeHTML(out_file_html,img_paths,captions);


    return
    dir_curr='/disk3/maheen_data/headC_160/withFlow_xavier_16_score/intermediate';
    range_files=range(2000,96000,2000);
    to_del=range(4000,94000,4000);
    for model_num in to_del:
        file_curr=os.path.join(dir_curr,'model_all_'+str(model_num)+'.dat');
        os.remove(file_curr);
        print file_curr;
        # assert os.path.exists(file_curr);



    return

    out_dir='/disk3/maheen_data/flo_only_training_files';
    util.mkdir(out_dir);
    out_file_pos=os.path.join(out_dir,'pos_human_only_flo.txt');
    out_file_neg=os.path.join(out_dir,'neg_human_only_flo.txt');

    neg_flo='/disk3/maheen_data/headC_160/neg_flos/negatives_onlyHuman_withFlow.txt';
    pos_flo='/disk3/maheen_data/headC_160/noFlow_gaussian_human/pos_flos/positives_onlyHuman_withFlow.txt';
    
    convertFileToFloOnly(neg_flo,out_file_neg)
    convertFileToFloOnly(pos_flo,out_file_pos);

    return

    pos_all=util.readLinesFromFile(pos_all);
    pos_flo=util.readLinesFromFile(pos_flo);

    neg_all=util.readLinesFromFile(neg_all);
    neg_flo=util.readLinesFromFile(neg_flo);

    print pos_all[0];
    print pos_flo[0];
    print '___';

    print neg_all[0];
    print neg_flo[0];


    return
    pos_human='/disk2/aprilExperiments/positives_160.txt';
    neg_human='/disk2/marchExperiments/deep_proposals/negatives.txt';

    pos_human_small='/disk2/aprilExperiments/positives_160_oneHundreth.txt';
    neg_human_small='/disk2/marchExperiments/deep_proposals/negatives_oneHundreth.txt';

    ratio_txt=100;
    shortenTrainingData(pos_human,pos_human_small,ratio_txt);
    shortenTrainingData(neg_human,neg_human_small,ratio_txt);


    return
    files=['/disk3/maheen_data/headC_160/withFlow_human_debug/correct/100000/im_layer_weights_1.npy',
        '/disk3/maheen_data/headC_160/withFlow_human_debug/correct/100000/flo_layer_weights_1.npy',
        '/disk3/maheen_data/headC_160/withFlow_human_debug/incorrect/100000/im_layer_weights_1.npy',
        '/disk3/maheen_data/headC_160/withFlow_human_debug/incorrect/100000/flo_layer_weights_1.npy'];
    
    for file_curr in files:
        out_dir_curr=os.path.join(file_curr.rsplit('/',2)[0],'plots');
        weights=np.load(file_curr);
        # print weights.shape;
        # weights=weights[:,:,:2,:];
        # print weights.shape
        
        # weights=weights.transpose((0,2,3,1));
        # print weights.shape
        data=vis_square(weights);
        # print data.shape
        out_file=os.path.join(out_dir_curr,util.getFileNames([file_curr],ext=False)[0]+'.png')
        print out_file;
        scipy.misc.imsave(out_file,data);
        # visualize.saveMatAsImage(data,out_file)

        # break;
        # single_filter=weights[0];

        # single_filter_t=np.transpose(single_filter);
        # single_filter_t=(single_filter_t-np.min(single_filter_t));
        # single_filter_t=single_filter_t/np.max(single_filter_t)*255;
        # print np.min(single_filter_t),np.max(single_filter_t);
        # scipy.misc.imwrite(
        # break;




    


    



if __name__=='__main__':
    main();

