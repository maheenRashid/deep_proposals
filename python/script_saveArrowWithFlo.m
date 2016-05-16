
cd /home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction_test/examples/opticalflow
path_to_text='/disk2/aprilExperiments/deep_proposals/flow_neg/flo_subset_for_pos_cropped/files_for_mat.txt'
fid=fopen(path_to_text);
tline=fgetl(fid);
while ischar(tline)
	spl=regexp(tline,' ','split')
	try
		loadResultsFromMat(spl{1},spl{2},spl{3});
	catch
		fprintf('error\n');	
	end
	tline=fgetl(fid);
end
