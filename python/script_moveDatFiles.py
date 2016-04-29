import os;
import subprocess;
import util;

def main():
	text_list='/disk2/aprilExperiments/dual_flow/list_of_dats_to_move.txt';
	text_mv='/disk2/aprilExperiments/dual_flow/list_of_dats_to_move_commands.sh';
	models=util.readLinesFromFile(text_list);
	path_to_storage='/media/maheenrashid/Seagate\ Backup\ Plus\ Drive/maheen_data';
	path_to_replace='/disk2';
	
	mv_commands=[];

	for model in models:
		if not os.path.exists(model):
			continue;
		dir_curr=model[:model.rindex('/')];
		dir_new=dir_curr.replace(path_to_replace,path_to_storage);

		# print dir_new;
		
		command='mkdir -p '+dir_new;
		# print command;
		mv_command='mv -v '+model+' '+dir_new+'/';
		# print mv_command
		mv_commands.append(mv_command);
		subprocess.call(command,shell=True);
		# raw_input();

	util.writeFile(text_mv,mv_commands);
	print text_mv


if __name__=='__main__':
	main();