import shutil
import os

PATH = "/mydata/Data_Copy/"
DEST = "/mydata/Synthetic_25"

DEST_SUB_DIRS = os.listdir(DEST)

for subdir in os.listdir(PATH):
	for file in os.listdir(PATH/subdir)[:(len(os.listdir(PATH/subdir))*0.75)]:
		current_dir = DEST_SUB_DIRS[os.listdir(PATH).index(subdir)]
		shutil.move(PATH/subdir/file, DEST/DEST_SUB_DIRS)
		
