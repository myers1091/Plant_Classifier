import glob
import re
import os
import shutil

categories = os.listdir('./dataset/images/field/')

newtop = './combinedData'
try:
    os.mkdir(newtop)
except OSError:
    print('Directory Exists')  
else:
    print('Successfully created the directory %s' % newtop)
for cat in categories:
    newdir = newtop+ '/' + cat
    try:
        os.mkdir(newdir)
    except OSError:
        print('Directory Exists')  
    else:
        print('Successfully created the directory %s' % newdir)

    for files in glob.glob('./dataset/images/**/' + cat +'/*'):
        newfile = os.path.basename(files)
        shutil.copyfile(files,newdir +'/'+ newfile)    
    