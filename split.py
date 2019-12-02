import split_folders
import argparse

parser = argparse.ArgumentParser(description='Basic Arg parser')
parser.add_argument("--input", default=1, type=str, help="Input Directory",required = True)
parser.add_argument("--output", default=1, type=str, help="Output Directory",required = True)
parser.add_argument("--ratio", nargs='+',help="Train/val/test or train/val",required = True)

args = parser.parse_args()
indir = args.input
outdir = args.output
if len(args.ratio) == 3:
    ratio = (float(args.ratio[0]),float(args.ratio[1]),float(args.ratio[2]))
else:
        ratio = (float(args.ratio[0]),float(args.ratio[1]))


split_folders.ratio(indir,output = outdir, seed = 1337, ratio = ratio)
