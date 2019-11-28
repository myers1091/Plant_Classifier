import split_folders
import argparse

parser = argparse.ArgumentParser(description='Basic Arg parser')
parser.add_argument("--input", default=1, type=str, help="Input Directory",required = True)
parser.add_argument("--output", default=1, type=str, help="Output Directory",required = True)
parser.add_argument("--ratio", nargs='+',help="Split ratio, default 0.8,0.1,0.1",required = True)

args = parser.parse_args()
indir = args.input
outdir = args.output
ratio = (float(args.ratio[0]),float(args.ratio[1]),float(args.ratio[2]))

split_folders.ratio(indir,output = outdir, seed = 1337, ratio = ratio)
