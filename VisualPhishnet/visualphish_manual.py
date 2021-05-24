import visualphish_main
import visualphish_manual
import os
import argparse
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--folder", help='Input the folder of files to parse', required=True)
    parser.add_argument('-r', "--results", help='Input the folder of files to parse', required=True)
    args = parser.parse_args()
    path = args.folder
    if not os.path.exists(args.results):
        os.makedirs(args.results)
    results = os.path.join(args.results, 'results.txt')

    visualphish_main.main(path, results)

    ## Example: python visualphish_manual.py --folder ..\\backup\\DATABASE\\260720 --results .



