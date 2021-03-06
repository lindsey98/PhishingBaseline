from PIL import Image
from collections import Counter
import numpy as np
from math import sqrt
import os
import operator
import datetime
import argparse
from utils import brand_converter

# Define parameters
w = h = 100
s = 20
CDF = 32
p = q = 0.5


class Emd:#自定义的元素
    def __init__(self,emd,targetlist_name):
        self.emd = emd
        self.targetlist_name = targetlist_name

def get_signature(path):
    img = Image.open(path)
    img = img.resize((w, h), Image.ANTIALIAS)
    if img.mode == 'RGBA':
        r, g, b, a = img.split()
    else:
        img = img.convert("RGBA")
        r, g, b, a = img.split()
    # RGBA
    RGBA = []
    pixel = []
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            pixel.append((i, j))
            RGBA.append((r.getpixel((i, j)) // CDF,
                         g.getpixel((i, j)) // CDF,
                         b.getpixel((i, j)) // CDF,
                         a.getpixel((i, j)) // CDF))
    # Centroid
    Ss = Counter(RGBA).most_common(s)
    signature = []
    # max_rgba
    r = []
    g = []
    b = []
    a = []
    for item in Ss:
        Cdcx = 0
        Cdcy = 0
        dc = item[0]
        r.append(dc[0])
        g.append(dc[1])
        b.append(dc[2])
        a.append(dc[3])
        for i, rgba in enumerate(RGBA):
            if rgba == dc:
                Cdcx += pixel[i][0]
                Cdcy += pixel[i][1]
        Cdc = (Cdcx/item[1], Cdcy/item[1])
        Fdc = (dc, Cdc)
        signature.append((Fdc, item[1]))
    md_color = sqrt(pow(max(r), 2)+pow(max(g), 2)+pow(max(b), 2)+pow(max(a), 2))
    return signature, md_color

def get_feature(signatureA, signatureB, md_colorA, md_colorB):
    md_color = max(md_colorA, md_colorB)
    md_centroid = sqrt(w*h)
    dis_color = np.zeros((s, s), dtype=np.float)
    dis_centroid = np.zeros((s, s), dtype=np.float)
    emd = 0
    for i, pixA in enumerate(signatureA):
        colorA = pixA[0][0]
        centroidA = pixA[0][1]
        for j, pixB in enumerate(signatureB):
            colorB = pixB[0][0]
            centroidB = pixB[0][1]
            color = (colorA[0]-colorB[0], colorA[1]-colorB[1], colorA[2]-colorB[2], colorA[3]-colorB[3])
            centroid = (centroidA[0]-centroidB[0], centroidA[1]-centroidB[1])
            dis_color[i][j] = sqrt(np.dot(color, color))
            dis_centroid[i][j] = sqrt(np.dot(centroid, centroid))
    dis_color /= md_color
    dis_centroid /= md_centroid
    dis = p*dis_color + q*dis_centroid
    for i in range(s):
        mind = np.min(dis[i], axis=0)
        ind = np.where(dis[i] == mind)
        dis = np.delete(dis, ind[0], axis=1)
        emd += mind
    emd /= s
    if emd > 0.3:
        emd *= 2
    elif emd < 0.3:
        emd /= 2
    if 1-emd < 0:
        return 0
    return 1 - emd

def main(data_folder, mode, outfile, targetlist):

    assert mode in ['phish', 'benign']  ## must specify mode is for phish/benign
    # emd_data = [0.94, 0.93, 0.92, 0.91, 0.9, 0.89, 0.88, 0.87]  ## threshold to use
    emd_data = [0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    N = [1, 3, 5, 10] ## top1/3/5/10
    result = []

    ## cache features for targetlist screenshots
    signatureB_list = []
    md_colorB_list = []
    tar_list = []
    for roottar, dirstar, filestar in os.walk(targetlist):
        for tar in dirstar:
            img2url = os.path.join(roottar, tar, "loginpage.png")
            if not os.path.exists(img2url):
                img2url = os.path.join(roottar, tar, "homepage.png")
            signatureB_this, md_colorB_this = get_signature(img2url)
            signatureB_list.append(signatureB_this)
            md_colorB_list.append(md_colorB_this)
            tar_list.append(tar)

    assert len(signatureB_list) == len(md_colorB_list) ## assert singature list and md_color list must have the same length
    assert len(signatureB_list) == len(tar_list) ## each target brand get 1 feature vector

    for emd_ in emd_data:
        if os.path.exists(os.path.join(outfile, 'emd_30k_'+str((round(emd_, 2)))+'_' + mode + '.txt')):
            file = os.path.join(outfile, 'emd_30k_'+str((round(emd_, 2)))+'_' + mode + '.txt')
            f = open(file, 'a+')
        else:
            file = os.path.join(outfile, 'emd_30k_'+str((round(emd_, 2)))+'_' + mode + '.txt')
            f = open(file, 'w')

        count = len(os.listdir(data_folder))
        detection = 0
        identi = np.zeros((1, 4))

        for dir_ in os.listdir(data_folder):
            if mode == 'phish':
                phishname = dir_[:dir_.find("+")]  ##gt brand name
            else:
                phishname = dir_[:dir_.find(".")] ##for benign get the domain

            ## open the screenshot
            img1url = os.path.join(data_folder, dir_, "shot.png")
            if img1url in open(file).read():
                continue
            f.write(img1url + "\t")
            f.flush()

            count_emd = []
            starttime = datetime.datetime.now()
            signatureA, md_colorA = get_signature(img1url)

            for i in range(len(signatureB_list)):
                signatureB = signatureB_list[i]
                md_colorB = md_colorB_list[i]
                tar = tar_list[i]
                try:
                    emd = get_feature(signatureA, signatureB, md_colorA, md_colorB)
                    if emd > emd_:  # Find all brands that exceeds similarity threshold
                        count_emd.append(Emd(emd, tar))
                except Exception as e:
                    print(e)
                    print("Cannot compare features for: ", tar, ' and ', dir_)

            # if prediction is non-empty
            if len(count_emd) != 0:
                f.write("True\t")
                detection += 1 ## detected as phish
                if mode == 'phish':
                    ## sort according to similarity
                    cmpfun = operator.attrgetter('emd', 'targetlist_name')
                    count_emd.sort(key=cmpfun, reverse=True)
                    for i, n in enumerate(N):
                        target_list = count_emd[:n]
                        for target in target_list:
                            if brand_converter(phishname) == brand_converter(target.targetlist_name):
                                f.write(str(n) + "\t")
                                f.flush()
                                identi[0][i] += 1  ## detected as the ground-truth brand
                                break
            f.write(str(datetime.datetime.now() - starttime))  # record the time for 1 screenshot
            f.write("\n")
            f.flush()
        result.append([("emd", emd_), ("count", count), ("detec", detection), ("identi", identi)])
        print(result)
        f.close()
    print('Final result:', result)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--data_folder", help='Website folder', required=True)
    parser.add_argument('-m', "--mode", help='Mode of testing: phish|benign', required=True)
    parser.add_argument('-o', '--output_basedir', help='Output text file', default='TestResult')
    parser.add_argument('-t', '--targetlist', help='Path to targetlist folder', required=True)
    args = parser.parse_args()

    if not os.path.exists(args.output_basedir):
        os.mkdir(args.output_basedir)

    main(args.data_folder, args.mode, args.output_basedir, args.targetlist)
