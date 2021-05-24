import pandas as pd
import os

def create_urllist(folder, mode):
    url_list = []
    label = []
    for file in os.listdir(folder):
        url_path = folder + '/' + file + '/' + 'info.txt'
        if mode == 'phish':
            url_list.append(eval(open(url_path, 'r', encoding= "ISO-8859-1").read())['url'])
            label.append(1)
        else:
            try:
                url_list.append(open(url_path, 'r').read())
            except:
                url_list.append('https://www.'+file)
            label.append(0)

    assert len(url_list) == len(label)
    print(len(url_list))

    f = open('./experiment/URLnet/test_'+mode+'.txt', 'w')
    for i in range(len(url_list)):
        f.writelines(str(label[i]) + '\t' + url_list[i] + '\n')
    f.close()

if __name__ == '__main__':
    '''data preparation'''
    create_urllist('./benchmark/test15k_wo_localcontent/phish_sample_15k', 'phish')
    create_urllist('./benchmark/test15k_wo_localcontent/benign_sample_15k', 'benign')
