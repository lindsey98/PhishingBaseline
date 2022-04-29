import os
import cv2 as cv
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import datetime
import argparse
from tqdm import tqdm


class TokenSite():
    '''Tokenize html and url'''
    def __init__(self, html, url):  ##input the html and the url
        self.html = html
        self.url = url

    def token_keyword_html(self):
        if len(self.html) == 0:
            return ''
        text = self.html

        # kill all script and style elements
        for script in text(["script", "style"]):
            script.extract()  # rip it out

        # get text
        text = text.get_text()
        text = re.sub(r'[`\=@©#$%^*()_+\[\]{};\'\\:"|<,./<>?‘’-]', ' ', ''.join(text.splitlines()))
        text = text.replace(u'\xa0', u' ')
        text = text.split()
        for txt in text:
            if len(txt) < 3:
                text.remove(txt)
        if len(text) == 0:
            return ''
        else:
            return ' '.join(text)

    def token_keyword_url(self):
        token_list = re.split(r'[`\=@©#$%^*()_+\[\]{};\'\\:"|<,./<>?‘’-]', self.url)
        for token in token_list:
            if ('www' in token) or ('http' in token) or ('com' in token) or token == '':
                token_list.remove(token)
        if len(token_list) == 0:
            return ''
        else:
            return ' '.join(token_list)

    def output(self):
        html_tok = self.token_keyword_html()
        url_tok = self.token_keyword_url()
        final_tok = html_tok
        final_tok = final_tok + url_tok
        return final_tok

def getMatchNum(matches, ratio):
    '''return number of matched keypoints'''
    matchesMask = [[0, 0] for i in range(len(matches))]
    matchNum = 0
    for i, (m, n) in enumerate(matches):
        if m.distance < ratio * n.distance:  # compute good matches
            matchesMask[i] = [1, 0]
            matchNum += 1
    return (matchNum, matchesMask)

'''SIFT extractor'''
class SIFT():

    def __init__(self, login_path, brand_folder):
        self.login_path = login_path
        self.brand_folder = brand_folder
        self.logo_kp_list, self.logo_des_list, self.logo_file_list = self.logo_kp()
        assert len(self.logo_kp_list) > 0
        assert len(self.logo_kp_list) == len(self.logo_des_list)
        assert len(self.logo_file_list) == len(self.logo_des_list)

    def match(self):
        # construct sift extractor
        sift = cv.xfeatures2d.SIFT_create()
        # FLANN match
        FLANN_INDEX_KDTREE = 0
        indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        searchParams = dict(checks=50)
        flann = cv.FlannBasedMatcher(indexParams, searchParams)

        # extract webpage kp
        try:
            queryImage = cv.imread(self.login_path, cv.IMREAD_GRAYSCALE)
            kp, des = sift.detectAndCompute(queryImage, None)
        except Exception as e:
            print(e)
            print('Cannot identify the screenshot')
            return self.brand_folder.split('/')[-1], None, 0

        similarity = []
        filename = []
        for i in range(len(self.logo_kp_list)):
            # extract kp from logo
            logo_kp = self.logo_kp_list[i]
            logo_des = self.logo_des_list[i]
            try:
                matches = flann.knnMatch(logo_des, des, k=2)  # match keypoint
                (matchNum, matchesMask) = getMatchNum(matches, 0.9)  # calculate matchratio
                matchRatio = matchNum * 100 / len(logo_kp)
            except:
                matchRatio = 0
            similarity.append(matchRatio)
            filename.append(self.logo_file_list[i])
            del matchRatio ## delete matchRatio and go to next round

        maxfilename = np.array(filename)[np.argsort(similarity)[::-1]][0] ## which logo gives the maximum similarity
        maxscore = max(similarity) ## maximum similarity
        return self.brand_folder.split('/')[-1], maxfilename, maxscore

    def logo_kp(self):
        sift = cv.xfeatures2d.SIFT_create()
        img_kp_list = []
        img_des_list = []
        img_file_list = []
        for file in os.listdir(self.brand_folder):
            if not file.startswith('loginpage') and not file.startswith('homepage'):
                try:
                    img = cv.imread(self.brand_folder + '/' + file, cv.IMREAD_GRAYSCALE) ## convert to grayscale
                    kp, des = sift.detectAndCompute(img, None)
                    img_kp_list.append(kp)
                    img_des_list.append(des)
                    img_file_list.append(self.brand_folder + '/' + file)
                except Exception as e:
                    print(e)
        return img_kp_list, img_des_list, img_file_list

def dict_construct(target_root, domain_map_path):
    '''dictionary constructor'''
    with open(domain_map_path, 'rb') as handle:
        domain_map = pickle.load(handle)
    # print(domain_map)

    sample_dir_list = []
    for folder in os.listdir(target_root):
        sample_dir = target_root + '/' + folder + '/' + 'homepage_html.txt'
        sample_dir_list.append(sample_dir)

    ground_brand = os.listdir(target_root)
    ground_domain = [domain_map[brand_converter(x)][0] for x in ground_brand]
    ground_html = []
    ## extract html content
    for i in tqdm(range(len(sample_dir_list))):
        try:
            with open(sample_dir_list[i]) as handle:
                soup = BeautifulSoup(handle.read(), "html5lib")
                if len(soup) == 0:
                    ground_html.append('')
                else:
                    try:
                        ground_html.append(soup)
                    except:
                        ground_html.append('')
        except:
            ground_html.append('')

    print('Reading completed...')
    assert len(ground_html) == len(ground_domain)

    token_list = []
    for i in tqdm(range(len(ground_domain))):
        if len(ground_html[i]) != 0:
            token_this = TokenSite(ground_html[i], ground_domain[i])
            token_list.append(token_this.output())
        else:
            token_list.append(ground_domain[i])

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(token_list)

    for i in range(X.shape[0]):
        # get the first vector out (for the first document)
        first_vector_tfidfvectorizer = X[i]
        # place tf-idf values in a pandas data frame, select first 5 most frequent terms
        df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=vectorizer.get_feature_names(), columns=["tfidf"])
        df = df.sort_values(by=["tfidf"], ascending=False).iloc[:5, :]
        df.to_csv(target_root + '/' + ground_brand[i] + '/tfidf.csv')

def main(data_folder, mode, outfile, target_root):

    N = [1,3,5,10]
    # ts_list = [10, 20, 30, 40, 50, 60, 70, 80]
    # ts_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 93, 95, 97, 99]
    ts_list = [90, 93, 95, 97, 99]
    count = len(os.listdir(data_folder))

    '''Parsing html --> tokenize html and url'''
    urls = []
    htmls_path = []
    data_path = []
    for subfolder in tqdm(os.listdir(data_folder)):
        data_path.append(data_folder + '/' + subfolder)
        file_path = data_folder + '/' + subfolder + "/info.txt"
        if mode == 'phish':
            f = eval(open(file_path, 'r', encoding="ISO-8859-1").read())
            urls.append(f["url"])
        else:
            try:
                f = open(file_path, 'r').read()
            except:
                f = 'https://www.' + subfolder
            urls.append(f)
        html_path = data_folder + '/' + subfolder + "/html.txt"
        htmls_path.append(html_path)

    content = []
    for i in tqdm(range(len(htmls_path))):
        try:
            with open(htmls_path[i]) as handle:
                soup = BeautifulSoup(handle.read(), "html5lib")
                if len(soup) == 0:
                    content.append('')
                else:
                    try:
                        content.append(soup)
                    except:
                        content.append('')
        except:
            content.append('')

    print('Reading completed...')
    print('Number of instances:', len(content))
    assert len(content) == len(urls)

    '''Make SIFT prediction only if it contains popular(Top 5) tokens from targeted brand'''
    detection = np.zeros((1, len(ts_list)))
    identi = np.zeros((4, len(ts_list)))

    for i in tqdm(range(len(urls))):
        if os.path.exists(os.path.join(outfile, 'phishzoo_30k_' + str(95) + '_' + mode + '.txt')):
            if data_path[i] in open(os.path.join(outfile, 'phishzoo_30k_' + str(95) + '_' + mode + '.txt')).read():
                print(i)
                continue

        starttime = datetime.datetime.now()
        if mode == 'phish':
            true_brand = brand_converter(data_path[i].split('/')[-1].split('+')[0])
        else:
            true_brand = brand_converter(data_path[i].split('/')[-1].split('.')[0])

        # tokenize this site (html + url)
        check = TokenSite(content[i], urls[i])
        web_token = check.output()

        ## check against protected logos
        pred_brand = []
        similarity = []
        for folder in os.listdir(target_root):
            ground_token = pd.read_csv(target_root + '/' + folder + '/tfidf.csv')
            ground_token = list(ground_token.iloc[:, 0]) ## get first column
            for token in web_token.split():
                if token in ground_token: # trigger SIFT when at least 1 token match
                    print('success match: ', folder, data_path[i])
                    check = SIFT(data_path[i] + '/shot.png', target_root + '/' + folder)
                    _, _, sim = check.match()
                    similarity.append(sim)
                    pred_brand.append(folder)
                    break
        assert len(similarity) == len(pred_brand)  ## each protected brand is associated with 1 similarity

        ## sort according to similarity in descending order
        pred_brand_sort = np.array(pred_brand)[np.argsort(similarity)[::-1]]
        similarity_sort = np.array(similarity)[np.argsort(similarity)[::-1]]
        end_time = str(datetime.datetime.now() - starttime)

        for m in range(len(ts_list)):
            t_s = ts_list[m]
            file = os.path.join(outfile, 'phishzoo_30k_' + str(t_s) + '_' + mode + '.txt')
            f = open(file, 'a+')
            f.write(data_path[i] + '/shot.png' + "\t")

            ## filter predictions which exceed threshold t_s
            pred_brand_sort_filter = pred_brand_sort[similarity_sort > t_s]
            similarity_sort_filter = similarity_sort[similarity_sort > t_s]
            assert len(pred_brand_sort_filter) == len(similarity_sort_filter)

            ## if prediction is not None
            if len(similarity_sort_filter) > 0:
                f.write("True\t")
                detection[0][m] += 1
                if mode == 'phish':
                    for j, n in enumerate(N): # check identification for Top 1/3/5/10 prediction
                        similarity_topN, pred_brand_topN = similarity_sort_filter[:min(n, len(similarity_sort_filter))], \
                                                           pred_brand_sort_filter[:min(n, len(pred_brand_sort_filter))]
                        for pred in pred_brand_topN:
                            if brand_converter(pred) == brand_converter(true_brand):
                                f.write(str(n) + "\t")
                                identi[j][m] += 1
                                break

            f.write(end_time)  # record the time for 1 screenshot
            f.write("\n")
            f.close()

    '''Write the final result'''
    result = [("t_s", ts_list), ("count", count), ("detec", detection), ("identi", identi)]
    print('Final result:', result)
    with open(os.path.join(outfile, 'phishzoo_all_30k_%s.txt'%mode), 'w') as f:
        f.write(str(result))

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--data_folder", help='Website folder', required=True)
    parser.add_argument('-m', "--mode", help='Mode of testing: phish|benign', required=True)
    parser.add_argument('-o', '--output_basedir', help='Output text file', default='TestResult')
    parser.add_argument('-t', '--targetlist', help='Path to targetlist folder', required=True)
    args = parser.parse_args()

    # domain_map = '../../benchmark/domain_map.pkl'
    # dict_construct(args.targetlist, domain_map)

    if not os.path.exists(args.output_basedir):
        os.mkdir(args.output_basedir)

    main(args.data_folder, args.mode, args.output_basedir, args.targetlist)



