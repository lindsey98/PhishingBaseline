
#import libraries
import numpy as np
import pandas as pd
import tldextract
import os
import pickle
from bs4 import Comment
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import time
import argparse

## HTML - based feature extraction

class html_check():
    def __init__(self,text,url):  ##input the html and the domain name for example
                                     
        self.text = text
        self.url = url
        self.domain = tldextract.extract(self.url).domain
        
#------------------------------length of HTML----------------------------------------------      
    
    def find_len(self,tag):
        if tag =='!--':
            soup = self.text
            len_sum = 0
            for comment in soup.find_all(string=lambda text:isinstance(text, Comment)):
                len_sum += len(comment)
            return len_sum          
        else:
            soup = self.text
            scripts = soup.find_all(str(tag))
            len_sum = 0

            for script in scripts:
                  len_sum += len(script.text)
            return len_sum
        
    def len_html_tag(self):  ## this is the total length for 5 special tags
        return html_check.find_len(self,"style") + html_check.find_len(self,"link") + html_check.find_len(self,"form") + html_check.find_len(self,"!--") + html_check.find_len(self,"script")

    def len_html(self):
        return len(self.text.text)
    
#------------------------------hidden content------------------------------------------------------------------------
    def hidden_div(self):
        soup = self.text
        scripts = soup.find_all('div')          
        find = 0
        for script in scripts:
            try:
                if str(script.attrs['style'])=='visibility:hidden' or str(script.attrs['style'])=='display:none':
                    find = 1    
                    break               
            except:
                continue
        return find 
    
    def hidden_button(self):
        soup = self.text
        scripts = soup.find_all('button')          
        find = 0
        for script in scripts:
            try:
                if str(script.attrs['disabled'])=='disabled':
                    find = 1   
                    break
            except:
                continue
        return find 
    
    def hidden_input(self):
        soup = self.text
        scripts = soup.find_all('input')          
        find = 0
        for script in scripts:
            try:
                if str(script.attrs['type'])=='hidden' or str(script.attrs['disabled'])=='disabled':
                    find = 1
                    break
            except:
                continue                
        return find 
    
  
    def hidden(self): ## have hidden content
        return int(html_check.hidden_div(self) | html_check.hidden_button(self) | html_check.hidden_input(self))
    
#----------------------------link based------------------------------------------------    

    def find_all_link(self):
        soup = self.text
        a_tags = soup.find_all('a')
        a_data = []

        for a_tag in a_tags:
            try:
                a_data.append(a_tag.attrs['href'])
            except:
                continue
        
        return a_data
    
    
    def find_source(self,tag):  ## find src attribute in <img> <link> ... 
        if tag =='link':
            soup = self.text
            links = soup.find_all('link')
            link_data = []

            for link in links:
                try:
                    link_data.append(link.attrs['href'])
                except:
                    continue
            return link_data
        
        else:
            soup = self.text
            resources = soup.find_all(str(tag))
            data = []

            for resource in resources:
                try:
                    data.append(resource.attrs['src'])       
                except:
                    continue
            return data
        
    def internal_external_link(self):  ## Number of internal hyperlinks and number of external hyperlinks
        link_list = html_check.find_all_link(self)
        if len(link_list)==0:  ## in case there is no hyperlink
            return [0, 0]  
        
        count = 0
        for j in link_list:
            if "http" in j:
                brand = tldextract.extract(j).domain
                if str(brand) == self.domain:
                    count += 1
            else:
                count +=1                
       
        return [count, len(link_list) - count]

    def empty_link(self): ## Number of empty links
        link_list = html_check.find_all_link(self)
        count = 0
        for j in link_list:
            if j=="" or j=="#" or j=='#javascript::void(0)' or j=='#content' or j=='#skip' or j=='javascript:;' or j=='javascript::void(0);' or j=='javascript::void(0)':
                count += 1
        if len(link_list)==0:
            return 0
        return count 

#----------------------------form based---------------------------------------------------    
    
    def find_form(self):
        soup = self.text
        forms = soup.find_all('form')
        data = []

        for form in forms:
            input_tags = form.find_all('input')
            for input_ in input_tags:
                try:
                    if input_.has_key('name'):
                        data.append(str(input_['name']))
                except:
                    continue
        
        return data
    
    def login_form(self): ## have login-form requires password
        input_list = html_check.find_form(self)
        result = 0
        for j in input_list:
            if j.find("password")!=-1 or j.find("pass")!=-1 or j.find("login")!=-1 or j.find("signin")!=-1:
                result = 1
                break
        return result
    
    
    def internal_external_resource(self): ##
        tag_list = ['link','img','script','noscript']
        resource_list = []
        count = 0
        for tag in tag_list:
            resource_list.append(html_check.find_source(self,tag))
        
        resource_list = [y for x in resource_list for y in x]
        if len(resource_list)==0: ## in case there is no resource link
            return [0, 0]

        for j in resource_list:
            if "http" in j:
                if not(self.domain == tldextract.extract(j).domain):
                    count +=1   
        
        return len(resource_list) - count, count
    
    
#-----------------suspicious element HTML-------------------------------------------------        
     
    def redirect(self): ##auto-refresh webpage
        soup = self.text
        return int('redirect' in soup)   
                                  
    def alarm_window(self):  ## alert window pop up
        soup = self.text
        scripts = soup.find_all('script')          
        find = 0
        for script in scripts:
            try:
                if('alert' in str(script.contents)) or ('window.open' in str(script.contents)):
                    find = 1
                    break  
            except:
                continue

        return find
#---------------------------------domain vs HTML content----------------------------------------
    def title_domain(self):
        soup = self.text
        try:
            return int(self.domain.lower() in soup.title.text.lower())
        except:
            return 0
        
    def domain_occurrence(self):
        try:
            return str(self.text).count(self.domain)
        except:
            return 0
        
    def brand_freq_domain(self):
        link_list = html_check.find_all_link(self)
        domain_list = []
        
        for j in link_list:
            if "http" in j:
                brand = tldextract.extract(j).domain
                domain_list.append(brand)
            else:
                domain_list.append(self.domain)
              
        if len(domain_list) == 0:
            return 1
        if pd.Series(domain_list).value_counts().index[0] == self.domain:
            return 1
        else:
            return 0

## URL based features

class URL_check():
    
    def __init__(self, url, tldlist_path):
        self.url = url.lower()
        self.tldlist_path = tldlist_path
    def domain_is_IP(self):
        
        if len(tldextract.extract(self.url).subdomain) == 0:
            hostname = tldextract.extract(self.url).domain
        else:
            hostname = '.'.join([tldextract.extract(self.url).subdomain, tldextract.extract(self.url).domain])
          
        if np.sum([i.isdigit() for i in hostname.split(".")])==4:
            return 1
        else:
            return 0   

    def symbol_count(self):
        
        punctuation_list = ["@", '-' ,'~']
        count=0
        for j in punctuation_list:
            if j in self.url:
                count+=1
        return count

    def https(self):
        return int("https://" in self.url)
    
    def domain_len(self):
        if len(tldextract.extract(self.url).subdomain)==0:
            domain_len = len(tldextract.extract(self.url).domain+"."+tldextract.extract(self.url).suffix)
        else:
            domain_len=len(tldextract.extract(self.url).subdomain +"."+ tldextract.extract(self.url).domain+"."+tldextract.extract(self.url).suffix)
        return domain_len
    
    def url_len(self):
        return len(self.url)

    def num_dot_hostname(self):
        if len(tldextract.extract(self.url).subdomain) == 0:
            hostname = tldextract.extract(self.url).domain
        else:
            hostname = '.'.join([tldextract.extract(self.url).subdomain, tldextract.extract(self.url).domain])
        return hostname.count('.')
    
    def sensitive_word(self):
        sensitive_list = ["secure", "account", "webscr", "login",
                          "signin", "ebayisapi", "banking", "confirm"]
        return int(any(x in self.url for x in sensitive_list))
 
    def tld_in_domain(self):
        path = self.tldlist_path
        tld_list = list(pd.read_csv(path, encoding = "ISO-8859-1").Domain.apply(lambda x: x.replace('.','')))
        
        for tld in tld_list: 
            if tld == tldextract.extract(self.url).subdomain or tld == tldextract.extract(self.url).domain:
                return 1
        return 0
        
    def tld_in_path(self):
        path = self.tldlist_path
        tld_list = list(pd.read_csv(path, encoding = "ISO-8859-1").Domain.apply(lambda x: x.replace('.','')))
        
        for tld in tld_list:
            if tld == urlparse(self.url).path or tld == urlparse(self.url).params or tld == urlparse(self.url).query or tld == urlparse(self.url).fragment:
                return 1
        return 0
        
def main(path,  outfile, modeldir, tldlist):
    files = os.listdir(path)
    '''read urls and html documents'''
    urls = []
    htmls_path = []
    for i in files:
        file_path = path + '/' + i + "/info.txt"
        try:
            urls.append(open(file_path, 'r').read())
        except:
            urls.append('https://www.' + i)
        html_path = path + '/' + i + '/' + "/html.txt"
        htmls_path.append(html_path)

    '''parse html content with bs4'''
    content = []
    for i in range(len(htmls_path)):
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
    assert len(content) == len(urls)

    '''extract html features'''
    start_time = time.time()

    internal_link = []
    external_link = []
    empty_link = []
    login_form = []
    html_len_tag = []
    html_len = []
    alarm_window = []
    redirection = []
    hidden = []
    title_domain = []
    brand_domain = []
    internal_resource = []
    external_resource = []
    domain_occurrence = []

    for i in range(len(urls)):
        if len(content[i]) == 0:
            internal_link.append(0)
            external_link.append(0)
            empty_link.append(0)
            login_form.append(0)

            html_len_tag.append(0)
            html_len.append(0)

            alarm_window.append(0)
            redirection.append(0)
            hidden.append(0)

            title_domain.append(0)
            internal_resource.append(0)
            external_resource.append(0)
            domain_occurrence.append(0)
            brand_domain.append(0)
        else:
            test = html_check(content[i], urls[i])
            internal_link.append(test.internal_external_link()[0])
            external_link.append(test.internal_external_link()[1])
            empty_link.append(test.empty_link())
            login_form.append(test.login_form())

            html_len_tag.append(test.len_html_tag())
            html_len.append(test.len_html())

            alarm_window.append(test.alarm_window())
            redirection.append(test.redirect())
            hidden.append(test.hidden())

            title_domain.append(test.title_domain())
            internal_resource.append(test.internal_external_resource()[0])
            external_resource.append(test.internal_external_resource()[1])
            domain_occurrence.append(test.domain_occurrence())
            brand_domain.append(test.brand_freq_domain())
        if i%1000 == 0:
            print('%d urls completed for html feature extraction'%i)

    '''extract url features'''
    domain_is_ip=[]
    symbol_count=[]
    http = []
    domain_len = []
    url_len = []
    num_dot_hostname = []
    sensitive_word=[]
    tld_in_domain = []
    tld_in_path = []

    for i in range(len(urls)):
        url_class = URL_check(urls[i], tldlist)
        domain_is_ip.append(url_class.domain_is_IP())
        symbol_count.append(url_class.symbol_count())
        http.append(url_class.https())
        domain_len.append(url_class.domain_len())
        url_len.append(url_class.url_len())
        num_dot_hostname.append(url_class.num_dot_hostname())
        sensitive_word.append(url_class.sensitive_word())
        tld_in_domain.append(url_class.tld_in_domain())
        tld_in_path.append(url_class.tld_in_path())
        if i%1000 == 0:
            print('%d urls completed for url feature extraction'%i)

    print('Feature Extraction completed, average time taken:', (time.time() - start_time)/len(urls))
    '''Combine features into a dataframe'''
    df_feature = pd.concat([pd.Series(internal_link), pd.Series(external_link), pd.Series(empty_link),
                           pd.Series(login_form), pd.Series(html_len_tag), pd.Series(html_len),
                           pd.Series(alarm_window), pd.Series(redirection), pd.Series(hidden),
                           pd.Series(title_domain), pd.Series(brand_domain), pd.Series(internal_resource),
                           pd.Series(external_resource), pd.Series(domain_occurrence), pd.Series(domain_is_ip),
                           pd.Series(symbol_count), pd.Series(http), pd.Series(domain_len),
                           pd.Series(url_len), pd.Series(num_dot_hostname), pd.Series(sensitive_word),
                           pd.Series(tld_in_domain), pd.Series(tld_in_path)], axis=1)

    df_feature.columns = ['internal_link', 'external_link', 'empty_link',
                          'login_form', 'html_len_tag', 'html_len',
                          'alarm_window',  'redirection', 'hidden',
                          'title_domain', 'brand_domain','internal_resource', 'external_resource',
                          'domain_occurrence', 'domain_is_ip', 'symbol_count', 'http',
                           'domain_len','url_len', 'num_dot_hostname','sensitive_word',
                           'tld_in_domain', 'tld_in_path']

    df_feature.index = urls
    df_feature.to_csv(os.path.join(outfile, 'feature.csv'))

    '''get predictions'''
    x = df_feature
    with open(os.path.join(modeldir, 'model.pkl'), 'rb') as handle:
        model = pickle.load(handle)
    start_time = time.time()

    y_prob = model.predict_proba(x)[:, 1]
    y_pred = model.predict(x)

    print('Average runtime per url: ', (time.time() - start_time)/len(x))

    pred_df = pd.DataFrame({'URL': urls, 'y_pred': y_pred, 'y_prob': y_prob})
    print("Number of predicted positives:", list(pred_df['y_pred']).count(1))
    pred_df.to_csv(os.path.join(outfile, 'predict.csv'), index=None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--data_folder", help='Website folder', required=True)
    parser.add_argument('-o', '--output_basedir', help='Output text file', default='TestResult')
    parser.add_argument('-md', '--modeldir', help='Saved model directory', default='')
    parser.add_argument('-t', '--tldlist', help='Collection of all TLDs', default='tld.csv')
    args = parser.parse_args()

    if not os.path.exists(args.output_basedir):
        os.mkdir(args.output_basedir)

    main(args.data_folder, args.output_basedir, args.modeldir, args.tldlist)

