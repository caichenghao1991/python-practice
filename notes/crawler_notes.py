'''
    Web Crawler
        steps: data crawl, analysis, and storage
        usage: data science/AI, application cold start(collect data for base data), social media/ competitors monitoring

    URL
        https://guye:123       @list.jd.com:443/  list.html?cat=9987&page=1#J_main
        scheme username+password  hostname, port, path,      query,          fragment
        netloc: username+password+hostname+port

    data crawling:
        library: requests, urllib, pycurl
        tools: curl (depend on openssl), wget, httpie

    curl https://www.baidu.com   # return html code, base user-agent is curl
        curl -A "Chrome" https://www.baidu.com  # to specifies user-agent to chrome browser, will include js, css...
            -X POST  # specifies method     -I  # only return response header
            crul -d data=123 https://www.baidu.com  # use post method send data start request
                curl -d a=1 -d b=2 https://www.baidu.com,  curl -d "a=1&b=2" https://www.baidu.com  # send multiple data
                curl -d /post.data https://www.baidu.com      post.data: a=1&b=2   # send post request data via file
                curl -O https://www.baidu.com/logo.ico     # download server original file and save with same name
                    # curl -o hi.ico https://www.baidu.com/logo.ico   # download and rename file
                -L  # follow redirect request (usually use)
                curl -OH "accept:image/ico" https://www.baidu.com/logo.ico   # specifies header information
                -k  # allow send unsafe SSL request
                curl -b test=hi https://www.baidu.com/cookie  # send request with cookie inside
                -s   # only return result (no speed, time, total bytes... info)
                -v   # display all information during connection
                curl --help    man curl  # help file

                alias myip="curl http://httpbin.org/get|grep -E '\d+'|grep -v User-Agent|cut -d '\"' -f4"


    wget   used for download
        wget https://www.baidu.com/logo.ico   # download from url
            wget -O hi.ico https://www.baidu.com/logo.ico  # specifies downloaded file name
        --limit-rate=20k   # limit download speed
        -c                 # allow continue download for unfinished job
        -b                 # download backend
        wget -U "Windows IE 10.0" https://www.baidu.com/logo.ico   # specifies user-agent
        tail -f wget-log   # check wget download log (while downloading at backend)
        -p                 # download all related resource (js, css, img)
        --mirrow           # mirrow specific website
        -r                 # recursive download all links inside website
        --convert-links    # change relative path to absolute path (used to make website links works correctly)

        wget --help   man wget  # help file

        wget -c --mirror -U "Mozilla" -p --convert-links http://docs.python-requests.org
        python -m http-server   # make current folder (with index.html) a localhost server


    httpie   similar tool of curl
        pip install httpie
        http https://www.baidu.com

    other software: charles, fiddler, postman

    urllib  python 3 base library
        error, parse, request, response, robotparser
        import requests
        import json
        # or import urllib.parse, urllib.request
        resp = urlopen('https://www.baidu.com')   # return http.client.HTTPResponse(extend io.BufferedIOBase)
            # input either string url or urllib.request.Request object
        text = resp.read() # byte stream
        text = text.decode()  # string of dictionary
        text = json.load(text)  # return dict  # data need to be json

        resp.status   # status code
        resp.reason   # status reason "OK" for 200
        resp.headers.get_all('Content-Type')  # get header content-type
        resp.headers.keys()  # all header keys
        dict(r.headers._headers)

        # get method with parameters
        params = urllib.parse.urlencode({'Harry Potter': 10, 'Ronald Weasley': 11})
        url = 'http://httpbin.org/get?%s' % params
        with urllib.request.urlopen(url) as resp:
            print(json.load(resp))

        # post method with parameters
        data =  urllib.parse.urlencode({'Harry Potter': 10, 'Ronald Weasley': 11})
        data = data.encode()  # byte
        with urllib.request.urlopen('http://httpbin.org/post', data) as resp:
            print(json.load(resp))  # read json response

        # use proxy to request url
        proxy_handler = urllib.request.ProxyHandler({'sock5': 'localhost:1080'})
            # need valid proxy{'http':'http://iguye.com:41801'}

            # basic auth authentication need usernamem password
            # proxy_handler = urllib.request.ProxyBasicAuthHandler()
            #proxy_auth_handler.add_password('realm', 'host', 'username', 'password')
        opener = urllib.request.build_opener(proxy_handler) # add , proxy_auth_handler for basic auth
        resp = opener.open('http://httpbin.org/ip')
        print(resp.read())  # read response content
        resp.close()

        o = urllib.parse.urlparse('https://www.baidu.com')  # splitting a URL string into its components
        print(o.port, o.scheme, o.encode,  o.netloc, o.username, o.geturl())
        print(dir(o))


    requests
        # get request
        headers = {'User-Agent': 'Mozilla/5.0'}  # some websites need headers, default user-agent: python-requests
        resp = requests.get('http://httpbin.org/get')  # , headers=headers)
        print(resp.status_code, resp.reason, resp.text)  # return response object string response content
        # get request with parameter
        resp = requests.get('http://httpbin.org/get', params={'Harry Potter': 10})
        print(resp.json())
        # get request with cookie
        cookies = dict(user='Harry Potter', token='xxxx')
        resp = requests.get('http://httpbin.org/cookies', cookies=cookies)
        print(resp.json())
        # Basic-auth request
        resp = requests.get('http://httpbin.org/basic-auth/harry/123456', auth=('harry','123456'))
        print(resp.json())
        # request with proxies
        proxies = { "http": "http://222.74.202.245:8080", "https": "http://64.124.38.142:8080"}
        resp = requests.get('http://httpbin.org/get', proxies=proxies, params={'Harry Potter': 10})
            # use tinyproxy to setup own proxy
        print(resp.json())

        # post request
        resp = requests.post('http://httpbin.org/post',data={'Harry Potter': 10})
        print(resp.json())   # json response content, data inside 'form'

        bad_resp = requests.get('http://httpbin.org/status/404')
        #bad_resp.raise_status()  # raise error if 404 instead of default no effect on server

        s = requests.Session()  # get session object
        s.get('http://httpbin.org/cookies/set.userud/123456') #session will save the response content
        resp = s.get('http://httpbin.org/cookies')  # automatically add all session data inside header of request
        print(resp.json())


    bs4 (beautiful spoup) : process HTML page
        pip install bs4
        from bs4 import BeautifulSoup
        hmtl_doc="""
            <html><head><title>Hogwarts</title></head>
            <body><p>Welcome to Hogwarts!</p>
            <a class="link" href="https://www.baidu.com" id="link1">reference</a>
            </html>
            """
        soup = BeautifulSoup(html_doc, "html.parser")
        print(soup.prettify())   # reformat code (switch lines, tabs)
        print(soup.title)  # <title>Hogwarts</title>   bs4.element.Tag object
        print(soup.title.text)  # Hogwarts
        print(soup.a)  # <a class="link" href="https://www.baidu.com" id="link1">reference</a>
            # only return first <a>
        print(soup.a.attrs)   # {'class':['link'], 'href':'https://www.baidu.com', 'id':'link1'}
        print(soup.a.attrs['href']   # 'https://www.baidu.com'
        print(list(soup.p.children)[0].text)  # Welcome to Hogwarts!   first <p> text
        print(soup.find_all('a'))   # list of <a>
        print(soup.find(id='link1'))  # find tag with id
        print(soup.find(id='link1').has_attr('href')) # return boolean whether has attribute
        print(soup.select('.link'))   # find by class name with css selector
        print(soup.get_text())   # return all text (title, a, p...) separated by \n


    lxml   faster than beautiful soup html parser, slower than re (regular expression)
        xpath is a language searching content in xml document
        node: include element, attribute, context, name space, document node
        relationship between nodes: parent, children, sibling, ancestor, descendant

        xpath grammar (for html and xml)
            nodename  # select all child node under nodename
            //        # select from any path
            /         # select from root node
            .         # select current node
            ..        # select parent node
            @         # select attribute
            text()    # text content inside tag

        pip install lxml
        soup_lxml = BeautifulSoup(hmtl_doc, "lxml")
        print(soup_lxml.a)   # return first <a>
        selector = etree.HTML(hmtl_doc)
        print(selector.xpath('//p[@class="link"]/a/@href'))  # return list under <p class='link'><a> href attribute
        links = selector.xpath('//p[@class="link"]/a[1]/text()')  # return list, index start with 1
            # a[last()-1]  for second last element
            # a[position()<3]  for first two element, can't use slice
        print(links)
        '//book[price>35]/price/text()'
        links = selector.xpath('//p[contains(@class="lin")]/a[1]/text()')  # class name contain lin


    scrappy

'''
# pass parameter for get method
import os
import re
import threading
import time
from io import BytesIO
from queue import Queue

import certifi
import pycurl
import requests
import json
import urllib
#import urllib.parse, urllib.request
from pycurl import Curl
from bs4 import BeautifulSoup
# get method with parameters
from lxml import etree

def urllib_sample():
    params = urllib.parse.urlencode({'Harry Potter': 10, 'Ronald Weasley': 11})
    url = 'http://httpbin.org/get?%s' % params
    with urllib.request.urlopen(url) as resp:
        print(json.load(resp))


    # post method with parameters
    data =  urllib.parse.urlencode({'Harry Potter': 10, 'Ronald Weasley': 11})
    data = data.encode()  # byte
    with urllib.request.urlopen('http://httpbin.org/post', data) as resp:
        print(json.load(resp))   # read json response

    # use proxy to request url
    proxy_handler = urllib.request.ProxyHandler({'sock5': 'localhost:1080'})
        # need valid proxy{"http": "http://222.74.202.245:8080"}
        # proxy_handler = urllib.request.ProxyBasicAuthHandler()  # basic auth authentication need usernamem password
        #proxy_auth_handler.add_password('realm', 'host', 'username', 'password')
    opener = urllib.request.build_opener(proxy_handler)#, proxy_auth_handler
    resp = opener.open('http://httpbin.org/ip')
    print(resp.read())  # read string response content
    resp.close()

    o = urllib.parse.urlparse('https://www.baidu.com') # splitting a URL string into its components
    print(o.port, o.scheme, o.encode,  o.netloc, o.username, o.geturl())
    print(dir(o))

def requests_sample():
    # get request
    print('-----requests library-----')
    resp = requests.get('http://httpbin.org/get')
    print(resp.status_code, resp.reason, resp.text)  # return response object string response content
    # get request with parameter
    resp = requests.get('http://httpbin.org/get', params={'Harry Potter': 10})
    print(resp.json())
    # get request with cookie
    cookies = dict(user='Harry Potter', token='xxxx')
    resp = requests.get('http://httpbin.org/cookies', cookies=cookies)
    print(resp.json())
    # Basic-auth request
    resp = requests.get('http://httpbin.org/basic-auth/harry/123456', auth=('harry','123456'))
    print(resp.json())
    # request with proxies
    proxies = { "http": "http://222.74.202.245:8080", "https": "https://64.124.38.142:8080"}
    resp = requests.get('http://httpbin.org/get', proxies=proxies, params={'Harry Potter': 10})
        # use tinyproxy to setup own proxy
    print(resp.json())

    # post request
    resp = requests.post('http://httpbin.org/post',data={'Harry Potter': 10})
    print(resp.json())   # json response content, data inside 'form'

    bad_resp = requests.get('http://httpbin.org/status/404')
    #bad_resp.raise_status()  # raise error if 404 instead of default no effect on server

    s = requests.Session()  # get session object
    s.get('http://httpbin.org/cookies/set.userud/123456') #session will save the response content
    resp = s.get('http://httpbin.org/cookies')  # automatically add all session data inside header of request
    print(resp.json())


def beautiful_soup_sample():
    hmtl_doc = """
            <html><head><title>Hogwarts</title></head>
            <body><p>Welcome to Hogwarts!</p>
            <a class="link" href="https://www.baidu.com" id="link1">reference</a>
            <p class="link"><a href="https://www.google.com" id="link2">reference2</a>
            <a href="https://www.facebook.com" id="link3">reference3</a></p>
            </html>
        """
    #soup = BeautifulSoup(requests.get('https://www.baidu.com').text)
    soup = BeautifulSoup(hmtl_doc, "html.parser")
    print(soup.prettify())   # reformat code (switch lines, tabs)
    print(soup.title)  # <title>Hogwarts</title>   bs4.element.Tag object
    print(soup.title.text)  # Hogwarts
    print(soup.a)  # <a class="link" href="https://www.baidu.com" id="link1">reference</a>
        # only return first <a>
    print(soup.a.attrs)   # {'class':['link'], 'href':'https://www.baidu.com', 'id':'link1'}
    print(soup.a['href'])   # 'https://www.baidu.com'
    print(list(soup.p.children)[0].text)  # Welcome to Hogwarts!   first <p> text
    print(soup.find_all('a'))   # list of <a>
    print(soup.find(id='link1'))  # find tag with id
    print(soup.select('.link'))   # find by class name with css selector
    print(soup.get_text())   # return all text (title, a, p...) separated by \n

    # lxml
    soup_lxml = BeautifulSoup(hmtl_doc, "lxml")
    print(soup_lxml.a)   # return first <a>
    # xpath
    selector = etree.HTML(hmtl_doc)
    print(selector.xpath('//p[@class="link"]/a/@href'))  # xpath  return list under <p class='link'><a> href attribute
    links = selector.xpath('//p[@class="link"]/a[1]/text()')  # return list, index start with 1
    print(links)


# Beautiful Soup + requests
def practice1():
    headers = {'User-Agent': 'Mozilla/5.0'}
    r = requests.get('https://www.xiachufang.com/', headers=headers)
    soup = BeautifulSoup(r.text, "lxml")
    img_list = []

    for img in soup.select('img'):
        if img:
            if img.has_attr('data-src'):
                img_list.append(img['data-src'])

    image_dir = os.path.join('..','resources','images')
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)

    for img in img_list[:2]:
        o = urllib.parse.urlparse(img)
        file_name = o.path[1:].split('@')[0]
        file_path = os.path.join(image_dir, file_name)

        if not os.path.exists(file_path):
            url = '%s://%s/%s' % (o.scheme, o.netloc, file_name)
            # https://i2.chuimg.com/cf0dc39b06cd4c74b36a666b327dacdc_1080w_1428h.jpg
            resp = requests.get(url)
            with open(file_path, 'wb') as f:
                for chunk in resp.iter_content(1024):
                    f.write(chunk)


# pycurl + re
def download_image(images,count):
    imgdir = os.path.join('..', 'resources', 'images')
    for url in images[:count]:
        img_name = url.split('/')[-1]
        path = os.path.join(imgdir, img_name)
        if not os.path.exists(path):
            r = requests.get(url)
            with open(path, 'wb') as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)

def practice2():
    buffer = BytesIO()
    c = Curl()
    c.setopt(c.URL, 'https://www.xiachufang.com/')
    c.setopt(pycurl.CAINFO, certifi.where())  # add certificate fo SSL (https)
    c.setopt(c.WRITEDATA, buffer)    # specific output write to buffer
    c.perform()
    c.close()

    body = buffer.getvalue()
    text = body.decode('utf8')  # html page
    img_list = re.findall(r'src=\"(https://i2\.chuimg\.com/\w+\.jpg)', text)
    #download_image(img_list,2)


# curl command download all images into current path
#curl -s https://www.xiachufang.com/|grep -oP '(?<=src=\")https://i2\.chuimg\.com/\w+\.jpg'|xargs -i curl {} -O
    # (?<=src=\")  match in regular expression, but not include src=" in result
    # xargs: use previous output as input parameter in -i {}

def download(queue):
    while True:
        url = queue.get()  # will always wait until next value
        if url is None:   #queue.empty()
            print(queue.empty())
            break
        imgdir = os.path.join('..', 'resources', 'images')
        img_name = url.split('/')[-1]
        path = os.path.join(imgdir, img_name)
        if not os.path.exists(path):
            r = requests.get(url)
            print('downloading %s' % path)
            with open(path, 'wb') as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
    queue.task_done()


# lxml
def multi_thread_download():
    start_time = time.time()
    q = Queue()
    THREAD_COUNT = 3
    thread_pool = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    resp = requests.get('https://www.xiachufang.com/', headers=headers)
    selector = etree.HTML(resp.text)
    links = selector.xpath('//div[@class="pop-recipes block"]/ul/li/div/a/img/@src')  # return list, index start with 1
    links = [i.split('.jpg')[0]+'.jpg' for i in links]
    for l in links:
        q.put(l)
    for i in range(THREAD_COUNT):
        t = threading.Thread(target=download, args=(q,))
        t.start()
        thread_pool.append(t)
    #.join()  # block q until empty
    for i in range(THREAD_COUNT):
         q.put(None)

    for t in thread_pool:
        t.join()
    print('download finished. take %.2f seconds' % (time.time()-start_time))



if __name__=='__main__':
    urllib_sample()
    requests_sample()
    beautiful_soup_sample()
    practice1()
    practice2()
    multi_thread_download()