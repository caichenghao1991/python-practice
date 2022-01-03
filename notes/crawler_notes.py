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
        hmtl_doc = requests.get(url).text
        selector = etree.HTML(hmtl_doc)
        print(selector.xpath('//p[@class="link"]/a/@href'))  # return list under <p class='link'><a> href attribute
        links = selector.xpath('//p[@class="link"]/a[1]/text()')  # return list, index start with 1
            # a[last()-1]  for second last element
            # a[position()<3]  for first two element, can't use slice
        print(links)
        '//book[price>35]/price/text()'
        links = selector.xpath('//p[contains(@class="lin")]/a[1]/text()')  # class name contain lin


    scrapy  (pyspider alternative)
        depend on lxml, twisted, openssl
        framework for scraping, use scrapy shell for debug (shelp fetch('url')

        pip install Scrapy

        scrapy runspider basic_spider.py  # run single file

        or create project
        scrapy startproject scrapy_basic   # project name
        scrapy genspider example1 xiachufang.com  # scraper name  allowed domain
        scrapy crawl example1   # run spider for project


        settings.py USER_AGENT = 'Mozilla/5.0'   set user-agent
            ROBOTSTXT_OBEY = False    # allow print in parse()
            TTPCACHE_ENABLED = True  # if need cache static page
            #CONCURRENT_REQUESTS = 32  # max concurrent request

        start via a file
        import scrapy
        class BlogSpider(scrapy.Spider):
            name = 'blogspider'   # spider name
            start_urls = ['https://www.zyte.com/blog/']    # page start scraping

            def parse(self, response):
                blogs = response.css('div.oxy-post-wrap') # css selector  tag.classname
                #blogs = response.xpath('//div[@class="oxy-post-wrap"]')
                for blog in blogs:
                    yield {'title': blog.css('div a.oxy-post-title::text').extract_first(),
                                # extract and getall() will get the html content from selector, return a list
                                # extract_first() and get() return one
                            'author': blog.xpath('./div/div/div[@class="oxy-post-meta-author oxy-post-meta-item"]/text()
                                ').get(),
                           }

                #for next_page in response.css('a.next'):  #
                for next_page in response.css('a.next::attr(href)'):
                #for next_page in response.xpath('//a[@class="next page-numbers"]/@href'):
                    yield response.follow(next_page, self.parse)
                        # response.follow can use relative url,

                    person_url = response.urljoin(href)
                    yield Request(person_url, callback=self.parse)


        # scrapy runspider basic_spider.py   # print result in console
        # scrapy runspider basic_spider.py -o blogs.json  # save result in file
        # scrapy runspider basic_spider.py -o blogs.csv -t csv  # specifies output file format
            # blogs = json.load(open('blogs.json'))
            # print(json['title'])


        # start a scrapy project
        scrapy startproject scrapy_basic   # project name
        cd scrapy_basic
        scrapy genspider example1 amazon.com  # scraper name  start url
            # create a template in scrapy_basic -> spiders -> example1.py
        scrapy crawl example1      # run project
        from __future__ import absolute_import   # solve same module name cause no module found
        class Example1Spider(scrapy.Spider):
            name = 'example1'
            allowed_domains = ['xiachufang.com']  # allowed domain for scraping
            start_urls = ['https://www.xiachufang.com/']  # url start to scraping, can have multiple
            user_agent = 'Mozilla/5.0'

            def parse(self, response):
                items = response.xpath('//div[@class="left-panel"]/ul/li')
                for item in items:
                    detail = item.css('a::attr(href)').extract_first()
                    if detail:
                        url = response.urljoin(detail)
                        category = item.css('a span::text').extract_first()
                        # request need urljoin to get the full url
                        request = scrapy.Request(url,
                                                 callback=self.parse_page2,
                                                 cb_kwargs=dict(main_url=response.url))
                        request.cb_kwargs['category'] = category  # add more arguments for the callback
                        request.meta['test'] = 1   # send meta data through request, some keyword is taken for settings
                                           # in single request override settings.py, while setting.py has global setting
                                           # meta data won't save in cache
                        yield request
                        # yield response.follow(detail, self.parse_page2, cb_kwargs=dict({'main_url':response.url,
                        #    'category':category}))   # self.parse_page2  is the call back function
                            # add  dont_filter=True   # scrape the scarped site again, default generate site md5 set


            def parse_page2(self, response, main_url, category):
                print("test===", response.meta['test'])  # receive meta in response
                response = response.replace(body=response.text.replace('\n', '').replace('\t', ''))
                items = response.css('p.name a::text').getall()  # return list of string
                for i in range(len(items)):
                    items[i] = items[i].strip()
                    if items[i]:
                        dish = DishItem()
                        if not category:
                            category = ""
                        dish['d_category'] = category
                        dish['d_name'] = items[i]

                        yield{
                            'd_name': items[i],    # d_name is the item key
                            'd_category': category
                        }

        items.py   # used for define object from scraped data
        class DishItem(scrapy.Item):  # define object retrieve from scraping
            d_name = scrapy.Field()  # define the fields for your item
            d_category = scrapy.Field()


        uncomment settings.py
            ITEM_PIPELINES = {'scrapy_basic.pipelines.RedisPipeline': 299,
                'scrapy_basic.pipelines.ScrapyBasicPipeline': 300,}
                # pipeline items have order, integer 0-1000 smaller first execute
                # default number in scrapy package settings default_settings.py
        pipeline.py   # needed for preparation and process item
            class RedisPipeline(object):
                def open_spider(self, spider):
                    self.r = redis.Redis(host='127.0.0.1', port=6379,  db=2)  #password=123456,

                def close_spider(self, spider):
                    self.r.close()
                def process_item(self, item, spider):
                    if self.r.sadd('dishes', item['d_name']):
                        return item
                    raise DropItem
            class ScrapyBasicPipeline:
                def open_spider(self, spider):    # run once before open scrapy spider, optional
                    self.conn = pymysql.connect(host='127.0.0.1', port=3306, db='company',
                                         user='cai', password='123456')
                    self.cur = self.conn.cursor()
                    self.cur.execute('delete from t_dish')
                    self.conn.commit()

                def close_spider(self, spider):  # run once after scrapy spider finish, optional
                    self.cur.close()
                    self.conn.close()

                def process_item(self, item, spider):  # run everytime when item is created through yield, required
                    keys = item.keys()
                    values = list(item.values())
                    sql = "insert into t_dish ({}) values ({})".format(','.join(keys), ','.join(['%s'] * len(values)))
                    self.cur.execute(sql, values)
                    self.conn.commit()
                    return item  # must return item


        spider send request of start url to engine -> engine send requests to scheduler for sorting order of execution
            -> scheduler return sorted requests to engine -> engine send request to downloader -> downloader retrieve
            information from internet and send back engine response -> engine send response to spiders -> spider sends
            items/requests(call back) to engine -> engine send items to item pipeline, requests to scheduler

        middleware
            (RobotsTxt, HttpAuth, DownloadTimeout, DefaultHeaders, UserAgent, Retry, MetaRefresh, HttpCompression,
                Redirect, Cookies, HttpProxy, DownloaderStats, HttpError, Offsite, Referer, UrlLength, Depth)
            process_request() called when engine send to downloader -> return response to engine -> return
                a. None: continue other middleware; b. Response:go to process_response; c. Request: go to scheduler;
                d. IgnoredRequest: handle by process_exception, if none then request.errback, if none ignored
            process_response() called when return response to engine -> return a.Response: go to other
                process_response; b. Request: go to scheduler; c. IgnoredRequest: handle by request.errback, if none
                ignored
            process_exception() called when exception in downloader or process_request() return IgnoreRequest ->
                return a. Respone: start  process_response(); b. Request: go to scheduler; c. None: continue other
                process_exception
            from_crawler(cls, crawler) used for build middleware -> return middleware object
        middlewares between engine and spiders, engine and downloader

        scrapy sends various signals and have callback function for asynchronous processing, based on twisted
            library.

        custom middleware
        middlewares.py
            class RandomHttpProxyMiddleware:
                override methods: __init__()   from_crawler()    process_request()   process_response()
                process_exception() based on needs
            then add middleware in settings.py


        extensions:  bind signal to function
        settings.py
            EXTENSIONS = {
            #    'scrapy.extensions.telnet.TelnetConsole': None,
                'scrapy_basic.extensions.SpiderOpenCloseLogging': 1,
            }
            MYEXT_ENABLED =True  # enable own extension, override setting in extension.py
            MYEXT_ITEMCOUNT = 10   # log every 10 items, override setting in extension.py

        extension.py
        from scrapy import signals
        logger = logging.getLogger(__name__)

        class SpiderOpenCloseLogging:

            def __init__(self, item_count):
                self.item_count = item_count
                self.items_scraped = 0
                self.items_dropped = 0

            @classmethod
            def from_crawler(cls, crawler):
                # first check if the extension should be enabled and raise
                # NotConfigured otherwise
                if not crawler.settings.getbool('MYEXT_ENABLED'):
                    raise NotConfigured

                # get the number of items from settings
                item_count = crawler.settings.getint('MYEXT_ITEMCOUNT', 1000)

                # instantiate the extension object
                ext = cls(item_count)

                # connect the extension object to signals

                crawler.signals.connect(ext.spider_opened, signal=signals.spider_opened)
                crawler.signals.connect(ext.spider_closed, signal=signals.spider_closed)
                crawler.signals.connect(ext.item_scraped, signal=signals.item_scraped)
                crawler.signals.connect(ext.item_dropped, signal=signals.item_dropped)
                # return the extension object
                return ext

            def spider_opened(self, spider):
                logger.info("opened spider %s", spider.name)

            def spider_closed(self, spider):
                logger.info("closed spider %s", spider.name)

            def item_scraped(self, item, spider):
                self.items_scraped += 1
                if self.items_scraped % self.item_count == 0:
                    logger.info("scraped %d items", self.items_scraped)

            def item_dropped(self, item, spider,response, exception):
                self.items_dropped += 1
                if self.items_dropped % self.item_count == 0:
                    logger.info("dropped %d items", self.items_dropped)


        scrapy-redis: allow distributed concurrent scraping
            pip install scrapy-redis

            settings.py
                REDIS_URL = 'redis://127.0.0.1:6379'  #'redis://user:pass@hostname:port'
                SCHEDULER = "scrapy_redis.scheduler.Scheduler"  # Enable scheduling storing requests queue in redis
                DUPEFILTER_CLASS = "scrapy_redis.dupefilter.REPDupeFilter"  # enable all spiders share same duplicates
                                                                            # filter through redis
                SCHEDULER_PERSIST = True  # don't cleanup redis queues, allow pause/resume crawls

                #ITEM_PIPELINES add 'scrapy_redis.pipelines.RedisPipeline':301  # optional, save items in redis (memory)

            extend RedisSpider to create long live crawler waiting for url (can perform on multiple machine)
            from scrapy_redis.spiders import RedisSpider
            class Example1Spider(RedisSpider):
                # remove url
            then in redis-cli:  lpush example1:start_urls https://www.xiachufang.com/    #scraper name in class starturl


        css encrypt: crawl css. tinycss parse css file,  find svg img, characters in svg image(x,y coordinate)
        font encrypt: download the random font file, analyze font and character  (TTFont)
        restriction: add cookie inside header, ip frequency restrict (need proxy (tinyproxy)), concurrency restrict
            validation code(hard to decrypt, might able to take detour path to evade validation(hidden field for extra
            piece info in form, send valid validation code and check header data send etc, use visual lib detect simple
            validation code(pytesseract)), or paid real person validation code typing platform(captcha human bypass) )

    selenium   control browser
        perform mouse key actions, retrieve elements on automatic control browser
        return error if can't find element
        slower than scrapy, and more unstable. but easier to perform, no need analyze dynamic backend js and trace api
        to get data, (request param, response result, json data, css, font) might be encrypted

        from selenium import webdriver
        option = webdriver.ChromeOptions()
        option.add_experimental_option('detach', True)  # solve chrome browser quit automatically
        option.add_argument('--headless')   # don't show browser, no need browser driver,
                                            # might cause website authorization after several runs
        driver = webdriver.Chrome(executable_path='../resources/chromedriver.exe', options=option)
            # specify chromedriver location or set driver in path
        driver.get(https://www.google.com)
            driver functions: find_element[s]_by (name, id, class_name, css_selector, xpath, link_text...)
                find element return WebElement, find elements return list[WebElement]  throw error if can't find
                get_cookie, get_cookies, forward, back, refresh, quit, title, current_url,
        element = driver.find_element_by_xpath('//div[@id="u1"]/a[1]')   # return WebElement
            element function: find_element[s]_by (name, id, class_name, css_selector, xpath...)
                id, location, get_property, get_attribute, is_displayed, parent, location, send_keys, click, text,
                tag_name, size, rect, is_enabled, is_selected, value_of_css_property, submit

        # find element
        driver.find_element_by_id('keyword').send_keys('Hogwarts')   # type in search input
        driver.find_element_by_id('sub').click()                     # click search
        h3_list = driver.find_elements_by_tag_name('h3')             # find all result page title
        driver.find_element_by_class_name('nex').click()                     # click next page button
        print(driver.find_element_by_class_name('nex').rect)   # {'height': 38, 'width': 394, 'x': 135, 'y': 17}
        print(driver.find_element_by_class_name('nex').tag_name)   # input  <input type="submit" class="nex"/>
        continue_link = driver.find_element_by_link_text('Continue')   # <a href="continue.html">Continue</a>
        continue_link = driver.find_element_by_partial_link_text('Conti')

        # element methods
        link = driver.find_element_by_xpath('//div[@id="u1"]/a[1]').text  # don't put text() inside xpath
        link = driver.find_elements_by_css_selector('h3 a').text
        print(driver.find_element_by_class_name('nex').get_attribute('type')   # submit
        print(driver.find_elements_by_css_selector('h3 a').get_property('href')))  # href url,  same as get_attribute
        print(driver.find_elements_by_css_selector('h3 a').value_of_css_property('color'))  #rgba(36, 64, 179, 1)

        # execute js
        driver.execute_script('window.scrollTo(0,document.body.scrollHeight)')    # scroll to the bottom
        driver.execute_async_script('xxx')  # async js function

        # browser functions
        driver.save_screenshot('1.png')  # save screenshot 1.png in current path
        driver.back()  # backward page
        print(driver.current_url)  # current url
        driver.forward()  # forward page
        driver.refresh()  # refresh page

        # wait
        driver.implicitly_wait(10)  # implicit wait (wait all dom loaded for at most 10 sec,raise error if not finished)
        links = driver.find_elements_by_xpath('//h3/a')
        links = WebDriverWait(driver,10).until(expected_conditions.presence_of_all_elements_located((By.XPATH,'//h3/a')))
            # explicit wait until element[s] loaded. poll(exam) frequency 0.5s
            # By.ID CLASS_NAME  CSS_SELECTOR   NAME  TAG_NAME
            # presence_of_all_elements_located  don't work with   option.add_argument('--headless')


        driver.quit()          # quit browser


    Splash
        faster than selenium, slower than scrapy
        install via docker recommended
        open local host in vm 8050  or ifconfig  and run that ip:8050
        render the desired page in web page
        show js after rendering

        curl "http://host ip:8050/render.html?url=desire_page" -o output.html
        f =open('output.html')
        text = f.read()
        selector = etree.HTML(text)
        link = selector.expath('//a/text()')

        can work together with scrappy
            https://github.com/scrapy-plugins/scrapy=splash
            pip install scrapy-splash
            docker run -p 8050:8050 scrapinghub/splash

            add downloader middleware, spider_middlewares
            DUPEFILTER_CLASS = 'scrapy_splash.SplashAwareDupeFilter'
            HTTPCACHE_STORAGE = 'scrapy_splash.SplashAwareFSCacheStorage'

            settings.py
                SPLASH_URL = 'http://serverip:8050'
            yield SplashRequest(url, self.parse_result)

    '''

import os
import re
import threading
import time
from io import BytesIO
from queue import Queue

import certifi
import pycurl
import pyexcel
import requests
import json
import urllib
#import urllib.parse, urllib.request
from pycurl import Curl
from bs4 import BeautifulSoup
# get method with parameters
from lxml import etree

# pass parameter for get method
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait


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

def selenium_sample(word):
    option = webdriver.ChromeOptions()
    option.add_experimental_option('detach', True)
    #option.add_argument('--headless')   # don't show browser, don't work together with WebDriverWait
    driver = webdriver.Chrome(executable_path='../resources/chromedriver.exe', options=option)
    driver.get('https://www.baidu.com/')
    inp = driver.find_element_by_id('kw')
    inp.send_keys(word)
    inp.send_keys(Keys.RETURN)
    print(inp.rect)   #{'height': 38, 'width': 394, 'x': 135, 'y': 17}
    print(inp.tag_name)   # input
    #time.sleep(5)

    driver.implicitly_wait(10)  # implicit wait (wait all dom loaded for at most 10 sec, raise error if not finished)
    links = driver.find_elements_by_css_selector('h3 a')
    print(links[1].text, links[1].value_of_css_property('color'), links[1].get_property('href'))
    #links = WebDriverWait(driver,10).until(expected_conditions.presence_of_all_elements_located((By.XPATH,'//h3/a')))
        # explicit wait until element[s] loaded. poll(exam) frequency 0.5s
        # By.ID CLASS_NAME  CSS_SELECTOR   NAME  TAG_NAME
        # presence_of_all_elements_located  don't work with   option.add_argument('--headless')
    title = [_.text for _ in links]
    driver.save_screenshot(os.path.join('..', 'resources', 'images', '1.png'))  # save screenshot in current path
    url = [_.get_attribute('href') for _ in links]
    d = dict(zip(title, url))
    rows=[]
    for k,v in d.items():
        row = {}
        row['title'] = k
        row['link'] = v
        rows.append(row)
    print(rows)
    pyexcel.save_as(records=rows, dest_file_name=os.path.join('..', 'resources', '%s.xls' % word))
    driver.back()  # backward page
    print(driver.current_url)
    driver.forward()  # forward page
    driver.refresh()  # refresh page
    #driver.quit()

def selenium_sample2(word):
    option = webdriver.ChromeOptions()
    option.add_experimental_option('detach', True)
    driver = webdriver.Chrome(executable_path='../resources/chromedriver.exe', options=option)
    driver.get('https://www.google.com/')
    driver.find_element_by_xpath('//form/div/div/div/div/div/input').send_keys(word)
    driver.find_element_by_xpath('//form/div/div/div/div/div/input').send_keys(Keys.RETURN)

    driver.quit()

if __name__=='__main__':
    #urllib_sample()
    #requests_sample()
    #beautiful_soup_sample()
    #practice1()
    #practice2()
    #multi_thread_download()
    selenium_sample('Harry Potter')
