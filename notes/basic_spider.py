import scrapy

# setting file USER_AGENT = 'Mozilla/5.0'
class BlogSpider(scrapy.Spider):
    name = 'blogspider'   # spider name
    start_urls = ['https://www.zyte.com/blog/']    # page start scraping

    def parse(self, response):
        blogs = response.css('div.oxy-post-wrap') # css selector  tag.classname
        #blogs = response.xpath('//div[@class="oxy-post-wrap"]')
        for blog in blogs:
            yield {'title': blog.css('div a.oxy-post-title::text').extract_first(),
                    'author': blog.xpath('./div/div/div[@class="oxy-post-meta-author oxy-post-meta-item"]/text()').get(),
                   }

        #for next_page in response.css('a.next'):  #
        for next_page in response.css('a.next::attr(href)'):
        #for next_page in response.xpath('//a[@class="next page-numbers"]/@href'):
            yield response.follow(next_page, self.parse)



# scrapy runspider basic_spider.py   # print result in console
# scrapy runspider basic_spider.py -o blogs.json  # save result in file
# scrapy runspider basic_spider.py -o blogs.csv -t csv  # specifies output file format
    # blogs = json.load(open('blogs.json'))
    # print(json['title'])