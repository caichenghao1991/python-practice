# encoding:utf-8
from __future__ import absolute_import
import scrapy

from scrapy_basic.items import DishItem


class Example1Spider(scrapy.Spider):
    name = 'example1'
    allowed_domains = ['xiachufang.com']
    start_urls = ['https://www.xiachufang.com/']
    user_agent = 'Mozilla/5.0'

    def parse(self, response):
        items = response.xpath('//div[@class="left-panel"]/ul/li')
        for item in items:
            detail = item.css('a::attr(href)').get()
            if detail:

                category = item.css('a span::text').get()
                '''
                url = response.urljoin(detail)
                request = scrapy.Request(url,
                                         callback=self.parse_page2,
                                         cb_kwargs=dict(main_url=response.url))
                request.cb_kwargs['category'] = category  # add more arguments for the callback
                yield request'''
                request =response.follow(detail, self.parse_page2, cb_kwargs=dict({'main_url':response.url,
                                                                                'category':category}))
                request.meta['test'] = 1
                yield request

    def parse_page2(self, response, main_url, category):
        print("test===",response.meta['test'])
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
                    'd_name': items[i],
                    'd_category': category
                }

