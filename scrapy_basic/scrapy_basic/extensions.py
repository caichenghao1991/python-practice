import logging
from scrapy import signals
from scrapy.exceptions import NotConfigured

logger = logging.getLogger(__name__)

class SpiderOpenCloseLogging:

    def __init__(self, item_count):
        self.item_count = item_count
        self.items_scraped = 0
        self.items_dropped = 0
        self.response_receive = 0

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
        crawler.signals.connect(ext.response_rec, signal=signals.response_received)
        # return the extension object
        return ext

    def spider_opened(self, spider):
        logger.info("opened spider %s" % spider.name)

    def spider_closed(self, spider):
        logger.info("closed spider %s" % spider.name)

    def item_scraped(self, item, spider):
        self.items_scraped += 1
        if self.items_scraped % self.item_count == 0:
            logger.info("scraped %d items" % self.items_scraped)

    def item_dropped(self, item, spider,response, exception):
        self.items_dropped += 1
        if self.items_dropped % self.item_count == 0:
            logger.info("dropped %d items" % self.items_dropped)

    def response_rec(self, response, request, spider):
        if response.status is 200:
            print("xxxx")
            self.response_receive += 1
        if self.response_receive % self.item_count == 0:
            logger.info("**"*20, "%d response received  " % self.items_dropped)