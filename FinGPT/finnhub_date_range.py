import warnings

warnings.filterwarnings("ignore")

from tqdm import tqdm
from lxml import etree
import pandas as pd
import requests
import finnhub
import time
import json
##import parsel
import re


class FinNLP_Downloader:
    def __init__(self, args={}):
        self.use_proxy = True if "use_proxy" in args.keys() else False
        if self.use_proxy:
            self.country = args["use_proxy"]
        else:
            self.country = None
        self.max_retry = args["max_retry"] if "max_retry" in args.keys() else 1
        self.proxy_pages = args["proxy_pages"] if "proxy_pages" in args.keys() else 5
        if self.use_proxy:
            if "kuaidaili" in self.country:
                # tunnel, username, password
                assert "tunnel" in args.keys(), "Please make sure \'tunnel\' in your keys"
                assert "username" in args.keys(), "Please make sure \'username\' in your keys"
                assert "password" in args.keys(), "Please make sure \'password\' in your keys"
                self.proxy_list = Kuaidaili(args["tunnel"], args["username"], args["password"])
            else:
                self.proxy_id = 0
                self.proxy_list = self._update_proxy()
        else:
            self.proxy_list = []

    def _get_proxy(self):
        if self.use_proxy:
            if "kuaidaili" in self.country:
                proxy = self.proxy_list.get_kuaidaili_tunnel_proxy()
                return proxy
            elif len(self.proxy_list) > 0:
                proxy = self.proxy_list[self.proxy_id]
                self.proxy_id += 1
                if self.proxy_id == len(self.proxy_list):
                    self.proxy_id = 0
                return proxy
        else:
            return None

    def _update_proxy(self):
        if "china" in self.country or "China" in self.country:
            return get_china_free_proxy(self.proxy_pages)
        else:
            return get_us_free_proxy(self.proxy_pages)

    def _request_get(self, url, headers=None, verify=None, params=None):
        if headers is None:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/112.0"
            }
        max_retry = self.max_retry
        proxies = self._get_proxy()
        for _ in range(max_retry):
            try:
                response = requests.get(url=url, proxies=proxies, headers=headers, verify=verify, params=params)
                if response.status_code == 200:
                    break
            except:
                response = None

        if response is not None and response.status_code != 200:
            response = None

        return response

    def _request_post(self, url, headers, json):
        max_retry = self.max_retry
        proxies = self._get_proxy()
        for _ in range(max_retry):
            try:
                response = requests.post(url=url, headers=headers, json=json, proxies=proxies)
                if response.status_code == 200:
                    break
            except:
                response = None

        if response is not None and response.status_code != 200:
            response = None

        return response


def check_china_ips(proxies_list):
    """检测ip的方法"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.79 Safari/537.36'}

    can_use = []
    for proxy in tqdm(proxies_list, desc="Checking ips"):
        try:
            response = requests.get('http://www.baidu.com', headers=headers, proxies=proxy, timeout=1)  # 超时报错
            if response.status_code == 200:
                can_use.append(proxy)
        except Exception as error:
            # print(error)
            pass
    return can_use


def check_us_ips(proxies_list):
    """检测ip的方法"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.79 Safari/537.36'}

    can_use = []
    for proxy in tqdm(proxies_list, desc="Checking ips"):
        try:
            response = requests.get('http://www.google.com', headers=headers, proxies=proxy, timeout=1)  # 超时报错
            if response.status_code == 200:
                can_use.append(proxy)
        except Exception as error:
            # print(error)
            pass
    return can_use


def get_china_free_proxy(pages=10):
    proxies_list = []
    for page in tqdm(range(1, pages + 1), desc="Gathering free ips by pages..."):

        base_url = f'https://www.kuaidaili.com/free/inha/{page}'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.79 Safari/537.36'}
        success = False
        while not success:
            try:
                response = requests.get(base_url, headers=headers)
                data = response.text
                res = etree.HTML(data)
                trs = res.xpath('//table/tbody/tr')
                if len(trs) != 0:
                    success = True
                    for tr in trs:
                        proxies_dict = {}
                        http_type = tr.xpath('./td[4]/text()')[0]
                        ip_num = tr.xpath('./td[1]/text()')[0]
                        port_num = tr.xpath('./td[2]/text()')[0]
                        proxies_dict[http_type] = ip_num + ':' + port_num
                        proxies_list.append(proxies_dict)
                else:
                    time.delay(0.01)

            except:
                pass

    can_use = check_china_ips(proxies_list)

    print(f'获取到的代理ip数量: {len(proxies_list)} 。Get proxy ips: {len(proxies_list)}.')
    print(f'能用的代理数量： {len(can_use)}。Usable proxy ips: {len(can_use)}.')

    return can_use


def get_us_free_proxy(pages=10):
    url = "https://openproxy.space/list/http"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.79 Safari/537.36'}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print("Connection Error. Please make sure that your computer now have the access to Google.com")
    res = etree.HTML(response.text)
    http_type = "HTTP"
    proxies_list = []

    scripts = res.xpath("//script")
    content = scripts[3].xpath(".//text()")
    pattern = re.compile('LIST",data:(.+),added:')
    result_list = pattern.findall(content[0])
    result_list = result_list[0].strip("[{").strip("}]").split("},{")

    for result in result_list:
        pattern = re.compile('\[(.+)\]')
        result = pattern.findall(result)
        result = result[0].split(",")
        result = [r.strip("\"") for r in result]
        for ip in result:
            proxies_list.append(
                {http_type: ip}
            )
    total = pages * 15
    proxies_list = proxies_list[:total]
    can_use = check_us_ips(proxies_list)
    print(f'Get proxy ips: {len(proxies_list)}.')
    print(f'Usable proxy ips: {len(can_use)}.')

    return can_use


class Kuaidaili:
    def __init__(self, tunnel, username, password):
        self.tunnel = tunnel
        self.username = username
        self.password = password

    def get_kuaidaili_tunnel_proxy(self):
        proxies = {
            "http": "http://%(user)s:%(pwd)s@%(proxy)s/" % {"user": self.username, "pwd": self.password,
                                                            "proxy": self.tunnel},
            "https": "http://%(user)s:%(pwd)s@%(proxy)s/" % {"user": self.username, "pwd": self.password,
                                                             "proxy": self.tunnel}
        }
        return proxies


class News_Downloader(FinNLP_Downloader):

    def __init__(self, args={}):
        super().__init__(args)
        pass

    def download_date_range(self, start_date, end_date, stock=None):
        pass

    def download_streaming(self, stock=None):
        pass

    def clean_data(self):
        pass

    def _gather_one_part(self, date, stock=None, delay=0.1):
        pass

    def _gather_content(self):
        pass


class Finnhub_Date_Range(News_Downloader):
    def __init__(self, args={}):
        super().__init__(args)
        assert "token" in args.keys(), "Please input your finnhub token. Avaliable at https://finnhub.io/dashboard"
        self.finnhub_client = finnhub.Client(api_key=args["token"])

    def download_date_range_stock(self, start_date, end_date, stock="AAPL"):
        self.date_list = pd.date_range(start_date, end_date)
        self.dataframe = pd.DataFrame()

        days_each_time = 4
        date_list = self.date_list
        # cal total lenth
        if len(date_list) % days_each_time == 0:
            total = len(date_list) // days_each_time
        else:
            total = len(date_list) // days_each_time + 1

        with tqdm(total=total, desc="Downloading Titles") as bar:
            while len(date_list):
                tmp_date_list = date_list[:days_each_time]
                date_list = date_list[days_each_time:]
                tmp_start_date = tmp_date_list[0].strftime("%Y-%m-%d")
                tmp_end_date = tmp_date_list[-1].strftime("%Y-%m-%d")
                res = self._gather_one_part(tmp_start_date, tmp_end_date, stock=stock)
                self.dataframe = pd.concat([self.dataframe, res])
                bar.update(1)

        # res  = self.finnhub_client.company_news(stock, _from=start_date, to=end_date)
        self.dataframe.datetime = pd.to_datetime(self.dataframe.datetime, unit="s")
        self.dataframe = self.dataframe.reset_index(drop=True)

    def _gather_one_part(self, start_date, end_date, stock="AAPL", delay=1):
        res = self.finnhub_client.company_news(stock, _from=start_date, to=end_date)
        time.sleep(delay)
        return pd.DataFrame(res)

    def gather_content(self, delay=0.01):
        pbar = tqdm(total=self.dataframe.shape[0], desc="Gathering news contents")
        self.dataframe["content"] = self.dataframe.apply(lambda x: self._gather_content_apply(x, pbar, delay), axis=1)

    def _gather_content_apply(self, x, pbar, delay=0.01):
        time.sleep(delay)
        url = x.url
        source = x.source
        response = self._request_get(url=url)
        # response = self._request_get(url= url, headers= headers)
        pbar.update(1)
        if response is None:
            return "Connection Error"
        else:
            page = etree.HTML(response.text)

        try:
            # Yahoo Finance
            if source == "Yahoo":
                page = page.xpath(
                    "/html/body/div[3]/div[1]/div/main/div[1]/div/div/div/div/article/div/div/div/div/div/div[2]/div[4]")
                content = page[0].xpath(".//text()")
                content = "\n".join(content)
                return content

            # Reuters
            elif source == "Reuters":
                page = page.xpath("/html/body/div[1]/div[3]/div/main/article/div[1]/div[2]/div/div/div[2]")
                content = page[0].xpath(".//text()")
                content = "\n".join(content)
                return content

            # SeekingAlpha
            elif source == "SeekingAlpha":
                page = page.xpath(
                    "/html/body/div[2]/div/div[1]/main/div/div[2]/div/article/div/div/div[2]/div/section[1]/div/div/div")
                content = page[0].xpath(".//text()")
                content = "\n".join(content)
                return content

            # PennyStocks
            elif source == "PennyStocks":
                page = page.xpath("/html/body/div[3]/div/div[1]/div/div/div/main/article/div[2]/div[2]/div")
                content = page[0].xpath(".//text()")
                content = "\n".join(content)
                return content

            # MarketWatch
            elif source == "MarketWatch":
                page = page.xpath('//*[@id="js-article__body"]')
                content = page[0].xpath(".//text()")
                content = "".join(content)
                while "  " in content:
                    content = content.replace("  ", " ")
                while "\n \n" in content:
                    content = content.replace("\n \n", " ")
                while "\n  " in content:
                    content = content.replace("\n  ", " ")
                return content

            # Seeking Alpha
            elif source == "Seeking Alpha":
                # first get Seeking Alpha URL
                page = page.xpath('/html/body/div[5]/div[2]/section[1]/article[2]/div/div[2]/p/a/@href')
                url_new = page[0]
                response = self._request_get(url=url_new)
                if response is None:
                    return "Connection Error"
                else:
                    page = etree.HTML(response.text)

                content = page[0].xpath(".//text()")
                content = "\n".join(content)
                return content

            # Alliance News
            elif source == "Alliance News":
                page = page.xpath('//*[@id="comtext"]')
                content = page[0].xpath(".//text()")
                content = [c for c in content if not str(c).startswith("\r\n")]
                content = "\n".join(content)
                return content

            # Thefly.com
            elif source == "Thefly.com":
                page = page.xpath('/html/body/div[5]/div[2]/section[1]/article[2]/div/div[2]/p/a/@href')
                url_new = page[0]
                response = self._request_get(url=url_new, verify=False)
                if response is None:
                    return "Connection Error"
                else:
                    page = etree.HTML(response.text)

                page = page.xpath('/html/body/div[2]/div/div/div/div/div[2]/div[2]//text()')
                # content = page[0].xpath(".//text()")
                # content = [c for c in content if not str(c).startswith("\r\n")]
                content = "\n".join(page)
                content = content.replace("\r\n", "")

                return content

            # TalkMarkets
            elif source == "TalkMarkets":
                return "Not supported yet"

            # CNBC
            elif source == "CNBC":
                page = page.xpath('/html/body/div[3]/div/div[1]/div[3]/div/div/div/div[3]/div[1]/div[2]/div[3]//text()')
                content = "\n".join(page)

                return content

            # GuruFocus
            elif source == "GuruFocus":
                page = page.xpath('/html/body/div[5]/div[2]/section[1]/article[2]/div/div[2]/p/a/@href')
                url_new = page[0]
                response = self._request_get(url=url_new)
                if response is None:
                    return "Connection Error"
                else:
                    page = etree.HTML(response.text)

                page = page.xpath(
                    '/html/body/div[1]/div/section/section/main/section/main/div[1]/div/div/div[1]/div[2]/div//text()')
                page_new = []
                for c in page:
                    while "\n" in c:
                        c = c.replace("\n", "")
                    while "  " in c:
                        c = c.replace("  ", "")

                    page_new.append(c)

                content = "\n".join(page_new)

                return content

            # InvestorPlace
            elif source == "InvestorPlace":
                page = page.xpath('/html/body/div[5]/div[2]/section[1]/article[2]/div/div[2]/p/a/@href')
                url_new = page[0]
                response = self._request_get(url=url_new)
                if response is None:
                    return "Connection Error"
                else:
                    page = etree.HTML(response.text)
                    page = page.xpath('//script[@type="application/ld+json"]')[1]
                    content = page.xpath(".//text()")
                    content = json.loads(content[0])
                    content = content["articleBody"]

                    return content

            # TipRanks
            elif source == "TipRanks":
                page = page.xpath('/html/body/div[5]/div[2]/section[1]/article[2]/div/div[2]/p/a/@href')
                url_new = page[0]
                response = self._request_get(url=url_new)
                if response is None:
                    return "Connection Error"
                else:
                    page = etree.HTML(response.text)
                    # /html/body/div[1]/div[2]/div[5]/div[2]/div[2]/div/div[6]/div/article/p[1]/p
                    page = page.xpath('/html/body/div[1]/div[1]/div[4]/div[2]/div[2]/div[1]/div[6]//text()')
                    # content = page[0].xpath('.//text()')
                    page = [p.replace("\n", "") for p in page]
                    content = "".join(page)
                    return content

            else:
                return "Not supported yet"

        except:
            return "Error"
