import requests
##import parsel
from lxml import etree
from tqdm import tqdm
import time
import re

class FinNLP_Downloader:
    def __init__(self, args = {}):
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
            elif len(self.proxy_list) >0:
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

    def _request_get(self, url, headers = None, verify = None, params = None):
        if headers is None:
            headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/112.0"
            }
        max_retry = self.max_retry
        proxies = self._get_proxy()
        for _ in range(max_retry):
            try:
                response = requests.get(url = url, proxies = proxies, headers = headers, verify = verify, params = params)
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
                response = requests.post(url = url, headers = headers, json = json, proxies = proxies)
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
