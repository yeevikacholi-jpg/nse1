import requests
from bs4 import BeautifulSoup
headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
for ticker in ['reliance','tcs','infy','hdfc','sbin','icicibank']:
    url=f'https://www.screener.in/company/{ticker}/consolidated/'
    r=requests.get(url, headers=headers, timeout=30)
    print('---', ticker, 'status', r.status_code)
    if r.status_code != 200:
        continue
    soup=BeautifulSoup(r.text, 'html.parser')
    scripts=soup.find_all('script')
    hit=False
    for i, script in enumerate(scripts):
        text=script.string or ''
        if 'prices' in text or 'volumes' in text or 'window.chartData' in text or 'chart' in text:
            if 'var prices' in text or 'chart data' in text or 'window.chartData' in text:
                print('ticker', ticker, 'script', i, 'len', len(text))
                print(text[:700].replace('\n',' '))
                hit=True
                break
    print('contains prices/volumes:', hit)
