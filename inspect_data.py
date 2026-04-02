import requests
url='https://www.screener.in/company/reliance/consolidated/'
headers={'User-Agent':'Mozilla/5.0'}
r=requests.get(url, headers=headers, timeout=30)
print('status', r.status_code)
text=r.text
print('len', len(text))
for k in ['prices','volumes','series','Date','close','chart','Screener','window','json','graph']:
    print(k, 'idx', text.find(k))
print('---- snippet 3500-4100 ----')
print(text[3500:4100])
