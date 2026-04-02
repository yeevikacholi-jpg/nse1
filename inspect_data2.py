import requests
url='https://www.screener.in/company/reliance/consolidated/'
headers={'User-Agent':'Mozilla/5.0'}
r=requests.get(url, headers=headers, timeout=30)
text=r.text
print('close idx', text.find('close'))
start = text.find('close')-200
end = text.find('close')+400
print(text[start:end])
