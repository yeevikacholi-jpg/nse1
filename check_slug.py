import requests
slugs=['tata-consultancy-services','reliance-industries','reliance','tcs','infy','hdfc','sbin','icicibank','tata-steel','adani-enterprises']
for s in slugs:
    url=f'https://www.screener.in/company/{s}/consolidated/'
    try:
        r=requests.get(url, timeout=15)
        print(s, r.status_code)
    except Exception as e:
        print(s, 'err', e)
