import requests
urls=['http://127.0.0.1:5000/api/load?ticker=tcs','http://127.0.0.1:5000/api/results?ticker=tcs']
for u in urls:
    r=requests.get(u, timeout=30)
    print(u, r.status_code, r.text[:360])
