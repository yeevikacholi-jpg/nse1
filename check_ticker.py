from data_collector import fetch_screener_data
for t in ['RELIANCE','RELIANCEINDUSTRIES','TCS','INFY','HDFC','SBIN','ICICIBANK']:
    try:
        df=fetch_screener_data(t)
        print(t, len(df))
    except Exception as e:
        print(t, 'err', e)
