from webull import paper_webull, endpoints # for real money trading, just import 'webull' instead
import json
import requests
import pandas as pd
from webulltest import Plotdata

def login(wb=None):
    print("Logging in to WeBull...")
    #login to Webull
    if not wb:
    	wb = paper_webull()

    loginInfo = None

    f = open("token.txt", "r")
    loginInfo = json.load(f)
    
    wb._refresh_token = loginInfo["refreshToken"]
    wb._access_token = loginInfo["accessToken"]
    wb._token_expire = loginInfo["tokenExpireTime"]
    wb._uuid = loginInfo["uuid"]
    

    new_data = wb.refresh_login()

    try:

        loginInfo["refreshToken"] = new_data["refreshToken"]
        loginInfo["accessToken"] = new_data["accessToken"]
        loginInfo["tokenExpireTime"]= new_data["tokenExpireTime"]
        print("refreshing token....")
        file = open("token.txt", 'w')
        json.dump(loginInfo, file)
    except KeyError as e:
        pass


    webull_email = '' #change to your webul email
    webull_pass = '' #change to your webul email

    loginInfo = wb.login(webull_email, webull_pass)
    id_acc = wb.get_account_id()
    print("account established ", id_acc)

    return wb


def get_stock_list():
    headers = {
        'authority': 'old.nasdaq.com',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'sec-fetch-site': 'cross-site',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-user': '?1',
        'sec-fetch-dest': 'document',
        'referer': 'https://github.com/shilewenuw/get_all_tickers/issues/2',
        'accept-language': 'en-US,en;q=0.9',
        'cookie': 'AKA_A2=A; NSC_W.TJUFEFGFOEFS.OBTEBR.443=ffffffffc3a0f70e45525d5f4f58455e445a4a42378b',
    }
    response = requests.get('https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25&offset=0&exchange=nasdaq&download=true', headers=headers)
    data = response.json()
    #print(data["data"]["rows"][0])

    mydf = pd.DataFrame(data["data"]["rows"])

    #mydf = data["data"]["rows"]

    #print(mydf)

    return mydf["symbol"]

