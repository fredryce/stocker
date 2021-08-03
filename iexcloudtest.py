from iexfinance.stocks import Stock
tsla = Stock('TSLA', token="sk_86d5fb508e004a658c87b13c31b487a2")
print(tsla.get_price())