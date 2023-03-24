import json

with open('company_tickers.json', 'r') as file:
    data = json.load(file)

# unique_data = []
# unique_set = set()
# for value in data.values():
#     data_tuple = (value['cik_str'], value['ticker'], value['title'])
#     if data_tuple not in unique_set:
#         unique_set.add(data_tuple)
#         unique_data.append(value)

unique_data = []
unique_set = set()
for value in data.values():
    title = value['title']
    if title not in unique_set:
        unique_set.add(title)
        unique_data.append(title)

for value in data.values():
    if title in unique_set:
        print(value)

"""
I see that different companies can be listed with the same CIK and company name,
but have a different ticker. I was going to remove them as duplicates, but I guess
users from different countries may use the stock ticker to search company names
and so this informaiton is relevant.
"""

with open('company_tickers_clean.json', 'w') as file:
    json.dump(unique_data, file)
