import os

try:
    from dotenv import load_dotenv
    load_dotenv()

    import requests
    import calendar
    import wolframalpha
    import datetime
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from operator import pow, truediv, mul, add, sub

    # Optional imports
    from googleapiclient.discovery import build

except ImportError:
    print('please run `pip install tools-requirements.txt` first at project directory')
    exit()

'''
Calendar

Uses Python's datetime and calendar libraries to retrieve the current date.

input - None

output - A string, the current date.
'''
def Calendar():
    now = datetime.datetime.now()
    return f'Today is {calendar.day_name[now.weekday()]}, {calendar.month_name[now.month]} {now.day}, {now.year}.'


'''
Wikipedia Search

Uses ColBERTv2 to retrieve Wikipedia documents.

input_query - A string, the input query (e.g. "what is a dog?")
k - The number of documents to retrieve

output - A list of strings, each string is a Wikipedia document

Adapted from Stanford's DSP: https://github.com/stanfordnlp/dsp/
Also see: https://github.com/lucabeetz/dsp
'''
class ColBERTv2:
    def __init__(self, url: str):
        self.url = url

    def __call__(self, query, k=10):
        topk = colbertv2_get_request(self.url, query, k)

        topk = [doc['text'] for doc in topk]
        return topk

def colbertv2_get_request(url: str, query: str, k: int):
    payload = {'query': query, 'k': k}
    res = requests.get(url, params=payload)

    topk = res.json()['topk'][:k]
    return topk

def WikiSearch(
    input_query: str,
    url: str = 'http://ec2-44-228-128-229.us-west-2.compute.amazonaws.com:8893/api/search',
    k: int = 10
):
    retrieval_model = ColBERTv2(url)
    output = retrieval_model(input_query, k)
    return output

'''
Machine Translation - NLLB-600M

Uses HuggingFace's transformers library to translate input query to English.

input_query - A string, the input query (e.g. "what is a dog?")

output - A string, the translated input query.
'''
def MT(input_query: str, model_name: str = "facebook/nllb-200-distilled-600M"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    input_ids = tokenizer(input_query, return_tensors='pt')
    outputs = model.generate(
        **input_ids,
        forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"], 
        )
    output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return output


'''
Calculator

Calculates the result of a mathematical expression.

input_query - A string, the input query (e.g. "400/1400")

output - A float, the result of the calculation

Adapted from: https://levelup.gitconnected.com/3-ways-to-write-a-calculator-in-python-61642f2e4a9a 
'''
def Calculator(input_query: str):
    operators = {
        '+': add,
        '-': sub,
        '*': mul,
        '/': truediv
        }
    if input_query.isdigit():
        return float(input_query)
    for c in operators.keys():
        left, operator, right = input_query.partition(c)
        if operator in operators:
            return round(operators[operator](Calculator(left), Calculator(right)), 2)


# Other Optional Tools


'''
Wolfram Alpha Calculator

pip install wolframalpha

Uses Wolfram Alpha API to calculate input query.

input_query - A string, the input query (e.g. "what is 2 + 2?")

output - A string, the answer to the input query

wolfarm_alpha_appid - your Wolfram Alpha API key
'''
def WolframAlphaCalculator(input_query: str):
    wolfram_alpha_appid = os.environ.get('WOLFRAM_ALPHA_APPID')
    wolfram_client = wolframalpha.Client(wolfram_alpha_appid)
    res = wolfram_client.query(input_query)
    assumption = next(res.pods).text
    answer = next(res.results).text
    return f'Assumption: {assumption} \nAnswer: {answer}'


'''
Google Search

Uses Google's Custom Search API to retrieve Google Search results.

input_query - The query to search for.
num_results - The number of results to return.
api_key - Your Google API key.
cse_id - Your Google Custom Search Engine ID.

output - A list of dictionaries, each dictionary is a Google Search result
'''
def custom_search(query, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=cse_id, **kwargs).execute()
    return res['items']

def google_search(input_query: str, num_results: int = 10):
    api_key = os.environ.get('GOOGLE_API_KEY')
    cse_id = os.environ.get('GOOGLE_CSE_ID')

    metadata_results = []
    results = custom_search(input_query, num=num_results, api_key=api_key, cse_id=cse_id)
    for result in results:
        metadata_result = {
            "snippet": result["snippet"],
            "title": result["title"],
            "link": result["link"],
        }
        metadata_results.append(metadata_result)
    return metadata_results


'''
Bing Search

Uses Bing's Custom Search API to retrieve Bing Search results.

input_query: The query to search for.
bing_subscription_key: Your Bing API key.
num_results: The number of results to return.

output: A list of dictionaries, each dictionary is a Bing Search result
'''
def _bing_search_results(
    search_term: str,
    bing_subscription_key: str,
    count: int,
    url: str = "https://api.bing.microsoft.com/v7.0/search"
):
    headers = {"Ocp-Apim-Subscription-Key": bing_subscription_key}
    params = {
        "q": search_term,
        "count": count,
        "textDecorations": True,
        "textFormat": "HTML",
    }
    response = requests.get(
        url, headers=headers, params=params
    )
    response.raise_for_status()
    search_results = response.json()
    return search_results["webPages"]["value"]

def bing_search(
    input_query: str,
    num_results: int = 10
):
    bing_subscription_key = os.environ.get("BING_API_KEY")
    metadata_results = []
    results = _bing_search_results(input_query, bing_subscription_key, count=num_results)
    for result in results:
        metadata_result = {
            "snippet": result["snippet"],
            "title": result["name"],
            "link": result["url"],
        }
        metadata_results.append(metadata_result)
    return metadata_results


if __name__ == '__main__':
 
    print(Calendar()) # Outputs a string, the current date

    print(Calculator('400/1400')) # For Optional Basic Calculator

    print(WikiSearch('What is a dog?')) # Outputs a list of strings, each string is a Wikipedia document

    print(MT("Un chien c'est quoi?")) # What is a dog?

    # Optional Tools

    print(WolframAlphaCalculator('What is 2 + 2?')) # 4

    print(google_search('What is a dog?')) 
    # Outputs a list of dictionaries, each dictionary is a Google Search result

    print(bing_search('What is a dog?')) 
    # Outputs a list of dictionaries, each dictionary is a Bing Search result
