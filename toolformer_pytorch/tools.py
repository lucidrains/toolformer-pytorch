import requests
import calendar
import wolframalpha
import datetime
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Optional Tool imports
from googleapiclient.discovery import build


'''
Calculator

pip install wolframalpha

Uses Wolfram Alpha API to calculate input query.

input_query - A string, the input query (e.g. "what is 2 + 2?")

output - A string, the answer to the input query

wolfarm_alpha_appid - your Wolfram Alpha API key
'''
def Calculator(input_query: str):
    wolfram_alpha_appid = 'YOUR_WOLFRAM_ALPHA_APPID'
    wolfram_client = wolframalpha.Client(wolfram_alpha_appid)
    res = wolfram_client.query(input_query)
    assumption = next(res.pods).text
    answer = next(res.results).text
    return f'Assumption: {assumption} \nAnswer: {answer}'


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

def WikiSearch(input_query: str):
    k = 10
    retrieval_model = ColBERTv2('http://ec2-44-228-128-229.us-west-2.compute.amazonaws.com:8893/api/search')
    output = retrieval_model(input_query, k)
    return output


'''
Machine Translation - NLLB-600M

Uses HuggingFace's transformers library to translate input query to English.

input_query - A string, the input query (e.g. "what is a dog?")

output - A string, the translated input query.
'''
def MT(input_query: str):
    model_name = "facebook/nllb-200-distilled-600M"
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
Calendar

Uses Python's datetime and calendar libraries to retrieve the current date.

output - A string, the current date.
'''
def Calendar():
    now = datetime.datetime.now()
    return f'Today is {calendar.day_name[now.weekday()]}, {calendar.month_name[now.month]} {now.day}, {now.year}'


# WIP
def question_answering_system():
    pass


# Other Optional Search Tools

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

def google_search(input_query: str, api_key: str, cse_id: str, num_results: int = 10):
    """Searches the API for the query."""
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
def _bing_search_results(search_term: str, bing_subscription_key: str, count: int):
    headers = {"Ocp-Apim-Subscription-Key": bing_subscription_key}
    params = {
        "q": search_term,
        "count": count,
        "textDecorations": True,
        "textFormat": "HTML",
    }
    response = requests.get(
        "https://api.bing.microsoft.com/v7.0/search", headers=headers, params=params
    )
    response.raise_for_status()
    search_results = response.json()
    return search_results["webPages"]["value"]

def bing_search(input_query: str, bing_subscription_key: str, num_results: int):
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
    
    print(Calculator('What is 2 + 2?')) # 4
    
    print(WikiSearch('what is a dog?')) # Outputs a list of strings, each string is a Wikipedia document

    print(MT("Un chien c'est quoi?")) # What is a dog?

    print(Calendar()) # Outputs a string, the current date

    # Optional Tools
    print(google_search('what is a dog?', api_key="YOUR_GOOGLE_API_KEY", cse_id="YOUR_CSE_ID", num_results=10)) # Outputs a list of dictionaries, each dictionary is a Google Search result

    print(bing_search('what is a dog?', bing_subscription_key="YOUR_BING_API_KEY", num_results=10)) # Outputs a list of dictionaries, each dictionary is a Bing Search result