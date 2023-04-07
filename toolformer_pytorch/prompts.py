DEFAULT_PROMPT_INPUT_TAG = '[input]'

calculator_prompt = f"""
Your task is to add calls to a Calculator API to a piece of text.
The calls should help you get information required to complete the text. 
You can call the API by writing "[Calculator(expression)]" where "expression" is the expression to be computed. 
Here are some examples of API calls:
Input: The number in the next term is 18 + 12 x 3 = 54.
Output: The number in the next term is 18 + 12 x 3 = [Calculator(18 + 12 * 3)] 54.
Input: The population is 658,893 people. This is 11.4% of the national average of 5,763,868 people.
Output: The population is 658,893 people. This is 11.4% of the national average of [Calculator(658,893 / 11.4%)] 5,763,868 people.
Input: A total of 252 qualifying matches were played, and 723 goals were scored (an average of 2.87 per match). This is three times less than the 2169 goals last year.
Output: A total of 252 qualifying matches were played, and 723 goals were scored (an average of [Calculator(723 / 252)] 2.87 per match). This is twenty goals more than the [Calculator(723 - 20)] 703 goals last year.
Input: I went to Paris in 1994 and stayed there until 2011, so in total, it was 17 years.
Output: I went to Paris in 1994 and stayed there until 2011, so in total, it was [Calculator(2011 - 1994)] 17 years.
Input: From this, we have 4 * 30 minutes = 120 minutes.
Output: From this, we have 4 * 30 minutes = [Calculator(4 * 30)] 120 minutes.
Input: {DEFAULT_PROMPT_INPUT_TAG}
Output:
"""

wikipedia_search_prompt = f"""
Your task is to complete a given piece of text. 
You can use a Wikipedia Search API to look up information. 
You can do so by writing "[WikiSearch(term)]" where "term" is the search term you want to look up. 
Here are some examples of API calls:
Input: The colors on the flag of Ghana have the following meanings: red is for the blood of martyrs, green for forests, and gold for mineral wealth.
Output: The colors on the flag of Ghana have the following meanings: red is for [WikiSearch("Ghana flag red meaning")] the blood of martyrs, green for forests, and gold for mineral wealth.
Input: But what are the risks during production of nanomaterials? Some nanomaterials may give rise to various kinds of lung damage.
Output: But what are the risks during production of nanomaterials? [WikiSearch("nanomaterial production risks")] Some nanomaterials may give rise to various kinds of lung damage.
Input: Metformin is the first-line drug for patients with type 2 diabetes and obesity.
Output: Metformin is the first-line drug for [WikiSearch("Metformin first-line drug")] patients with type 2 diabetes and obesity.
Input: {DEFAULT_PROMPT_INPUT_TAG}
Output:
"""

machine_translation_prompt = f"""
Your task is to complete a given piece of text by using a Machine Translation API.
You can do so by writing "[MT(text)]" where text is the text to be translated into English.
Here are some examples:
Input: He has published one book: O homem suprimido (“The Supressed Man”)
Output: He has published one book: O homem suprimido [MT(O homem suprimido)] (“The Supressed Man”)
Input: In Morris de Jonge’s Jeschuah, der klassische jüdische Mann, there is a description of a Jewish writer
Output: In Morris de Jonge’s Jeschuah, der klassische jüdische Mann [MT(der klassische jüdische Mann)], there is a description of a Jewish writer
Input: 南 京 高 淳 县 住 房 和 城 乡 建 设 局 城 市 新 区 设 计 a plane of reference Gaochun is one of seven districts of the provincial capital Nanjing
Output: [MT(南京高淳县住房和城乡建设局 城市新 区 设 计)] a plane of reference Gaochun is one of seven districts of the provincial capital Nanjing
Input: {DEFAULT_PROMPT_INPUT_TAG}
Output:
"""

calendar_prompt = f"""
Your task is to add calls to a Calendar API to a piece of text. 
The API calls should help you get information required to complete the text. 
You can call the API by writing "[Calendar()]" 
Here are some examples of API calls:
Input: Today is the first Friday of the year.
Output: Today is the first [Calendar()] Friday of the year.
Input: The president of the United States is Joe Biden.
Output: The president of the United States is [Calendar()] Joe Biden.
Input: The current day of the week is Wednesday.
Output: The current day of the week is [Calendar()] Wednesday.
Input: The number of days from now until Christmas is 30.
Output: The number of days from now until Christmas is [Calendar()] 30.
Input: The store is never open on the weekend, so today it is closed.
Output: The store is never open on the weekend, so today [Calendar()] it is closed.
Input: {DEFAULT_PROMPT_INPUT_TAG}
Output:
"""
