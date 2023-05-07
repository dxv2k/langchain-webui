SUMMARY_DEVELOPMENT_TEMPLATE_PROMPT = """[Summary]: [Write a concise summary of the review which is approximately 25-35% of the length, with no paragraphs, and designed so that if you were reading a high volume of these summaries, you would get a good understanding of the review.]
[Sentiment] : [One sentence summarising the sentiment of the review, taking account of the text and the scores].
[Positive Quote]: [Set out one positive direct quote from the review which represents the review well.]
[Negative Quote]: [Set out one negative direct quote, if there is one.]
[Topic Quotes]: [Set out any short, direct quotes on facilitates, location, building management, design, pets or children, and label each quote accordingly.]
[Topics]: [Set out a list of topics, described in a single word, which are covered in the review, with no more than 6. If there are more than 6, then just list the 6 most prevalent.]
""" 


def _get_header_prompt(development_id: str | int) -> str: 
    header = f"This is the review of development {development_id}\n"
    return header 


def get_summary_prompt(development_id: str | int, custom_body_prompt: str = None) -> str: 
    header =  _get_header_prompt(development_id=development_id)
    if custom_body_prompt: 
        prompt = header + custom_body_prompt 
        return prompt
    return header + SUMMARY_DEVELOPMENT_TEMPLATE_PROMPT