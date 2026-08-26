from typing import Tuple, Any
from openai import OpenAI

from config import OPENAI_API_KEY, USER_LOCATION, get_logger

logger = get_logger(__name__)

def generate(system_msg, user_msg):
    logger.info("Calling LLM model=gpt-5-search-api ...")

    client = OpenAI(api_key=OPENAI_API_KEY)

    resp = client.chat.completions.create(
        model="gpt-5-search-api",
        web_search_options={"user_location": USER_LOCATION},
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    text = resp.choices[0].message.content
    citations = None
    return str(text), citations
