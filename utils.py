from scipy import spatial
import openai
from dotenv import load_dotenv
import os
import pandas as pd
import ast

load_dotenv(override=True)

def set_openai_key():
    API_KEY = os.getenv("API_KEY")
    openai.api_key = API_KEY

set_openai_key()


def strings_ranked_by_relatedness(
        query,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
        top_n = 5
    ):
        """Returns a list of strings and relatednesses, sorted from most related to least."""
        query_embedding_response = openai.embeddings.create(
            model= "text-embedding-3-small",
            input=query,
        )
    
        query_embedding = query_embedding_response.data[0].embedding
        
        df_kb = pd.read_csv("./kb/topics.csv")
        df_kb["embedding"] = df_kb['embedding'].apply(ast.literal_eval)

        strings_and_relatednesses = [
            (row["text"], relatedness_fn(query_embedding, row["embedding"]))
            for i, row in df_kb.iterrows()
        ]
        strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
        strings, relatednesses = zip(*strings_and_relatednesses)
        info_texts = [{"content":text, "relatedness": relat} for text, relat in zip(strings[:top_n], relatednesses[:top_n])]
        return info_texts