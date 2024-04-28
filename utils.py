from scipy import spatial
import openai
from dotenv import load_dotenv
import os
import pandas as pd
import ast
import tiktoken

load_dotenv(override=True)
GPT_MODEL = "gpt-3.5-turbo"

def set_openai_key():
    API_KEY = os.getenv("API_KEY")
    openai.api_key = API_KEY


set_openai_key()

def count_num_tokens(text, model = GPT_MODEL):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def join_docs(query, documents, token_budget):
    instrucction = """Proporciona una respuesta concisa y significativa al siguiente mensaje del usuario, considerando el contexto del historial del diálogo en curso. Utiliza solo la información entre tres comillas invertidas para responder de manera informativa. Evita proporcionar datos no respaldados. Usa máximo 100 palabras."""
        
    mensaje_user = f"""Mensaje del usuario: {query}"""

    information = ""

    for doc in documents:
        text = doc["content"]
        #template_information = f"""\nInformacion: ```{information}```\n"""
        template_information = f"\nInformación: ```{information + text}```\n"
        prompt_response_to_query = instrucction + template_information + mensaje_user

        if count_num_tokens(prompt_response_to_query, model=GPT_MODEL) > token_budget:
            break

        information += "\n"+ text
    return information

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