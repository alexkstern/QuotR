from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
from PyPDF2 import PdfReader
from tqdm import tqdm
import glob
import os
import gradio as gr
import openai
import subprocess

os.environ['OPENAI_API_KEY'] = 'PUT YOUR OPENAI API KEY HERE'
openai.api_key = 'PUT YOUR OPENAI API KEY HERE'


def similarity_search_interface(input):
    #Output string
    output = ""
    # Load the index
    try:
        with open('.\Data\store.pkl', 'rb') as file:
            # code that reads from the file
            index = pickle.load(file)
    except FileNotFoundError:
        output+="You have to upload your pdfs to the Data file and run the ingest.py file first"
        print(output)
        return output
    # Perform the similarity search with maximum source diversity
    result = index.max_marginal_relevance_search(input, k=5)

    # Compile the result for display
    
    for i, doc in enumerate(result):
        chunk = doc.page_content
        source = doc.metadata
        
        formatted_chunk = format_similarity_search(chunk)
        quote_eval= quote_evaluation(input,formatted_chunk)
        
        
        output += f"Quote {i+1}:\n{formatted_chunk}\n{source}\n\n"
        output += f"Does Quote {i+1} help your claim?\n{quote_eval}\n\n\n"
    if output=="":
        output+="There arent any quotes that resemble your claim"    
    return output



def format_similarity_search(input_str):
    prompt = f"Reformat the text to make it look nice without introducing any new words nor removing any. The output should be full quotes from the original text, correcting things like 'in decisive' to 'indecisive':\n'{input_str}'###"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=2000,
        n=1,
        stop='###',
        temperature=0.2,
    )
    formatted_result = response["choices"][0]["text"]
    return formatted_result


def quote_evaluation(claim,input_str):
    prompt= f"Does this quote: \n'{input_str}'\nhelp answer the statement '{claim}'###"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=2000,
        n=1,
        stop='###',
        temperature=0.2,
    )
    output = response["choices"][0]["text"]
    return output



with gr.Blocks() as demo:
    gr.Markdown("Find the quotes for your claims")
    with gr.Tab("Quote Searcher"):
        text_input = gr.Textbox(label="Claim or Statement")
        text_output = gr.Textbox(label="Quotes")
        text_button = gr.Button("Search for quotes that resemble your claim")
    with gr.Accordion("Open for instructions"):
        gr.Markdown("FIRST you must upload your relevant readings into the Data file (must be pdfs).\n\nThen you must run the ingest.py file.\n\nOnce these steps have been done you can start finding the quotes for your claims.\n\nHere is an example if your readings are related to international trade.\n\nEXAMPLE: Make statements or claims like 'The WTO is beneficial for trade liberalization'.\n\nThe quotes are simply the parts of your pdfs that MOST resemble your claim, which is why an evaluation for how relevant the quote is, is also provided.")

    text_button.click(similarity_search_interface, inputs=text_input, outputs=text_output)

demo.launch()



