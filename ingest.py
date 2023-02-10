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

os.environ['OPENAI_API_KEY'] = 'PUT YOUR OPENAI API KEY HERE'

def get_all_pdf_filenames(paths, recursive):
    extensions = ["pdf"]
    filenames = []
    for ext_name in extensions:
        ext = f"**/*.{ext_name}" if recursive else f"*.{ext_name}"
        for path in paths:
            filenames.extend(glob.glob(os.path.join(path, ext), recursive=recursive))
    return filenames


class Ingest():
    def __init__(
        self,
        # separator - The string used to separate the chunks in the final quote
        separator='\n',

        # chunk_overlap - The number of characters that should overlap between chunks
        chunk_overlap=200,

        # chunk_size - The number of characters in each chunk of the final quote
        chunk_size=200,
    ):
        self.splitter = CharacterTextSplitter(chunk_size=chunk_size, separator=separator, chunk_overlap=chunk_overlap)
        
    def __call__(self, path):
        ps = get_all_pdf_filenames([path], recursive=True) # get paths
        
        data = []
        sources = []
        for p in tqdm(ps): # extract data from paths
            reader = PdfReader(p)
            page = '\n'.join([reader.pages[i].extract_text() for i in range(len(reader.pages))])
            data.append(page)
            sources.append(p)

        docs = []
        metadatas = []
        for i, d in tqdm(enumerate(data)): # split text and make documents
            splits = self.splitter.split_text(d)
            if all(s != "" for s in splits):
                docs.extend(splits)
                metadatas.extend([{"source": sources[i]}] * len(splits))
                
        print(docs[2])
        print("\n\n\n\n")
        print(docs[3])
        print("\n\n\n\n")
        print(docs[4])
        print("\n\n\n\n")
                
        print("Extracting embeddings")
        store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
        
        with open(os.path.join(path, 'store.pkl'), "wb") as f:
            pickle.dump(store, f)
            
        print(f"Saved store at {os.path.join(path, 'store.pkl')}.")
            

if __name__ == '__main__':
    
    content_ingester = Ingest(chunk_size=750, separator='.')
    content_ingester('./Data')

