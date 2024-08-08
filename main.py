import getopt
import sys
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import analyzing_interface as ai



def read_api_keys(file_path):
    api_keys = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for line in lines:
        key, value = line.strip().split(' = ')
        api_keys[key] = value
    os.environ['OPENAI_API_KEY'] = api_keys['OPENAI_API_KEY']
    os.environ['ACTIVELOOP_TOKEN'] = api_keys['ACTIVELOOP_TOKEN']


def load_code(dir):
    docs = []
    for dirpath, dirnames, filenames in os.walk(dir):
        for file in filenames:
            if file.endswith('.java'):
                try:
                    loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                    docs.extend(loader.load_and_split())
                except Exception as e:
                    pass
    return docs


def split_text(docs):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
    return texts


def embedding(project_name, account_name):
    docs = load_code('./project')
    texts = split_text(docs)
    embeddings = OpenAIEmbeddings()
    DEEPLAKE_ACCOUNT_NAME = account_name
    DEEPLAKE_PROJECT_NAME = project_name
    db = DeepLake.from_documents(texts, embeddings,
                                 dataset_path=f"hub://{DEEPLAKE_ACCOUNT_NAME}/{DEEPLAKE_PROJECT_NAME}")
    return db


def load_embedding(project_name, account_name):
    DEEPLAKE_ACCOUNT_NAME = account_name
    DEEPLAKE_PROJECT_NAME = project_name
    embeddings = OpenAIEmbeddings()
    db = DeepLake(dataset_path=f"hub://{DEEPLAKE_ACCOUNT_NAME}/{DEEPLAKE_PROJECT_NAME}", embedding=embeddings,
                  read_only=True)
    return db


if __name__ == '__main__':
    read_api_keys('key.txt')
    emb = None
    account = None
    mode = None
    debug = None
    embeddings = OpenAIEmbeddings()
    try:
        options, remainder = getopt.getopt(
            sys.argv[1:],
            '',
            ['embedding=', 'account=', 'mode=', 'debug='])
    except getopt.GetoptError as err:
        print('ERROR:', err)
        sys.exit(1)

    for opt, arg in options:
        if opt == '--embedding':
            emb = arg
        if opt == '--account':
            account = arg
        if opt == '--mode':
            mode = arg
        if opt == '--debug':
            debug = arg
    try:
        if emb is None:
            raise ValueError("embedding is error.")
        elif account is None:
            raise ValueError("account is error.")
        elif mode is None:
            raise ValueError("mode is error.")
        elif debug is None:
            raise ValueError("debug is error.")

    except NameError:
        print("embedding or account is empty")

    if debug == 'true':
        print("set debug to true")
        os.environ['LANGCHAIN_DEBUG'] = '1'

    if mode == '1':
        embeddings = embedding(emb, account)
    elif mode == '2':
        embeddings = load_embedding(emb, account)
    elif mode == '3':
        embeddings = load_embedding(emb, account)
        if debug == 'true':
            print('loading embeddings successfully.')
        ai.analyze_interface(embeddings)
        if debug == 'true':
            print('analyzing embeddings successfully.')
