import json
from pprint import pprint
import numpy as np
import tqdm
import pandas as pd
from collections import Counter
import requests
from pprint import pprint
import time
import xml.etree.ElementTree as ET



with open("modeldb-metadata.json") as peach:
    data_lib = json.load(peach)

#find the list of all "object_id" and store it in "unique_list"
def find_within_item(item):
        return item['object_id']


def find_within_list(list_of_items):
    try:
        s=[]
        for item in list_of_items:
            s.append(find_within_item(item))
        return s
    except:
        return []
    

def find_within_model(model):
    s=[]
    for v in model.values():
        s.extend(find_within_list(v))
    return s

def find_within_data(dat):
    s={}
    for key,val in dat.items():
        s[key]=find_within_model(val)
    return s


def find_meta_within_data(dat):
    s=[]
    for v in dat.values():
        s.extend(find_within_model(v))
    return s

#link each model ID with its multiple model paper ID (m-n relationship)
# model_paper={}
# for item,val in data_lib.items():
#     try:    
#         model_paper[item]=find_within_list(val["model_paper"])
#     except:
#         model_paper[item]=[]


#print(model_paper['232875'])
# pprint(model_paper)
#print(find_within_list(data_lib["232876"]["model_paper"]))

# with open("Cited_paper.json") as cucumber:
#     cite_lib = json.load(cucumber) #cite_lib is a dict with key as model paper iD, and its values are the bundle of things including the paper it cites

#pprint(cite_lib['232876'])


# for each model paper id, gets its corresponding pubmed id
def get_pubmed_id(item):
    for key,val in item.items():
        if key == "pubmed_id":
            return val["value"]
        

# pmid_dct={} #pmid_dct is a dict with key as model iD, and its values are the list of its model papers' corresponding pubmed id. 
# # model_tofix={}
# for model_id, paper_id in tqdm.tqdm(cite_lib.items()):
#     try:
#         pmid_dct[model_id]=[]
#         # model_tofix[model_id]=[]
#         for s in paper_id.values():
#             pmid_dct[model_id].append(get_pubmed_id(s))
#             # if get_pubmed_id(s)==None:
#             #     model_tofix[model_id].append(s['id'])
#     except:
#         pmid_dct[model_id]=[]

# for k,v in model_tofix.items():
#     if v!=[]:
#         pprint(str(k)+str(model_tofix[k]))



# for each model paper ID, taking its corresponding pmids and getting abstracts and titles, then you'd call process_pmids and it would return a data structure
def lookup_pmids(pmids, delay_interval=1):
    time.sleep(delay_interval)
    return ET.fromstring(
        requests.post(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            data={
                "db": "pubmed",
                "retmode": "xml",
                "id": ",".join(str(pmid) for pmid in pmids),
            },
        ).text
    )


def parse_paper(paper):
    abstract_block = paper.find(".//Abstract")
    mesh_heading = paper.find(".//MeshHeadingList")
    try:
        pmid = int(paper.find(".//PMID").text)
    except AttributeError:
        raise Exception("Bad paper? " + ET.tostring(paper, method="text").decode())
    title = paper.find(".//ArticleTitle")
    if title is None:
        title = paper.find(".//BookTitle")

    if abstract_block is not None:
        abstract = [
            {
                "section": item.get("Label"),
                "text": ET.tostring(item, method="text").decode(),
            }
            for item in abstract_block
            if item.tag == "AbstractText"
        ]
    else:
        abstract = ""
    assert title is not None
    title = ET.tostring(title, method="text").decode()
    mesh = []
    if mesh_heading is not None:
        for item in mesh_heading:
            item = item.find("DescriptorName")
            if item is not None:
                mesh.append(item.text)
    return pmid, {"mesh": mesh, "AbstractText": abstract, "ArticleTitle": title}


def process_pmids(pmids, delay_interval=1):
    results = {}
    papers = lookup_pmids(pmids, delay_interval=delay_interval)
    for paper in papers:
        pmid, parsed_paper = parse_paper(paper)
        results[pmid] = parsed_paper
    return results




# # pubmed_papers is a dictionary where the key is the model ID, and its value is a list of model paper ID and its assocaited data structure from PubMed
# pubmed_papers={}
# for key, val in tqdm.tqdm(pmid_dct.items()):
#     try:
#         pubmed_papers[key]=process_pmids(val, delay_interval=1)
#     except:
#         print(f"bad model: {key}")


# with open("pubmed_papers.json", "w") as apple:
#     apple.write(json.dumps(pubmed_papers))



# build a persistent dictionary (i.e. it only does the calculation once) for LinkBERT embeddings



from transformers import AutoTokenizer, BertAdapterModel
from adapter_transformers import adapter_hub
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('malteos/scincl')
model = BertAdapterModel.from_pretrained("malteos/scincl")

# Load the adapter
adapter_name = model.load_adapter("allenai/scirepeval_adapters_clf", source="hf", set_active=True)



#loading "pubmed_papers" which is a dictionary where the key is the model ID, and its value is a list of model paper ID and its assocaited data structure from PubMed
with open("pubmed_papers.json") as peach:
    pubmed_papers_0 = json.load(peach)

# print(len(pubmed_papers_0))

pubmed_papers={}
for val in pubmed_papers_0.values():
    for key1,val1 in val.items():
        pubmed_papers[key1]=val1

# print(len(pubmed_papers))

# pprint(pubmed_papers)



# for each pubmed id, gets its corresponding abstract
def get_abstract(item):
    for key,val in item.items():
        if key == "AbstractText":
            try:
                return val[0]["text"]
            except:
                return ''
        

import shelve

# Load the input data
with shelve.open("yujie-scincl-adapters-clf") as embeddings:
    for pmid, paper in tqdm.tqdm(pubmed_papers.items()):
        if pmid not in embeddings:
            data = [paper["ArticleTitle"] + tokenizer.sep_token + get_abstract(paper)]
            inputs = tokenizer(
                data, padding=True, truncation=True, return_tensors="pt", max_length=512
            )

            # Generate embeddings using the adapter
            with adapter_hub.adapter_scope(model, adapter_name):
                result = model(**inputs)
                embeddings[pmid] = result.last_hidden_state[:, 0, :].detach().numpy()[0]


