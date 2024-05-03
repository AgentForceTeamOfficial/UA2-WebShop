"""
"""
import os
import re
import json
import random
from collections import defaultdict
from ast import literal_eval
from decimal import Decimal
import math
import numpy as np

import cleantext
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from flask import render_template_string
from rich import print
from pyserini.search.lucene import LuceneSearcher

from web_agent_site.utils import (
    BASE_DIR,
    DEFAULT_FILE_PATH,
    DEFAULT_REVIEW_PATH,
    DEFAULT_ATTR_PATH,
    HUMAN_ATTR_PATH,
    ITEM_EMBEDDING_PATH,
    ITEM_ID2IDX_PATH,
    PRIME_USER_PATH,
)

TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

SEARCH_RETURN_N = 50
PRODUCT_WINDOW = 10
TOP_K_ATTR = 10

TOP_K_PRODUCT = 50

END_BUTTON = 'Buy Now'
NEXT_PAGE = 'Next >'
PREV_PAGE = '< Prev'
BACK_TO_SEARCH = 'Back to Search'

ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html',
    'Features': 'features_page.html',
    'Reviews': 'review_page.html',
    'Attributes': 'attributes_page.html',
}

def map_action_to_html(action, **kwargs):
    action_name, action_arg = parse_action(action)

    if action_name == 'start':
        path = os.path.join(TEMPLATE_DIR, 'search_page.html')
        html = render_template_string(
            read_html_template(path=path),
            session_id=kwargs['session_id'],
            task_id=kwargs['task_id'],
            instruction_text=kwargs['instruction_text'],
            user_profile_text=kwargs['user_profile_text'],
            # username=kwargs['username'],
        )
    elif action_name == 'search':
        path = os.path.join(TEMPLATE_DIR, 'results_page.html')
        html = render_template_string(
            read_html_template(path=path),
            session_id=kwargs['session_id'],
            task_id=kwargs['task_id'],
            products=kwargs['products'],
            keywords=kwargs['keywords'],
            page=kwargs['page'],
            total=kwargs['total'],
            instruction_text=kwargs['instruction_text'],
            user_profile_text=kwargs['user_profile_text'],
            # username=kwargs['username'],
        )
    elif action_name == 'login':
        path = os.path.join(TEMPLATE_DIR, 'login_page.html')
        html = render_template_string(
            read_html_template(path=path),
        )
    elif action_name == 'instruction_history':
        path = os.path.join(TEMPLATE_DIR, 'instruction_history_page.html')
        html = render_template_string(
            read_html_template(path=path),
            session_id=kwargs['session_id'],
            task_id=kwargs['task_id'],
            instruction_text=kwargs['instruction_text'],
            user_profile_text=kwargs['user_profile_text'],
            instruction_history_list=kwargs['instruction_history_list']                        
        )        
    elif action_name == 'item_oops':
        path = os.path.join(TEMPLATE_DIR, 'item_oops_page.html')
        html = render_template_string(
            read_html_template(path),
            session_id=kwargs['session_id'],
            task_id=kwargs['task_id'],
            product_info=kwargs['product_info'],
            keywords=kwargs['keywords'],
            page=kwargs['page'],
            asin=kwargs['asin'],
            options=kwargs['options'],
            instruction_text=kwargs.get('instruction_text'),
            show_attrs=kwargs['show_attrs'],
            user_profile_text=kwargs['user_profile_text'],
            # username=kwargs['username'],
        )       
    elif action_name == 'click' and action_arg == END_BUTTON:
        path = os.path.join(TEMPLATE_DIR, 'done_page.html')
        html = render_template_string(
            read_html_template(path),
            session_id=kwargs['session_id'],
            task_id=kwargs['task_id'],
            reward=kwargs['reward'],
            asin=kwargs['asin'],
            options=kwargs['options'],
            reward_info=kwargs.get('reward_info'),
            goal_attrs=kwargs.get('goal_attrs'),
            purchased_attrs=kwargs.get('purchased_attrs'),
            goal=kwargs.get('goal'),
            mturk_code=kwargs.get('mturk_code'),
            query=kwargs.get('query'),
            category=kwargs.get('category'),
            product_category=kwargs.get('product_category'),
            # username=kwargs['username'],
        )
        
    elif action_name == 'click' and action_arg in ACTION_TO_TEMPLATE:
        path = os.path.join(TEMPLATE_DIR, ACTION_TO_TEMPLATE[action_arg])
        html = render_template_string(
            read_html_template(path),
            session_id=kwargs['session_id'],
            task_id=kwargs['task_id'],
            product_info=kwargs['product_info'],
            keywords=kwargs['keywords'],
            page=kwargs['page'],
            asin=kwargs['asin'],
            options=kwargs['options'],
            instruction_text=kwargs.get('instruction_text'),
            user_profile_text=kwargs['user_profile_text'],
            # username=kwargs['username'],
        )
    elif action_name == 'click':
        path = os.path.join(TEMPLATE_DIR, 'item_page.html')
        html = render_template_string(
            read_html_template(path),
            session_id=kwargs['session_id'],
            task_id=kwargs['task_id'],
            product_info=kwargs['product_info'],
            keywords=kwargs['keywords'],
            page=kwargs['page'],
            asin=kwargs['asin'],
            options=kwargs['options'],
            instruction_text=kwargs.get('instruction_text'),
            user_profile_text=kwargs['user_profile_text'],
            show_attrs=kwargs['show_attrs'],
            # username=kwargs['username'],
        )
        kwargs['user_history'][kwargs['asin']][0] += 1
    else:
        raise ValueError('Action name not recognized.')
    return html


def read_html_template(path):
    with open(path) as f:
        template = f.read()
    return template


def parse_action(action):
    """
    Parse action string to action name and its arguments.
    """
    pattern = re.compile(r'(.+)\[(.+)\]')
    m = re.match(pattern, action)
    if m is None:
        action_name = action
        action_arg = None
    else:
        action_name, action_arg = m.groups()
    return action_name, action_arg


def convert_web_app_string_to_var(name, string):
    if name == 'keywords':
        keywords = string
        if keywords.startswith('['):
            keywords = literal_eval(keywords)
        else:
            keywords = [keywords]
        var = keywords
    elif name == 'page':
        page = string
        page = int(page)
        var = page
    else:
        raise ValueError('Name of variable not recognized.')
    return var


def get_top_n_product_from_keywords(
        keywords,
        search_engine,
        all_products,
        product_item_dict,
        attribute_to_asins=None,
    ):
    if keywords[0] == '<r>':
        top_n_products = random.sample(all_products, k=SEARCH_RETURN_N)
    elif keywords[0] == '<a>':
        attribute = ' '.join(keywords[1:]).strip()
        asins = attribute_to_asins[attribute]
        top_n_products = [p for p in all_products if p['asin'] in asins]
    elif keywords[0] == '<c>':
        category = keywords[1].strip()
        top_n_products = [p for p in all_products if p['category'] == category]
    elif keywords[0] == '<q>':
        query = ' '.join(keywords[1:]).strip()
        top_n_products = [p for p in all_products if p['query'] == query]
    else:
        keywords = ' '.join(keywords)
        hits = search_engine.search(keywords, k=SEARCH_RETURN_N)
        docs = [search_engine.doc(hit.docid) for hit in hits]
        top_n_asins = [json.loads(doc.raw())['id'] for doc in docs]
        top_n_products = [product_item_dict[asin] for asin in top_n_asins if asin in product_item_dict]
    return top_n_products
    # return all_products[0:1] # for debug


def get_product_per_page(top_n_products, page):
    return top_n_products[(page - 1) * PRODUCT_WINDOW:page * PRODUCT_WINDOW]


def generate_product_prices(all_products):
    product_prices = dict()
    for product in all_products:
        asin = product['asin']
        pricing = product['pricing']
        if not pricing:
            price = 100.0
        elif len(pricing) == 1:
            price = pricing[0]
        else:
            price = random.uniform(*pricing[:2])
        product_prices[asin] = price
    return product_prices


def init_search_engine(num_products=None):
    if num_products == 100:
        indexes = 'indexes_100'
    elif num_products == 1000:
        indexes = 'indexes_1k'
    elif num_products == 100000:
        indexes = 'indexes_100k'
    elif num_products is None:
        indexes = 'indexes'
    else:
        raise NotImplementedError(f'num_products being {num_products} is not supported yet.')
    search_engine = LuceneSearcher(os.path.join(BASE_DIR, f'./search_engine/{indexes}'))
    return search_engine


def clean_product_keys(products):
    for product in products:
        product.pop('product_information', None)
        product.pop('brand', None)
        product.pop('brand_url', None)
        product.pop('list_price', None)
        product.pop('availability_quantity', None)
        product.pop('availability_status', None)
        product.pop('total_reviews', None)
        product.pop('total_answered_questions', None)
        product.pop('seller_id', None)
        product.pop('seller_name', None)
        product.pop('fulfilled_by_amazon', None)
        product.pop('fast_track_message', None)
        product.pop('aplus_present', None)
        product.pop('small_description_old', None)
    print('Keys cleaned.')
    return products


def load_products(filepath, num_products=None, human_goals=True):
    # TODO: move to preprocessing step -> enforce single source of truth
    with open(filepath) as f:
        products = json.load(f)
    print('Products loaded.')
    products = clean_product_keys(products)
    # with open(DEFAULT_REVIEW_PATH) as f:
    #     reviews = json.load(f)
    all_reviews = dict()
    all_ratings = dict()
    # for r in reviews:
    #     all_reviews[r['asin']] = r['reviews']
    #     all_ratings[r['asin']] = r['average_rating']

    if human_goals:
        with open(HUMAN_ATTR_PATH) as f:
            human_attributes = json.load(f)
    with open(DEFAULT_ATTR_PATH) as f:
        attributes = json.load(f)
    with open(HUMAN_ATTR_PATH) as f:
        human_attributes = json.load(f)
    print('Attributes loaded.')

    asins = set()
    all_products = []
    attribute_to_asins = defaultdict(set)
    if num_products is not None:
        # using item_shuffle.json, we assume products already shuffled
        products = products[:num_products]
    for i, p in tqdm(enumerate(products), total=len(products)):
        asin = p['asin']
        if asin == 'nan' or len(asin) > 10:
            continue

        if asin in asins:
            continue
        else:
            asins.add(asin)

        products[i]['category'] = p['category']
        products[i]['query'] = p['query']
        products[i]['product_category'] = p['product_category']

        products[i]['Title'] = p['name']
        products[i]['Description'] = p['full_description']
        products[i]['Reviews'] = all_reviews.get(asin, [])
        products[i]['Rating'] = all_ratings.get(asin, 'N.A.')
        for r in products[i]['Reviews']:
            if 'score' not in r:
                r['score'] = r.pop('stars')
            if 'review' not in r:
                r['body'] = ''
            else:
                r['body'] = r.pop('review')
        products[i]['BulletPoints'] = p['small_description'] \
            if isinstance(p['small_description'], list) else [p['small_description']]

        pricing = p.get('pricing')
        if pricing is None or not pricing:
            pricing = [100.0]
            price_tag = '$100.0'
        else:
            pricing = [
                float(Decimal(re.sub(r'[^\d.]', '', price)))
                for price in pricing.split('$')[1:]
            ]
            if len(pricing) == 1:
                price_tag = f"${pricing[0]}"
            else:
                price_tag = f"${pricing[0]} to ${pricing[1]}"
                pricing = pricing[:2]
        products[i]['pricing'] = pricing
        products[i]['Price'] = price_tag

        options = dict()
        customization_options = p['customization_options']
        option_to_image = dict()
        if customization_options:
            for option_name, option_contents in customization_options.items():
                if option_contents is None:
                    continue
                option_name = option_name.lower()

                option_values = []
                for option_content in option_contents:
                    option_value = option_content['value'].strip().replace('/', ' | ').lower()
                    option_image = option_content.get('image', None)

                    option_values.append(option_value)
                    option_to_image[option_value] = option_image
                options[option_name] = option_values
        products[i]['options'] = options
        products[i]['option_to_image'] = option_to_image

        # without color, size, price, availability
        # if asin in attributes and 'attributes' in attributes[asin]:
        #     products[i]['Attributes'] = attributes[asin]['attributes']
        # else:
        #     products[i]['Attributes'] = ['DUMMY_ATTR']
        # products[i]['instruction_text'] = \
        #     attributes[asin].get('instruction', None)
        # products[i]['instruction_attributes'] = \
        #     attributes[asin].get('instruction_attributes', None)

        # without color, size, price, availability
        if asin in attributes and 'attributes' in attributes[asin]:
            products[i]['Attributes'] = attributes[asin]['attributes']
        else:
            products[i]['Attributes'] = ['DUMMY_ATTR']
            
        if human_goals:
            if asin in human_attributes:
                products[i]['instructions'] = human_attributes[asin]
        else:
            products[i]['instruction_text'] = \
                attributes[asin].get('instruction', None)

            products[i]['instruction_attributes'] = \
                attributes[asin].get('instruction_attributes', None)

        products[i]['MainImage'] = p['images'][0]
        products[i]['query'] = p['query'].lower().strip()

        all_products.append(products[i])

    for p in all_products:
        for a in p['Attributes']:
            attribute_to_asins[a].add(p['asin'])

    product_item_dict = {p['asin']: p for p in all_products}
    product_prices = generate_product_prices(all_products)
    # all_embeddings = np.load(DEFAULT_EMBEDDING_PATH)

    with open(ITEM_ID2IDX_PATH, 'r') as f:
        all_info = json.load(f)
    
    with open(PRIME_USER_PATH, 'r') as f:
        prime_user_score = json.load(f)    
    
    return all_products, product_item_dict, product_prices, attribute_to_asins, all_info, prime_user_score

# dpp-based reranking based on self historical actions
# https://arxiv.org/abs/1709.05135
def dpp_rerank(
        top_n_products,
        user_history,
        all_info, 
        ebd_path,
        top_k=TOP_K_PRODUCT,
        ):
    
    eps = 1e-10
    # The hyperparameter `lam` balances the weight between dpp_rank (based on self historical actions) and cf_rank (based on other users)
    lam = 0.8 
    max_ctr = 0
    for v in user_history.values():
        if v[0] / v[1] > max_ctr:
            max_ctr = v[0] / v[1]
    if max_ctr == 0:
        max_ctr = 1
    
    item_list = [(lam * v + (1 - lam) * min(1, user_history[k['asin']][0] / (user_history[k['asin']][1] * max_ctr)), k) if k['asin'] in user_history.keys() else (lam * v, k) 
                 for (k, v) in top_n_products]
    item_list = sorted(item_list, key=lambda x: x[0], reverse=True)
    
    all_embeddings = None
    for item in item_list:
        # retrieve embedding matrices from pre-computed embedding files
        now_item_id  = item[1]['asin']
        now_item_idx = all_info[now_item_id]
        now_dir_st   = (now_item_idx // 50000) * 50000
        now_dir_en   = min(now_dir_st + 50000, 1181437)
        now_npy_st   = (now_item_idx // 10) * 10
        now_npy_en   = min(now_npy_st + 10, 1181437)
        now_npy_path = os.path.join(ebd_path, f"{now_dir_st}-{now_dir_en}/embeddings-{now_npy_st}-{now_npy_en}.npy")
        now_ebd      = np.load(now_npy_path)[(now_item_idx%10):(now_item_idx%10 + 1),:]
        if all_embeddings is not None:
            all_embeddings = np.concatenate((all_embeddings, now_ebd), axis=0)
        else:
            all_embeddings = now_ebd  

    pairwise_sim = get_item_sim(all_embeddings)
    # print(pairwise_sim)
    item_score = np.array([v for v, _ in item_list])
    item_len = len(item_score)
    kernel = item_score.reshape((item_len, 1)) * pairwise_sim * item_score.reshape((1, item_len))
    
    cis = np.zeros((top_k, item_len))
    di2s = np.copy(np.diag(kernel))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < top_k:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel[selected_item, : ]
        eis = (elements - np.dot(ci_optimal, cis[ :k, :])) / di_optimal
        cis[k, : ] = eis
        di2s -= np.square(eis)
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < eps:
            break
        selected_items.append(selected_item)

    top_k_products = []
    res_products = []
    for rank, (_, item) in enumerate(item_list):
        if rank in selected_items:
            top_k_products.append(item)
        else:
            res_products.append(item)
    return top_k_products + res_products

def get_item_sim(all_embeddings):
    '''
    - description embedding
    '''
    limit = 1e-8
    print(all_embeddings.shape)
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    zero_mask = norms <= limit
    zero_embeddings = np.zeros_like(all_embeddings)
    all_embeddings = np.where(zero_mask, zero_embeddings, all_embeddings / norms)
    # all_embeddings = np.where(norms == 0, all_embeddings, all_embeddings / norms)
    pairwise_sim = np.dot(all_embeddings, all_embeddings.T)
    np.fill_diagonal(pairwise_sim, 1)
    return pairwise_sim

# collaborative filtering based on other users (simulated by ChatGPT role playing). 
# The weight file lies at PRIME_USER_PATH
def cf_rerank(
    top_n_products, 
    user_history,
    data,
    top_k=TOP_K_PRODUCT,
    ): 

    num_users = len(data)
    # ctr
    max_ctr = 0
    for v in user_history.values():
        if v[0] / v[1] > max_ctr:
            max_ctr = v[0] / v[1]
    if max_ctr == 0:
        max_ctr = 1
        
    user = {k: v[0] / (v[1] * max_ctr) if v[1] else 0 for (k, v) in user_history.items()}
    similarity = np.zeros(num_users)
    product_score = np.zeros((num_users, top_k))
    for i, scores in enumerate(data.values()):
        for j, product in enumerate(top_n_products):
            product_score[i][j] = scores[product['asin']] if product['asin'] in scores.keys() else 0
            
    for i, prime_user in enumerate(data.values()):
        similarity[i] = pearson_similarity(prime_user, user)
    
    print(similarity, product_score)
    cf_score = np.dot(similarity, product_score) / np.sum(similarity) if np.sum(similarity) else np.zeros(top_k)
    
    top_n_products = [(p, cf_score[i]) for i, p in enumerate(top_n_products)]
    
    return top_n_products

# compute similarity between two users
def pearson_similarity(user1, user2):
    common_items = set(user1) & set(user2)
    if len(common_items) == 0:
        return 0
    sumXY = 0.0
    sumX = 0.0
    sumY = 0.0
    sumX2 = 0.0
    sumY2 = 0.0
    n = len(common_items)
    for item in common_items:
        x = user1[item]
        y = user2[item]
        sumXY += x * y
        sumX += x
        sumY += y
        sumX2 += x ** 2
        sumY2 += y ** 2
    molecule = sumXY - (sumX * sumY) / n
    denominator = np.sqrt((sumX2 - (sumX ** 2) / n) * (sumY2 - (sumY ** 2) / n))
    if denominator == 0:
        return 0
    else:
        similarity = molecule / denominator
        return similarity
