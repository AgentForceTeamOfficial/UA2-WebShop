import argparse, json, logging, random
import os
from pathlib import Path
from ast import literal_eval

from flask import (
    Flask,
    request,
    redirect,
    url_for,
    session,
)

from rich import print

from web_agent_site.engine.engine import (
    load_products,
    init_search_engine,
    convert_web_app_string_to_var,
    get_top_n_product_from_keywords,
    cf_rerank,
    dpp_rerank,
    get_product_per_page,
    map_action_to_html,
    END_BUTTON,
)
from web_agent_site.engine.goal import get_reward, get_goals
from web_agent_site.utils import (
    generate_mturk_code,
    setup_logger,
    DEFAULT_FILE_PATH,
    DEBUG_PROD_SIZE,
    ITEM_EMBEDDING_PATH,
    NEW_TASKS_PATH,
    PRIME_USER_PATH,
)

import random

app = Flask(__name__)

search_engine = None
all_info = None
all_products = None
product_item_dict = None
product_prices = None
attribute_to_asins = None
goals = None
weights = None
prime_user_score = None
user_history = dict()
user_sessions = dict()
user_log_dir = None
SHOW_ATTRS_TAB = False

IS_RERANKING = True
IS_DROP_ITEMS = False

OOPS_PROBABILITY = 0.9
OOPS_SEED = 1
random.seed(OOPS_SEED)

current_script_path = os.path.abspath(__file__)

base_dir = os.path.dirname(current_script_path)

all_user_path = os.path.join(base_dir, "all_user", "all_user.json")
default_user_path = os.path.join(base_dir, "all_user", "default_user.json")

app.config["SECRET_KEY"] = "agent_force"


@app.route("/")
def home():
    return redirect(
        url_for("index", session_id="user_0", task_id="task_0")
    )  # url_for('login'))

# Implement username input to differentiate between two users,
# as we need to track one user's historical actions.
# The username is stored in the session
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST" and "username_query" in request.form:
        username = request.form["username_query"]
        # when the input is empty
        # python web_agent_site/all_user/init_user.py to reset
        if username == "":
            with open(default_user_path, "r+") as f:
                json_data = json.load(f)
                username = json_data["default_username_pred"] + str(json_data["now_id"])
                json_data["now_id"] += 1

                f.seek(0)
                # empty the file
                f.truncate()
                json.dump(json_data, f, indent=4)

        with open(all_user_path, "r+") as f:
            json_data = json.load(f)
            all_user = json_data["username"]
            if username in all_user:
                return map_action_to_html(
                    "login",
                )
            else:
                json_data["username"].append(username)

            f.seek(0)
            # empty the file
            f.truncate()
            json.dump(json_data, f, indent=4)

        session["username"] = username  # save it in the session
        print(username)
        return redirect(
            url_for(
                "index",
                session_id="user_0",
                task_id="task_0",
            )
        )
    return map_action_to_html(
        "login",
    )

# For each user_x (session_id), needs to consecutively complete the instruction groups (task_0~49)
@app.route("/<session_id>/<task_id>", methods=["GET", "POST"])
def index(session_id, task_id):
    global user_log_dir
    global all_products, product_item_dict, product_prices, attribute_to_asins, search_engine, goals, weights, user_sessions, all_info, prime_user_score

    username = session.get("username")  # use session to get username
    if username:
        pass
    else:
        return redirect(url_for("login"))

    if search_engine is None:
        (
            all_products,
            product_item_dict,
            product_prices,
            attribute_to_asins,
            all_info,
            prime_user_score,
        ) = load_products(filepath=DEFAULT_FILE_PATH, num_products=DEBUG_PROD_SIZE)
        search_engine = init_search_engine(num_products=DEBUG_PROD_SIZE)
        goals = get_goals(
            all_products, product_prices, product_item_dict, NEW_TASKS_PATH
        )
        # input already shuffled
        # random.seed(233)
        # random.shuffle(goals)
        # weights = [goal['weight'] for goal in goals]

    # init user_history
    if session_id not in user_history:
        user_history[session_id] = {}

    if (session_id not in user_sessions) or (
        (session_id in user_sessions) and ("task_id" not in user_sessions[session_id])
    ):  # and ('user' in session_id) and ('task' in task_id):
        goal_dix = int(session_id.split("_")[-1])
        task_dix = int(task_id.split("_")[-1])
        # print('test', goal_dix, task_dix, goals)
        goal = goals[goal_dix][task_dix]
        instruction_text = goal["instruction_text"]
        user_sessions[session_id] = {
            "task_id": task_id,
            "goal": goal,
            "done": False,
        }
        if user_log_dir is not None:
            setup_logger(session_id, task_id, user_log_dir)
    elif (session_id in user_sessions) and (
        user_sessions[session_id]["task_id"] != task_id
    ):
        goal_dix = int(session_id.split("_")[-1])
        task_dix = int(task_id.split("_")[-1])
        goal = goals[goal_dix][task_dix]
        instruction_text = goal["instruction_text"]
        # keep the user_history
        user_sessions[session_id]["task_id"] = task_id
        user_sessions[session_id]["goal"] = goal
        user_sessions[session_id]["done"] = False
        if user_log_dir is not None:
            setup_logger(session_id, task_id, user_log_dir)
    else:
        instruction_text = user_sessions[session_id]["goal"]["instruction_text"]

    user_profile_text = instruction_text.split("|||")[0]
    real_instruction_text = instruction_text.split("|||")[1]
    if request.method == "POST" and "search_query" in request.form:
        keywords = request.form["search_query"].lower().split(" ")
        return redirect(
            url_for(
                "search_results",
                session_id=session_id,
                task_id=task_id,
                keywords=keywords,
                page=1,
            )
        )
    if user_log_dir is not None:
        logger = logging.getLogger(f"{session_id}_{task_id}")
        logger.info(
            json.dumps(
                dict(
                    page="index",
                    url=request.url,
                    goal=user_sessions[session_id]["goal"],
                )
            )
        )
    return map_action_to_html(
        "start",
        session_id=session_id,
        task_id=task_id,
        user_profile_text=user_profile_text,
        instruction_text=real_instruction_text,
    )

# add a new button that hints the instruction history so far (in reverse chronological order)
@app.route("/instruction_history/<session_id>/<task_id>/", methods=["GET", "POST"])
def instruction_history(session_id, task_id):
    instruction_history_list = user_sessions[session_id]["goal"]["instruction_history"]
    instruction_text = user_sessions[session_id]["goal"]["instruction_text"]
    user_profile_text = instruction_text.split("|||")[0]
    real_instruction_text = instruction_text.split("|||")[1]

    html = map_action_to_html(
        "instruction_history",
        session_id=session_id,
        task_id=task_id,
        user_profile_text=user_profile_text,
        instruction_text=real_instruction_text,
        instruction_history_list=instruction_history_list,
    )
    logger = logging.getLogger(f"{session_id}_{task_id}")
    logger.info(
        json.dumps(
            dict(
                page="instruction_history",
                url=request.url,
                goal=user_sessions[session_id]["goal"],
            )
        )
    )
    return html


@app.route(
    "/search_results/<session_id>/<task_id>/<keywords>/<page>", methods=["GET", "POST"]
)
def search_results(session_id, task_id, keywords, page):
    instruction_text = user_sessions[session_id]["goal"]["instruction_text"]
    user_profile_text = instruction_text.split("|||")[0]
    real_instruction_text = instruction_text.split("|||")[1]
    page = convert_web_app_string_to_var("page", page)
    keywords = convert_web_app_string_to_var("keywords", keywords)
    top_n_products = get_top_n_product_from_keywords(
        keywords,
        search_engine,
        all_products,
        product_item_dict,
        attribute_to_asins,
    )

    if IS_RERANKING:
        # TODO
        top_n_products = cf_rerank(
            top_n_products,
            user_history[session_id],
            prime_user_score,
        )
        top_n_products = dpp_rerank(
            top_n_products,
            user_history[session_id],
            all_info,
            ITEM_EMBEDDING_PATH,
        )
    products = get_product_per_page(top_n_products, page)

    # record history
    for product in products:
        if product["asin"] not in user_history[session_id]:
            # history of show / click / buy
            user_history[session_id][product["asin"]] = [0, 0, 0]
        user_history[session_id][product["asin"]][1] += 1

    html = map_action_to_html(
        "search",
        session_id=session_id,
        task_id=task_id,
        products=products,
        keywords=keywords,
        page=page,
        total=len(top_n_products),
        user_profile_text=user_profile_text,
        instruction_text=real_instruction_text,
    )
    logger = logging.getLogger(f"{session_id}_{task_id}")
    logger.info(
        json.dumps(
            dict(
                page="search_results",
                url=request.url,
                goal=user_sessions[session_id]["goal"],
                content=dict(
                    keywords=keywords,
                    search_result_asins=[p["asin"] for p in products],
                    page=page,
                ),
            )
        )
    )
    return html


@app.route(
    "/item_page/<session_id>/<task_id>/<asin>/<keywords>/<page>/<options>",
    methods=["GET", "POST"],
)
def item_page(session_id, task_id, asin, keywords, page, options):
    options = literal_eval(options)
    product_info = product_item_dict[asin]

    instruction_text = user_sessions[session_id]["goal"]["instruction_text"]
    user_profile_text = instruction_text.split("|||")[0]
    real_instruction_text = instruction_text.split("|||")[1]
    product_info["goal_instruction"] = instruction_text

    is_bad = random.random()
    if (not IS_DROP_ITEMS) or (IS_DROP_ITEMS and (is_bad <= OOPS_PROBABILITY)):
        html = map_action_to_html(
            "click",
            session_id=session_id,
            task_id=task_id,
            product_info=product_info,
            keywords=keywords,
            page=page,
            asin=asin,
            options=options,
            user_profile_text=user_profile_text,
            instruction_text=real_instruction_text,
            show_attrs=SHOW_ATTRS_TAB,
            user_history=user_history[session_id],
        )
    else:
        html = map_action_to_html(
            "item_oops",
            session_id=session_id,
            task_id=task_id,
            product_info=product_info,
            keywords=keywords,
            page=page,
            asin=asin,
            options=options,
            user_profile_text=user_profile_text,
            instruction_text=real_instruction_text,
            show_attrs=SHOW_ATTRS_TAB,
        )

    logger = logging.getLogger(f"{session_id}_{task_id}")
    logger.info(
        json.dumps(
            dict(
                page="item_page",
                url=request.url,
                goal=user_sessions[session_id]["goal"],
                content=dict(
                    keywords=keywords,
                    page=page,
                    asin=asin,
                    options=options,
                ),
            )
        )
    )
    # print(user_history[session_id])
    return html


@app.route(
    "/item_sub_page/<session_id>/<task_id>/<asin>/<keywords>/<page>/<sub_page>/<options>",
    methods=["GET", "POST"],
)
def item_sub_page(session_id, task_id, asin, keywords, page, sub_page, options):
    options = literal_eval(options)
    product_info = product_item_dict[asin]

    instruction_text = user_sessions[session_id]["goal"]["instruction_text"]
    user_profile_text = instruction_text.split("|||")[0]
    real_instruction_text = instruction_text.split("|||")[1]
    product_info["goal_instruction"] = instruction_text

    html = map_action_to_html(
        f"click[{sub_page}]",
        session_id=session_id,
        task_id=task_id,
        product_info=product_info,
        keywords=keywords,
        page=page,
        asin=asin,
        options=options,
        user_profile_text=user_profile_text,
        instruction_text=real_instruction_text,
    )
    logger = logging.getLogger(f"{session_id}_{task_id}")
    logger.info(
        json.dumps(
            dict(
                page="item_sub_page",
                url=request.url,
                goal=user_sessions[session_id]["goal"],
                content=dict(
                    keywords=keywords,
                    page=page,
                    asin=asin,
                    options=options,
                ),
            )
        )
    )
    return html


@app.route("/done/<session_id>/<task_id>/<asin>/<options>", methods=["GET", "POST"])
def done(session_id, task_id, asin, options):
    options = literal_eval(options)
    goal = user_sessions[session_id]["goal"]
    purchased_product = product_item_dict[asin]
    price = product_prices[asin]

    reward, reward_info = get_reward(
        purchased_product, goal, price=price, options=options, verbose=True
    )
    user_sessions[session_id]["done"] = True
    user_sessions[session_id]["reward"] = reward
    # print(user_sessions)

    logger = logging.getLogger(f"{session_id}_{task_id}")
    logger.info(
        json.dumps(
            dict(
                page="done",
                url=request.url,
                goal=goal,
                content=dict(
                    asin=asin,
                    options=options,
                    price=price,
                ),
                reward=reward,
                reward_info=reward_info,
            )
        )
    )
    del logging.root.manager.loggerDict[f"{session_id}_{task_id}"]

    return map_action_to_html(
        f"click[{END_BUTTON}]",
        session_id=session_id,
        task_id=task_id,
        reward=reward,
        asin=asin,
        options=options,
        reward_info=reward_info,
        query=purchased_product["query"],
        category=purchased_product["category"],
        product_category=purchased_product["product_category"],
        goal_attrs=user_sessions[session_id]["goal"]["attributes"],
        purchased_attrs=purchased_product["Attributes"],
        goal=goal,
        mturk_code=generate_mturk_code(session_id),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebShop flask app backend configuration"
    )
    parser.add_argument(
        "--log", action="store_true", help="Log actions on WebShop in trajectory file"
    )
    parser.add_argument(
        "--attrs", action="store_true", help="Show attributes tab in item page"
    )
    parser.add_argument(
        "--rerank", action="store_true", help="Whether to rerank or not"
    )
    parser.add_argument(
        "--dropitems", action="store_true", help="Whether to randomly drop some items"
    )

    args = parser.parse_args()
    if args.log:
        user_log_dir = Path("user_session_logs/mturk")
        user_log_dir.mkdir(parents=True, exist_ok=True)
    SHOW_ATTRS_TAB = args.attrs

    IS_RERANKING = args.rerank
    IS_DROP_ITEMS = args.dropitems

    app.run(host="0.0.0.0", port=5000)
