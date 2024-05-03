import json


with open("./default_user.json", "w") as f:
    # json_data = json.load(f)
    write_json_data = {"default_username_pred": "DEFAULT_NAME_PRED_", "now_id": 0}
    json.dump(write_json_data, f, indent=4)

with open("./all_user.json", "w") as f:
    # json_data = json.load(f)
    write_json_data = {"username": []}
    json.dump(write_json_data, f, indent=4)
