from algorithm import get_all_suggestions, final_recs_based_on_answers
from flask import Flask, request, jsonify

import pandas as pd

from functools import cache
from json import loads

app = Flask(__name__)

with open("books_clustered.csv", newline='') as f:
    db = pd.read_csv(f).dropna(subset=["categories", "description", "thumbnail", "average_rating"]).fillna(0)

db['metadata'] = db.apply(lambda x: (''.join(x['authors']) + ' ' + ''.join(x['categories'])).lower(), axis=1)


@cache
def find_matching(query: bytes):
    remaining = db.copy()

    start_index = 0
    count = 20
    for k, v in loads(query).items():
        match (k):
            case "authors" | "categories":
                remaining = remaining[remaining[k].isin(v)]
            case "after_year":
                remaining = remaining[remaining["published_year"] >= v]
            case "before_year":
                remaining = remaining[remaining["published_year"] <= v]
            case "above_rating":
                remaining = remaining[remaining["average_rating"] >= v]
            case "under_rating":
                remaining = remaining[remaining["average_rating"] <= v]
            case "count":
                count = v
            case "start_index":
                start_index = v
            case _:
                remaining = remaining[remaining[k] == v]

    matches = remaining.to_dict("records")[start_index:start_index + count]
    return {
        "total": len(remaining),
        "count": min(count, len(matches)),
        "start_index": start_index,
        "books": matches
    }


@app.route("/search", methods=["POST"])
def search():
    return jsonify(find_matching(request.get_data()))


@app.route("/suggest", methods=["POST"])
def suggest():
    """take user then send books"""
    data = loads(request.get_data())

    return jsonify(get_all_suggestions(db, data["user"]).to_dict("records"))


@app.route("/recommend", methods=["POST"])
def recommend():
    """get back ratings and send final suggestions"""
    data = loads(request.get_data())

    return jsonify({
        "user": data["user"],  # user is modified inside final_recs_based_on_answers
        "books": final_recs_based_on_answers(db, data["user"], data["likes"], pd.DataFrame(data["books"])).to_dict("records")
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
