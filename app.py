# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import numpy as np
# import tensorflow as tf
# import pickle

# app = Flask(__name__)
# CORS(app)  # Cho phép gọi từ frontend

# # --- TẢI MÔ HÌNH VÀ DỮ LIỆU ---
# model = tf.keras.models.load_model("ncf_model_with_title_watchtime")

# with open("mappings_with_title_watchtime.pkl", "rb") as f:
#     data = pickle.load(f)

# user2idx = data["user2idx"]
# video2idx = data["video2idx"]
# idx2video = data["idx2video"]
# video_title_map = data["video_title_map"]

# @app.route("/recommend", methods=["POST"])
# def recommend():
#     data_input = request.json
#     user_id = data_input.get("user_id")
#     top_k = int(data_input.get("k", 5))

#     if user_id not in user2idx:
#         return jsonify([])

#     user_idx = user2idx[user_id]
#     video_indices = list(video2idx.values())

#     scores = model.predict([
#         np.full(len(video_indices), user_idx),
#         np.array(video_indices),
#         np.zeros(len(video_indices))
#     ], verbose=0).flatten()

#     top_indices = scores.argsort()[-top_k:][::-1]

#     results = [{
#         "video_id": idx2video[video_indices[i]],
#         "title": video_title_map.get(idx2video[video_indices[i]], "Unknown"),
#         "score": float(scores[i])
#     } for i in top_indices]

#     return jsonify(results)

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import pickle

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

# --- TẢI MÔ HÌNH VÀ DỮ LIỆU ---
model = tf.keras.models.load_model("ncf_model_with_title_watchtime")

with open("mappings_with_title_watchtime.pkl", "rb") as f:
    data = pickle.load(f)

user2idx = data["user2idx"]
video2idx = data["video2idx"]
idx2video = data["idx2video"]
video_title_map = data["video_title_map"]

@app.route("/recommend", methods=["GET"])
@app.route("/recommend", methods=["GET"])
def recommend():
    user_id = request.args.get("user_id")
    top_k = int(request.args.get("k", 30))

    if user_id not in user2idx:
        return jsonify([])  # Trả về mảng rỗng nếu user không tồn tại

    user_idx = user2idx[user_id]
    video_indices = list(video2idx.values())

    # Tính toán điểm gợi ý cho mỗi video
    scores = model.predict([
        np.full(len(video_indices), user_idx),
        np.array(video_indices),
        np.zeros(len(video_indices))
    ], verbose=0).flatten()

    # Lấy top k video có điểm cao nhất
    top_indices = scores.argsort()[-top_k:][::-1]

    # Kết quả trả về từ mô hình
    results = [{
        "video_id": idx2video[video_indices[i]],
        "title": video_title_map.get(idx2video[video_indices[i]], "Unknown"),
        "score": float(scores[i])
    } for i in top_indices]

    # In dữ liệu trả về để kiểm tra
    print("Kết quả trả về từ backend:", results)

    return jsonify(results)  # Đảm bảo trả về dữ liệu dạng mảng


if __name__ == "__main__":
    app.run(debug=True)

