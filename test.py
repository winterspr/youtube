import json
import re
import tensorflow as tf
import numpy as np
from collections import defaultdict
import pandas as pd
import pickle

# Bước 1: Load dữ liệu JSON
with open("test.watchhistories.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

user_counter = 0
user_sessions = defaultdict(list)
video_title_map = {}
video_id_set = set()

for entry in raw:
    title = entry.get("title", "")
    url = entry.get("titleUrl", "")
    match = re.search(r"v=([\w-]+)", url)
    if match:
        video_id = match.group(1)
        session_id = f"user_{user_counter // 20}" 
        user_sessions[session_id].append(video_id)
        video_title_map[video_id] = title
        video_id_set.add(video_id)
        user_counter += 1

# Bước 2: Encode user_id & video_id thành số
user2idx = {user: idx for idx, user in enumerate(user_sessions)}
video2idx = {vid: idx for idx, vid in enumerate(video_id_set)}
idx2video = {v: k for k, v in video2idx.items()}

# Bước 3: Tạo tập dữ liệu huấn luyện
train_data = []
for user, videos in user_sessions.items():
    for vid in videos:
        train_data.append((user2idx[user], video2idx[vid]))

df = pd.DataFrame(train_data, columns=["user", "video"])

# Bước 4: Xây dựng mô hình NCF
n_users = len(user2idx)
n_videos = len(video2idx)

user_input = tf.keras.Input(shape=(1,))
video_input = tf.keras.Input(shape=(1,))

user_embedding = tf.keras.layers.Embedding(n_users, 32)(user_input)
video_embedding = tf.keras.layers.Embedding(n_videos, 32)(video_input)

concat = tf.keras.layers.Concatenate()([user_embedding, video_embedding])
x = tf.keras.layers.Flatten()(concat)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=[user_input, video_input], outputs=x)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Bước 5: Huấn luyện mô hình
X = [df["user"].values, df["video"].values]
y = np.ones(len(df))  # Tất cả tương tác đều là positive

model.fit(X, y, epochs=10, batch_size=32, verbose=1)

# Lưu mô hình & mapping
model.save("ncf_model")
with open("mappings.pkl", "wb") as f:
    pickle.dump((user2idx, video2idx, idx2video, video_title_map), f)
