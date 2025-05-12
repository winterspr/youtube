# --- IMPORTS ---
# import json
# import tensorflow as tf
# import numpy as np
# import pandas as pd
# import pickle
# import random
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.feature_extraction.text import TfidfVectorizer

# # --- LOAD WATCH HISTORIES ---
# print("Loading data...")
# with open("test.watchhistories2.json", "r", encoding="utf-8") as f:
#     histories = json.load(f)

# user_video_data = []
# video_title_map = {}

# for entry in histories:
#     user_id = entry.get("userId")
#     video_id = entry.get("videoId")
#     title = entry.get("title", "")
#     watch_time = entry.get("watchTime", 20)

#     if user_id and video_id:
#         user_video_data.append((user_id, video_id, watch_time))
#         if title:
#             video_title_map[video_id] = title

# print(f"Loaded {len(user_video_data)} interactions")

# # --- MAPPINGS ---
# unique_users = sorted(list(set(u for u, _, _ in user_video_data)))
# unique_videos = sorted(list(set(v for _, v, _ in user_video_data)))

# user2idx = {u: i for i, u in enumerate(unique_users)}
# video2idx = {v: i for i, v in enumerate(unique_videos)}
# idx2video = {i: v for v, i in video2idx.items()}

# # --- DATAFRAME ---
# df = pd.DataFrame(user_video_data, columns=["user_id", "video_id", "watch_time"])
# df["user"] = df["user_id"].map(user2idx)
# df["video"] = df["video_id"].map(video2idx)

# scaler = MinMaxScaler()
# df["watch_time_normalized"] = scaler.fit_transform(df[["watch_time"]])

# # --- FILTER USERS WITH ENOUGH DATA ---
# min_videos = 2
# df_filtered = df[df.groupby("user")["video"].transform("count") >= min_videos]

# # --- NEGATIVE SAMPLING FUNCTION ---
# def generate_samples(df, neg_ratio=2):
#     samples = []
#     all_videos = set(df["video"].unique())

#     for user, group in df.groupby("user"):
#         pos_videos = set(group["video"])

#         for _, row in group.iterrows():
#             samples.append({
#                 "user": row["user"],
#                 "video": row["video"],
#                 "watch_time": row["watch_time_normalized"],
#                 "label": 1
#             })

#         n_neg = len(pos_videos) * neg_ratio
#         neg_videos = random.sample(list(all_videos - pos_videos),
#                                    min(n_neg, len(all_videos - pos_videos)))

#         for vid in neg_videos:
#             samples.append({
#                 "user": user,
#                 "video": vid,
#                 "watch_time": 0,
#                 "label": 0
#             })

#     return pd.DataFrame(samples)

# # --- GENERATE TRAINING DATA ---
# print("Generating training data...")
# train_df = generate_samples(df_filtered)
# train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)

# # --- PROCESS TITLES WITH TF-IDF ---
# print("Processing titles with TF-IDF...")
# ordered_titles = [video_title_map.get(idx2video[i], "") for i in range(len(video2idx))]
# vectorizer = TfidfVectorizer(max_features=100)
# title_features = vectorizer.fit_transform(ordered_titles).toarray()
# print(f"Title TF-IDF shape: {title_features.shape}")

# # --- BUILD MODEL ---
# print("Building model...")
# n_users = len(user2idx)
# n_videos = len(video2idx)
# emb_dim = 64

# user_input = tf.keras.Input(shape=(1,), name='user')
# video_input = tf.keras.Input(shape=(1,), name='video')
# watch_time_input = tf.keras.Input(shape=(1,), name='watch_time')

# user_embedding = tf.keras.layers.Embedding(n_users, emb_dim)(user_input)
# video_embedding = tf.keras.layers.Embedding(n_videos, emb_dim)(video_input)

# user_vec = tf.keras.layers.Flatten()(user_embedding)
# video_vec = tf.keras.layers.Flatten()(video_embedding)

# title_embedding_matrix = tf.constant(title_features, dtype=tf.float32)
# title_vec = tf.keras.layers.Lambda(
#     lambda x: tf.squeeze(tf.gather(title_embedding_matrix, tf.cast(x, tf.int32)), axis=1)
# )(video_input)


# watch_time_dense = tf.keras.layers.Dense(32, activation='selu')(watch_time_input)
# watch_time_bn = tf.keras.layers.BatchNormalization()(watch_time_dense)

# concat = tf.keras.layers.Concatenate()([user_vec, video_vec, title_vec, watch_time_bn])

# x = tf.keras.layers.Dense(256, activation='selu')(concat)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Dropout(0.3)(x)

# res = tf.keras.layers.Dense(256, activation='selu')(x)
# res = tf.keras.layers.BatchNormalization()(res)
# res = tf.keras.layers.Dropout(0.3)(res)
# x = tf.keras.layers.Add()([x, res])

# x = tf.keras.layers.Dense(128, activation='selu')(x)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Dropout(0.2)(x)

# output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# model = tf.keras.Model(inputs=[user_input, video_input, watch_time_input], outputs=output)

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#     loss='binary_crossentropy',
#     metrics=['accuracy', tf.keras.metrics.AUC()]
# )

# # --- TRAINING ---
# print("Training model...")
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
# reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-5)

# n_pos = len(train_df[train_df['label'] == 1])
# n_neg = len(train_df[train_df['label'] == 0])
# class_weights = {0: 1.0, 1: (n_neg / n_pos)}

# history = model.fit(
#     [train_df["user"], train_df["video"], train_df["watch_time"]],
#     train_df["label"],
#     validation_split=0.2,
#     epochs=20,
#     batch_size=512,
#     callbacks=[early_stopping, reduce_lr],
#     class_weight=class_weights,
#     verbose=1
# )

# # --- SAVE MODEL ---
# print("Saving model and mappings...")
# model.save("ncf_model_with_title_watchtime")

# with open("mappings_with_title_watchtime.pkl", "wb") as f:
#     pickle.dump({
#         'user2idx': user2idx,
#         'video2idx': video2idx,
#         'idx2video': idx2video,
#         'video_title_map': video_title_map,
#         'scaler': scaler,
#         'vectorizer': vectorizer,
#         'model_performance': history.history
#     }, f)

# # --- RECOMMENDATION FUNCTIONS ---
# def recommend_videos(user_id, k=5):
#     if user_id not in user2idx:
#         return []

#     user_idx = user2idx[user_id]
#     video_indices = list(video2idx.values())

#     scores = model.predict([
#         np.full(len(video_indices), user_idx),
#         np.array(video_indices),
#         np.zeros(len(video_indices))
#     ], verbose=0).flatten()

#     top_k = scores.argsort()[-k:][::-1]
#     return [{
#         'video_id': idx2video[video_indices[i]],
#         'title': video_title_map.get(idx2video[video_indices[i]], "Unknown"),
#         'score': float(scores[i])
#     } for i in top_k]

# def get_true_positives(user_id):
#     if user_id not in user2idx:
#         return set()
#     user_idx = user2idx[user_id]
#     true_positive_videos = test_df[(test_df["user"] == user_idx) & (test_df["label"] == 1)]["video"].unique()
#     return set(true_positive_videos)

# def evaluate_top_k(user_id, k=20):
#     true_videos = get_true_positives(user_id)
#     if not true_videos:
#         print("Người dùng không có dữ liệu trong tập kiểm tra.")
#         return

#     recommendations = recommend_videos(user_id, k=k)
#     recommended_ids = [video2idx[rec['video_id']] for rec in recommendations]

#     hits = [vid for vid in recommended_ids if vid in true_videos]

#     precision = len(hits) / k
#     recall = len(hits) / len(true_videos)

#     print(f"\n🎯 Đánh giá gợi ý cho người dùng {user_id}:")
#     print(f"- Số lượng video đúng trong test set: {len(true_videos)}")
#     print(f"- Số lượng đúng trong top-{k} gợi ý: {len(hits)}")
#     print(f"- ✅ Precision@{k}: {precision:.2f}")
#     print(f"- ✅ Recall@{k}: {recall:.2f}")

#     print("\n📺 Video gợi ý:")
#     for rec in recommendations:
#         print(f"- {rec['title']} (score: {rec['score']:.3f})")

# # --- TEST SAMPLE ---
# print("\nTesting recommendations...")
# test_users = test_df[test_df["label"] == 1]["user"].unique()
# random_user_idx = random.choice(test_users)
# random_user_id = unique_users[random_user_idx]

# user_idt = "66fe788e1a8cb1d00d69c453"

# # Gọi hàm đánh giá
# evaluate_top_k(user_idt, k=20)

# test_user = list(user2idx.keys())[0]
# print(f"\nTop 5 recommendations for user {test_user}:")
# for rec in recommend_videos(test_user):
#     print(f"- {rec['title']} (score: {rec['score']:.3f})")


import json
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# --- TẢI DỮ LIỆU LỊCH SỬ XEM VIDEO ---
print("Đang tải dữ liệu...")
with open("test.watchhistories2.json", "r", encoding="utf-8") as f:
    histories = json.load(f)

user_video_data = []
video_title_map = {}

for entry in histories:
    user_id = entry.get("userId")
    video_id = entry.get("videoId")
    title = entry.get("title", "")
    watch_time = entry.get("watchTime", 20)

    if user_id and video_id:
        user_video_data.append((user_id, video_id, watch_time))
        if title:
            video_title_map[video_id] = title

print(f"Đã tải {len(user_video_data)} lượt xem video")

# --- TẠO CÁC MAPPING ---
unique_users = sorted(list(set(u for u, _, _ in user_video_data)))
unique_videos = sorted(list(set(v for _, v, _ in user_video_data)))

user2idx = {u: i for i, u in enumerate(unique_users)}
video2idx = {v: i for i, v in enumerate(unique_videos)}
idx2video = {i: v for v, i in video2idx.items()}

# --- CHUYỂN ĐỔI DỮ LIỆU THÀNH DATAFRAME ---
df = pd.DataFrame(user_video_data, columns=["user_id", "video_id", "watch_time"])
df["user"] = df["user_id"].map(user2idx)
df["video"] = df["video_id"].map(video2idx)

scaler = MinMaxScaler()
df["watch_time_normalized"] = scaler.fit_transform(df[["watch_time"]])

# --- LỌC NGƯỜI DÙNG CÓ ĐỦ DỮ LIỆU ---
min_videos = 2
df_filtered = df[df.groupby("user")["video"].transform("count") >= min_videos]

# --- HÀM SINH DỮ LIỆU CHO HỌC ---
def generate_samples(df, neg_ratio=2):
    samples = []
    all_videos = set(df["video"].unique())

    for user, group in df.groupby("user"):
        pos_videos = set(group["video"])

        # Lấy các video đã xem
        for _, row in group.iterrows():
            samples.append({
                "user": row["user"],
                "video": row["video"],
                "watch_time": row["watch_time_normalized"],
                "label": 1
            })

        # Lấy các video âm tính (chưa xem)
        n_neg = len(pos_videos) * neg_ratio
        neg_videos = random.sample(list(all_videos - pos_videos),
                                   min(n_neg, len(all_videos - pos_videos)))

        for vid in neg_videos:
            samples.append({
                "user": user,
                "video": vid,
                "watch_time": 0,
                "label": 0
            })

    return pd.DataFrame(samples)

# --- TẠO DỮ LIỆU ĐÀO TẠO ---
print("Đang tạo dữ liệu đào tạo...")
train_df = generate_samples(df_filtered)
train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)

# --- XỬ LÝ TIÊU ĐỀ VIDEO VỚI TF-IDF ---
print("Đang xử lý tiêu đề với TF-IDF...")
ordered_titles = [video_title_map.get(idx2video[i], "") for i in range(len(video2idx))]
vectorizer = TfidfVectorizer(max_features=100)
title_features = vectorizer.fit_transform(ordered_titles).toarray()
print(f"Kích thước ma trận tiêu đề TF-IDF: {title_features.shape}")

# --- XÂY DỰNG MÔ HÌNH ---
print("Đang xây dựng mô hình...")
n_users = len(user2idx)
n_videos = len(video2idx)
emb_dim = 64

user_input = tf.keras.Input(shape=(1,), name='user')
video_input = tf.keras.Input(shape=(1,), name='video')
watch_time_input = tf.keras.Input(shape=(1,), name='watch_time')

# Lớp embedding cho người dùng và video
user_embedding = tf.keras.layers.Embedding(n_users, emb_dim)(user_input)
video_embedding = tf.keras.layers.Embedding(n_videos, emb_dim)(video_input)

user_vec = tf.keras.layers.Flatten()(user_embedding)
video_vec = tf.keras.layers.Flatten()(video_embedding)

# Xử lý tiêu đề video sử dụng ma trận embedding từ TF-IDF
# title_embedding_matrix = tf.constant(title_features, dtype=tf.float32)
# print("title_embedding_matrix shape:", title_embedding_matrix.shape)

# title_vec = tf.keras.layers.Lambda(
#     lambda x: tf.squeeze(tf.gather(title_embedding_matrix, tf.cast(x, tf.int32)), axis=1)
# )(video_input)
title_embedding = tf.keras.layers.Embedding(
    input_dim=title_features.shape[0],
    output_dim=title_features.shape[1],
    weights=[title_features],
    trainable=False
)

title_vec = title_embedding(video_input)
title_vec = tf.keras.layers.Flatten()(title_vec)


# Lớp dense cho thời gian xem
watch_time_dense = tf.keras.layers.Dense(32, activation='selu')(watch_time_input)
watch_time_bn = tf.keras.layers.BatchNormalization()(watch_time_dense)

# Kết hợp các đầu vào
concat = tf.keras.layers.Concatenate()([user_vec, video_vec, title_vec, watch_time_bn])

# Các lớp dense
x = tf.keras.layers.Dense(256, activation='selu')(concat)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)

res = tf.keras.layers.Dense(256, activation='selu')(x)
res = tf.keras.layers.BatchNormalization()(res)
res = tf.keras.layers.Dropout(0.3)(res)
x = tf.keras.layers.Add()([x, res])

x = tf.keras.layers.Dense(128, activation='selu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.2)(x)

# Lớp đầu ra
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# Xây dựng mô hình
model = tf.keras.Model(inputs=[user_input, video_input, watch_time_input], outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

# --- ĐÀO TẠO MÔ HÌNH ---
print("Đang đào tạo mô hình...")
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-5)

n_pos = len(train_df[train_df['label'] == 1])
n_neg = len(train_df[train_df['label'] == 0])
class_weights = {0: 1.0, 1: (n_neg / n_pos)}

history = model.fit(
    [train_df["user"], train_df["video"], train_df["watch_time"]],
    train_df["label"],
    validation_split=0.2,
    epochs=20,
    batch_size=512,
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weights,
    verbose=1
)

# --- LƯU MÔ HÌNH ---
print("Đang lưu mô hình và các ánh xạ...")
model.save("ncf_model_with_title_watchtime")

with open("mappings_with_title_watchtime.pkl", "wb") as f:
    pickle.dump({
        'user2idx': user2idx,
        'video2idx': video2idx,
        'idx2video': idx2video,
        'video_title_map': video_title_map,
        'scaler': scaler,
        'vectorizer': vectorizer,
        'model_performance': history.history
    }, f)

# --- HÀM GỢI Ý VIDEO ---
def recommend_videos(user_id, k=5):
    if user_id not in user2idx:
        return []

    user_idx = user2idx[user_id]
    video_indices = list(video2idx.values())

    # Dự đoán điểm số cho từng video
    scores = model.predict([
        np.full(len(video_indices), user_idx),
        np.array(video_indices),
        np.zeros(len(video_indices))
    ], verbose=0).flatten()

    top_k = scores.argsort()[-k:][::-1]
    return [{
        'video_id': idx2video[video_indices[i]],
        'title': video_title_map.get(idx2video[video_indices[i]], "Unknown"),
        'score': float(scores[i])
    } for i in top_k]


def get_true_positives(user_id):
    if user_id not in user2idx:
        return set()  # Nếu không tìm thấy user trong user2idx, trả về tập rỗng
    user_idx = user2idx[user_id]
    
    # Lọc các video trong test_df mà user đã xem (label = 1) và trả về set của các video đó
    true_positive_videos = test_df[(test_df["user"] == user_idx) & (test_df["label"] == 1)]["video"].unique()
    
    return set(true_positive_videos)


# --- ĐÁNH GIÁ CÁC GỢI Ý ---
def evaluate_top_k(user_id, k=20):
    true_videos = get_true_positives(user_id)
    if not true_videos:
        print("Người dùng không có dữ liệu trong tập kiểm tra.")
        return

    recommendations = recommend_videos(user_id, k=k)
    recommended_ids = [video2idx[rec['video_id']] for rec in recommendations]

    hits = [vid for vid in recommended_ids if vid in true_videos]

    precision = len(hits) / k
    recall = len(hits) / len(true_videos)

    print(f"\n🎯 Đánh giá gợi ý cho người dùng {user_id}:")
    print(f"- Số lượng video đúng trong test set: {len(true_videos)}")
    print(f"- Số lượng đúng trong top-{k} gợi ý: {len(hits)}")
    print(f"- ✅ Precision@{k}: {precision:.2f}")
    print(f"- ✅ Recall@{k}: {recall:.2f}")

    print("\n📺 Video gợi ý:")
    for rec in recommendations:
        print(f"- {rec['title']} (score: {rec['score']:.3f})")

# --- TEST VỚI NGƯỜI DÙNG ---
print("\nĐang kiểm tra các gợi ý...")
test_users = test_df[test_df["label"] == 1]["user"].unique()
random_user_idx = random.choice(test_users)
random_user_id = unique_users[random_user_idx]

user_idt = "66fe788e1a8cb1d00d69c453"  # ID người dùng mẫu

# Gọi hàm đánh giá
evaluate_top_k(user_idt, k=20)

test_user = list(user2idx.keys())[0]
print(f"\nTop 5 gợi ý cho người dùng {test_user}:")
for rec in recommend_videos(test_user):
    print(f"- {rec['title']} (score: {rec['score']:.3f})")
