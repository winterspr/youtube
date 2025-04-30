# import json
# import re
# import tensorflow as tf
# import numpy as np
# import pandas as pd
# import pickle
# from collections import defaultdict
# from sklearn.metrics import precision_score, recall_score
# from sklearn.model_selection import train_test_split

# # Load JSON

# user_video_pairs = []
# video_title_map = {}

# with open("test.watchhistories.json", "r", encoding="utf-8") as f:
#     raw = json.load(f)

# for entry in raw:
#     user_id = entry.get("userId")
#     video_id = entry.get("videoId")
#     title = entry.get("title", "")

#     if not user_id or not video_id:
#         continue

#     user_video_pairs.append((user_id, video_id))
#     video_title_map[video_id] = title

# # Encode user_id và video_id
# unique_users = sorted(list(set(u for u, _ in user_video_pairs)))
# unique_videos = sorted(list(set(v for _, v in user_video_pairs)))

# user2idx = {u: i for i, u in enumerate(unique_users)}
# video2idx = {v: i for i, v in enumerate(unique_videos)}
# idx2video = {i: v for v, i in video2idx.items()}

# # Chuẩn bị dữ liệu
# df = pd.DataFrame(user_video_pairs, columns=["user_id", "video_id"])
# df["user"] = df["user_id"].map(user2idx)
# df["video"] = df["video_id"].map(video2idx)

# # Xây mô hình NCF
# n_users = len(user2idx)
# n_videos = len(video2idx)

# user_input = tf.keras.Input(shape=(1,))
# video_input = tf.keras.Input(shape=(1,))
# user_emb = tf.keras.layers.Embedding(n_users, 32)(user_input)
# video_emb = tf.keras.layers.Embedding(n_videos, 32)(video_input)

# x = tf.keras.layers.Concatenate()([user_emb, video_emb])
# x = tf.keras.layers.Flatten()(x)
# x = tf.keras.layers.Dense(64, activation="relu")(x)
# x = tf.keras.layers.Dense(1, activation="sigmoid")(x)

# model = tf.keras.Model(inputs=[user_input, video_input], outputs=x)
# model.compile(optimizer="adam", loss="binary_crossentropy")

# X = [df["user"].values, df["video"].values]
# y = np.ones(len(df))  # mọi tương tác đều là tích cực (1)

# model.fit(X, y, epochs=10, batch_size=32, verbose=1)

# # Lưu mô hình và mapping
# model.save("ncf_model")
# with open("mappings.pkl", "wb") as f:
#     pickle.dump((user2idx, video2idx, idx2video, video_title_map), f)


# # user_counts = df["user"].value_counts()
# # df = df[df["user"].isin(user_counts[user_counts >= 2].index)]

# # train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["user"], random_state=42)

# user_counts = df["user"].value_counts()
# df_filtered = df[df["user"].isin(user_counts[user_counts >= 2].index)]

# train_df, test_df = train_test_split(df_filtered, test_size=0.2, stratify=df_filtered["user"], random_state=42)

# ##train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["user"], random_state=42)
# X_train = [train_df["user"].values, train_df["video"].values]
# y_train = np.ones(len(train_df))

# model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)



# # Đánh giá trên test_df
# X_test = [test_df["user"].values, test_df["video"].values]
# y_test = np.ones(len(test_df))

# y_pred = model.predict(X_test)
# y_pred_binary = (y_pred > 0.5).astype(int)

# precision = precision_score(y_test, y_pred_binary)
# recall = recall_score(y_test, y_pred_binary)

# print(f"🎯 Precision: {precision:.4f}")
# print(f"🎯 Recall: {recall:.4f}")
# print(f"🎯 F1 Score: {2 * precision * recall / (precision + recall + 1e-8):.4f}")

# import json
# import tensorflow as tf
# import numpy as np
# import pandas as pd
# import pickle
# import random
# from sklearn.metrics import precision_score, recall_score
# from sklearn.model_selection import train_test_split

# # -----------------------------
# # 1. Load dữ liệu JSON
# # -----------------------------
# user_video_pairs = []
# video_title_map = {}

# with open("test.watchhistories.json", "r", encoding="utf-8") as f:
#     raw = json.load(f)

# for entry in raw:
#     user_id = entry.get("userId")
#     video_id = entry.get("videoId")
#     title = entry.get("title", "")

#     if not user_id or not video_id:
#         continue

#     user_video_pairs.append((user_id, video_id))
#     video_title_map[video_id] = title

# # -----------------------------
# # 2. Encode user_id & video_id
# # -----------------------------
# unique_users = sorted(list(set(u for u, _ in user_video_pairs)))
# unique_videos = sorted(list(set(v for _, v in user_video_pairs)))

# user2idx = {u: i for i, u in enumerate(unique_users)}
# video2idx = {v: i for i, v in enumerate(unique_videos)}
# idx2video = {i: v for v, i in video2idx.items()}

# df = pd.DataFrame(user_video_pairs, columns=["user_id", "video_id"])
# df["user"] = df["user_id"].map(user2idx)
# df["video"] = df["video_id"].map(video2idx)

# # -----------------------------
# # 3. Lọc user có >= 2 video
# # -----------------------------
# user_counts = df["user"].value_counts()
# df_filtered = df[df["user"].isin(user_counts[user_counts >= 2].index)]

# # -----------------------------
# # 4. Negative sampling
# # -----------------------------
# def generate_negative_samples(df, n_neg=3):
#     all_videos = set(df["video"].unique())
#     user_to_positive = df.groupby("user")["video"].apply(set).to_dict()
#     data = []

#     for user, pos_videos in user_to_positive.items():
#         for video in pos_videos:
#             data.append((user, video, 1))  # positive

#             negatives = random.sample(all_videos - pos_videos, min(n_neg, len(all_videos - pos_videos)))
#             for neg_video in negatives:
#                 data.append((user, neg_video, 0))  # negative

#     return pd.DataFrame(data, columns=["user", "video", "label"])

# df_full = generate_negative_samples(df_filtered)

# # -----------------------------
# # 5. Train-test split
# # -----------------------------
# train_df, test_df = train_test_split(df_full, test_size=0.2, stratify=df_full["user"], random_state=42)

# # -----------------------------
# # 6. Xây mô hình NCF
# # -----------------------------
# n_users = len(user2idx)
# n_videos = len(video2idx)

# user_input = tf.keras.Input(shape=(1,))
# video_input = tf.keras.Input(shape=(1,))
# user_emb = tf.keras.layers.Embedding(n_users, 32)(user_input)
# video_emb = tf.keras.layers.Embedding(n_videos, 32)(video_input)

# x = tf.keras.layers.Concatenate()([user_emb, video_emb])
# x = tf.keras.layers.Flatten()(x)
# x = tf.keras.layers.Dense(64, activation="relu")(x)
# x = tf.keras.layers.Dense(1, activation="sigmoid")(x)

# model = tf.keras.Model(inputs=[user_input, video_input], outputs=x)
# model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# # -----------------------------
# # 7. Train mô hình
# # -----------------------------
# X_train = [train_df["user"].values, train_df["video"].values]
# y_train = train_df["label"].values

# model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=1)

# # -----------------------------
# # 8. Đánh giá classification
# # -----------------------------
# X_test = [test_df["user"].values, test_df["video"].values]
# y_test = test_df["label"].values

# y_pred = model.predict(X_test)
# y_pred_binary = (y_pred > 0.5).astype(int)

# precision = precision_score(y_test, y_pred_binary)
# recall = recall_score(y_test, y_pred_binary)
# f1 = 2 * precision * recall / (precision + recall + 1e-8)

# print(f"\n🎯 Binary Precision: {precision:.4f}")
# print(f"🎯 Binary Recall:    {recall:.4f}")
# print(f"🎯 Binary F1 Score:  {f1:.4f}")
# loss, acc = model.evaluate(X_test, y_test, verbose=0)
# print(f"🚨 Test Accuracy: {acc:.4f}")

# # -----------------------------
# # 9. Đánh giá Top-K (recommendation)
# # -----------------------------
# def evaluate_top_k(model, df_test, user2idx, video2idx, k=10):
#     user_pos_test = df_test[df_test["label"] == 1].groupby("user")["video"].apply(set).to_dict()
#     all_videos = list(video2idx.values())
#     precisions, recalls = [], []

#     for user in user_pos_test:
#         pos_videos = user_pos_test[user]

#         user_array = np.full(len(all_videos), user)
#         video_array = np.array(all_videos)

#         predictions = model.predict([user_array, video_array], verbose=0).flatten()
#         top_k_idx = predictions.argsort()[-k:][::-1]
#         top_k_videos = set(video_array[top_k_idx])

#         hits = len(top_k_videos & pos_videos)
#         precisions.append(hits / k)
#         recalls.append(hits / len(pos_videos))

#     return np.mean(precisions), np.mean(recalls)

# precision_k, recall_k = evaluate_top_k(model, test_df, user2idx, video2idx, k=10)
# print(f"\n📌 Precision@10: {precision_k:.4f}")
# print(f"📌 Recall@10:    {recall_k:.4f}")

# # -----------------------------
# # 10. Lưu mô hình & mapping
# # -----------------------------
# model.save("ncf_model")
# with open("mappings.pkl", "wb") as f:
#     pickle.dump((user2idx, video2idx, idx2video, video_title_map), f)

# # === GỢI Ý VIDEO CHO USER ===

# def recommend_videos_for_user(user_id, model, user2idx, video2idx, idx2video, video_title_map, top_k=10):
#     if user_id not in user2idx:
#         print("🛑 User chưa có trong tập huấn luyện.")
#         return []

#     user_idx = user2idx[user_id]
#     all_video_indices = list(video2idx.values())

#     # Dự đoán điểm cho từng video
#     user_input = np.full(len(all_video_indices), user_idx)
#     video_input = np.array(all_video_indices)

#     preds = model.predict([user_input, video_input], verbose=0).flatten()
#     top_indices = preds.argsort()[-top_k:][::-1]

#     recommended_videos = []
#     for idx in top_indices:
#         video_id = idx2video[video_input[idx]]
#         title = video_title_map.get(video_id, "Không rõ tiêu đề")
#         score = preds[idx]
#         recommended_videos.append((video_id, title, score))

#     return recommended_videos


# # === THỬ GỢI Ý VIDEO CHO 1 USER ===
# print("\n📣 Danh sách user trong tập huấn luyện:")
# print(list(user2idx.keys())[:5])  # In thử vài user

# test_user_id = list(user2idx.keys())[0]  # Chọn user đầu tiên để test
# print(f"\n🎯 Gợi ý video cho user: {test_user_id}")

# recommendations = recommend_videos_for_user(test_user_id, model, user2idx, video2idx, idx2video, video_title_map)

# for vid, title, score in recommendations:
#     print(f"🔹 {title} (videoId: {vid}, score: {score:.4f})")



# import json
# import tensorflow as tf
# import numpy as np
# import pandas as pd
# import pickle
# import random
# from sklearn.metrics import precision_score, recall_score
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler

# # -----------------------------
# # 1. Load dữ liệu JSON với Watch Time
# # -----------------------------
# user_video_data = []
# video_title_map = {}

# with open("test.watchhistories2.json", "r", encoding="utf-8") as f:
#     raw = json.load(f)

# for entry in raw:
#     user_id = entry.get("userId")
#     video_id = entry.get("videoId")
#     title = entry.get("title", "")
#     watch_time = entry.get("watchTime", 0)  # Thêm watch_time

#     if not user_id or not video_id:
#         continue

#     user_video_data.append((user_id, video_id, watch_time))
#     video_title_map[video_id] = title

# # -----------------------------
# # 2. Encode user_id & video_id
# # -----------------------------
# unique_users = sorted(list(set(u for u, _, _ in user_video_data)))
# unique_videos = sorted(list(set(v for _, v, _ in user_video_data)))

# user2idx = {u: i for i, u in enumerate(unique_users)}
# video2idx = {v: i for i, v in enumerate(unique_videos)}
# idx2video = {i: v for v, i in video2idx.items()}

# df = pd.DataFrame(user_video_data, columns=["user_id", "video_id", "watch_time"])
# df["user"] = df["user_id"].map(user2idx)
# df["video"] = df["video_id"].map(video2idx)

# # Chuẩn hóa watch_time
# scaler = MinMaxScaler()
# df["watch_time_normalized"] = scaler.fit_transform(df[["watch_time"]])

# # -----------------------------
# # 3. Lọc user có >= 2 video
# # -----------------------------
# user_counts = df["user"].value_counts()
# df_filtered = df[df["user"].isin(user_counts[user_counts >= 2].index)]

# # -----------------------------
# # 4. Negative sampling với watch time
# # -----------------------------
# def generate_negative_samples(df, n_neg=3):
#     all_videos = set(df["video"].unique())
#     user_to_positive = df.groupby("user").apply(
#         lambda x: dict(zip(x["video"], x["watch_time_normalized"]))
#     ).to_dict()
    
#     data = []
    
#     for user, pos_videos in user_to_positive.items():
#         for video, watch_time in pos_videos.items():
#             data.append((user, video, watch_time, 1))  # positive
            
#             negatives = random.sample(all_videos - set(pos_videos.keys()), 
#                                     min(n_neg, len(all_videos - set(pos_videos.keys()))))
#             for neg_video in negatives:
#                 data.append((user, neg_video, 0, 0))  # negative
    
#     return pd.DataFrame(data, columns=["user", "video", "watch_time", "label"])

# df_full = generate_negative_samples(df_filtered)

# # -----------------------------
# # 5. Train-test split
# # -----------------------------
# train_df, test_df = train_test_split(df_full, test_size=0.2, 
#                                     stratify=df_full["user"], random_state=42)

# # -----------------------------
# # 6. Xây mô hình NCF với Watch Time
# # -----------------------------
# n_users = len(user2idx)
# n_videos = len(video2idx)

# user_input = tf.keras.Input(shape=(1,))
# video_input = tf.keras.Input(shape=(1,))
# watch_time_input = tf.keras.Input(shape=(1,))

# # Embedding layers
# user_emb = tf.keras.layers.Embedding(n_users, 32)(user_input)
# video_emb = tf.keras.layers.Embedding(n_videos, 32)(video_input)

# # Concatenate embeddings với watch time
# x = tf.keras.layers.Concatenate()([
#     tf.keras.layers.Flatten()(user_emb),
#     tf.keras.layers.Flatten()(video_emb),
#     watch_time_input
# ])

# # Deeper network
# x = tf.keras.layers.Dense(128, activation="relu")(x)
# x = tf.keras.layers.Dropout(0.2)(x)
# x = tf.keras.layers.Dense(64, activation="relu")(x)
# x = tf.keras.layers.Dropout(0.2)(x)
# x = tf.keras.layers.Dense(32, activation="relu")(x)
# output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

# model = tf.keras.Model(
#     inputs=[user_input, video_input, watch_time_input], 
#     outputs=output
# )

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#     loss="binary_crossentropy",
#     metrics=["accuracy"]
# )

# # -----------------------------
# # 7. Train mô hình
# # -----------------------------
# X_train = [
#     train_df["user"].values,
#     train_df["video"].values,
#     train_df["watch_time"].values
# ]
# y_train = train_df["label"].values

# # Early stopping
# early_stopping = tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss',
#     patience=5,
#     restore_best_weights=True
# )

# model.fit(
#     X_train, 
#     y_train,
#     validation_split=0.2,
#     epochs=200,
#     batch_size=32,
#     callbacks=[early_stopping],
#     verbose=1
# )

# # -----------------------------
# # 8. Đánh giá classification
# # -----------------------------
# X_test = [
#     test_df["user"].values,
#     test_df["video"].values,
#     test_df["watch_time"].values
# ]
# y_test = test_df["label"].values

# y_pred = model.predict(X_test)
# y_pred_binary = (y_pred > 0.5).astype(int)

# precision = precision_score(y_test, y_pred_binary)
# recall = recall_score(y_test, y_pred_binary)
# f1 = 2 * precision * recall / (precision + recall + 1e-8)

# print(f"\n🎯 Binary Precision: {precision:.4f}")
# print(f"🎯 Binary Recall:    {recall:.4f}")
# print(f"🎯 Binary F1 Score:  {f1:.4f}")
# loss, acc = model.evaluate(X_test, y_test, verbose=0)
# print(f"🚨 Test Accuracy: {acc:.4f}")

# # -----------------------------
# # 9. Đánh giá Top-K (recommendation)
# # -----------------------------
# def evaluate_top_k(model, df_test, user2idx, video2idx, k=10):
#     user_pos_test = df_test[df_test["label"] == 1].groupby("user")["video"].apply(set).to_dict()
#     all_videos = list(video2idx.values())
#     precisions, recalls = [], []

#     for user in user_pos_test:
#         pos_videos = user_pos_test[user]
        
#         user_array = np.full(len(all_videos), user)
#         video_array = np.array(all_videos)
#         watch_time_array = np.zeros(len(all_videos))  # Default watch_time = 0 for prediction
        
#         predictions = model.predict(
#             [user_array, video_array, watch_time_array], 
#             verbose=0
#         ).flatten()
        
#         top_k_idx = predictions.argsort()[-k:][::-1]
#         top_k_videos = set(video_array[top_k_idx])
        
#         hits = len(top_k_videos & pos_videos)
#         precisions.append(hits / k)
#         recalls.append(hits / len(pos_videos))
    
#     return np.mean(precisions), np.mean(recalls)

# precision_k, recall_k = evaluate_top_k(model, test_df, user2idx, video2idx, k=10)
# print(f"\n📌 Precision@10: {precision_k:.4f}")
# print(f"📌 Recall@10:    {recall_k:.4f}")

# # -----------------------------
# # 10. Lưu mô hình & mapping
# # -----------------------------
# model.save("ncf_model_with_watchtime")
# with open("mappings_with_watchtime.pkl", "wb") as f:
#     pickle.dump((user2idx, video2idx, idx2video, video_title_map, scaler), f)

# # === GỢI Ý VIDEO CHO USER ===
# def recommend_videos_for_user(user_id, model, user2idx, video2idx, 
#                             idx2video, video_title_map, top_k=10):
#     if user_id not in user2idx:
#         print("🛑 User chưa có trong tập huấn luyện.")
#         return []
    
#     user_idx = user2idx[user_id]
#     all_video_indices = list(video2idx.values())
    
#     # Dự đoán điểm cho từng video
#     user_input = np.full(len(all_video_indices), user_idx)
#     video_input = np.array(all_video_indices)
#     watch_time_input = np.zeros(len(all_video_indices))  # Default watch_time = 0 for prediction
    
#     preds = model.predict(
#         [user_input, video_input, watch_time_input], 
#         verbose=0
#     ).flatten()
    
#     top_indices = preds.argsort()[-top_k:][::-1]
    
#     recommended_videos = []
#     for idx in top_indices:
#         video_id = idx2video[video_input[idx]]
#         title = video_title_map.get(video_id, "Không rõ tiêu đề")
#         score = preds[idx]
#         recommended_videos.append((video_id, title, score))
    
#     return recommended_videos

# # === THỬ GỢI Ý VIDEO CHO 1 USER ===
# print("\n📣 Danh sách user trong tập huấn luyện:")
# print(list(user2idx.keys())[:5])  # In thử vài user

# test_user_id = list(user2idx.keys())[0]  # Chọn user đầu tiên để test
# print(f"\n🎯 Gợi ý video cho user: {test_user_id}")

# recommendations = recommend_videos_for_user(
#     test_user_id, model, user2idx, video2idx, 
#     idx2video, video_title_map
# )

# for vid, title, score in recommendations:
#     print(f"🔹 {title} (videoId: {vid}, score: {score:.4f})")


# import json
# import tensorflow as tf
# import numpy as np
# import pandas as pd
# import pickle
# import random
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler

# # 1. Load watch histories
# print("Loading data...")
# with open("test.watchhistories2.json", "r", encoding="utf-8") as f:
#     histories = json.load(f)

# # 2. Process watch histories
# user_video_data = []
# video_title_map = {}

# for entry in histories:
#     user_id = entry.get("userId")
#     video_id = entry.get("videoId")
#     title = entry.get("title", "")
#     watch_time = entry.get("watchTime", 20)
#     # if "watchTime" not in entry:
#     #     entry["watchTime"] = 20
    
#     if user_id and video_id:
#         user_video_data.append((user_id, video_id, watch_time))
#         if title:
#             video_title_map[video_id] = title

# print(f"Loaded {len(user_video_data)} interactions")

# # 3. Create mappings
# unique_users = sorted(list(set(u for u, _, _ in user_video_data)))
# unique_videos = sorted(list(set(v for _, v, _ in user_video_data)))

# user2idx = {u: i for i, u in enumerate(unique_users)}
# video2idx = {v: i for i, v in enumerate(unique_videos)}
# idx2video = {i: v for v, i in video2idx.items()}

# # 4. Data preprocessing
# df = pd.DataFrame(user_video_data, columns=["user_id", "video_id", "watch_time"])
# df["user"] = df["user_id"].map(user2idx)
# df["video"] = df["video_id"].map(video2idx)

# scaler = MinMaxScaler()
# df["watch_time_normalized"] = scaler.fit_transform(df[["watch_time"]])

# # 5. Generate training data
# min_videos = 2
# df_filtered = df[df.groupby("user")["video"].transform("count") >= min_videos]

# def generate_samples(df, neg_ratio=2):
#     samples = []
#     all_videos = set(df["video"].unique())
    
#     for user, group in df.groupby("user"):
#         pos_videos = set(group["video"])
        
#         # Add positive samples
#         for _, row in group.iterrows():
#             samples.append({
#                 "user": row["user"],
#                 "video": row["video"],
#                 "watch_time": row["watch_time_normalized"],
#                 "label": 1
#             })
        
#         # Add negative samples
#         n_neg = len(pos_videos) * neg_ratio
#         neg_videos = random.sample(list(all_videos - pos_videos), 
#                                 min(n_neg, len(all_videos - pos_videos)))
        
#         for vid in neg_videos:
#             samples.append({
#                 "user": user,
#                 "video": vid,
#                 "watch_time": 0,
#                 "label": 0
#             })
    
#     return pd.DataFrame(samples)

# # Generate training data
# print("Generating training data...")
# train_df = generate_samples(df_filtered)
# train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)

# # 6. Build enhanced NCF model
# print("Building model...")
# n_users = len(user2idx)
# n_videos = len(video2idx)
# emb_dim = 64

# # Model architecture
# user_input = tf.keras.Input(shape=(1,), name='user')
# video_input = tf.keras.Input(shape=(1,), name='video')
# watch_time_input = tf.keras.Input(shape=(1,), name='watch_time')

# # Enhanced embedding layers
# user_embedding = tf.keras.layers.Embedding(
#     n_users, 
#     emb_dim,
#     embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
# )(user_input)
# video_embedding = tf.keras.layers.Embedding(
#     n_videos, 
#     emb_dim,
#     embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
# )(video_input)

# user_vec = tf.keras.layers.Flatten()(user_embedding)
# video_vec = tf.keras.layers.Flatten()(video_embedding)

# # Process watch time
# watch_time_dense = tf.keras.layers.Dense(32, activation='selu')(watch_time_input)
# watch_time_bn = tf.keras.layers.BatchNormalization()(watch_time_dense)

# # Merge features
# concat = tf.keras.layers.Concatenate()([user_vec, video_vec, watch_time_bn])

# # Deep network
# x = tf.keras.layers.Dense(256, activation='selu')(concat)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Dropout(0.3)(x)

# # Residual block
# res = tf.keras.layers.Dense(256, activation='selu')(x)
# res = tf.keras.layers.BatchNormalization()(res)
# res = tf.keras.layers.Dropout(0.3)(res)
# x = tf.keras.layers.Add()([x, res])

# x = tf.keras.layers.Dense(128, activation='selu')(x)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Dropout(0.2)(x)

# output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# # Compile model
# model = tf.keras.Model(
#     inputs=[user_input, video_input, watch_time_input],
#     outputs=output
# )

# # Learning rate schedule
# # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
# #     0.001, decay_steps=1000, decay_rate=0.9
# # )

# optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)


# model.compile(
#     #optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
#     optimizer = optimizer,
#     loss='binary_crossentropy',
#     metrics=['accuracy', tf.keras.metrics.AUC()]
# )

# # Train model
# print("Training model...")
# early_stopping = tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss',
#     patience=3,
#     restore_best_weights=True
# )

# reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
#     monitor='val_loss',
#     factor=0.2,
#     patience=2,
#     min_lr=0.00001
# )

# # Calculate class weights
# n_pos = len(train_df[train_df['label'] == 1])
# n_neg = len(train_df[train_df['label'] == 0])
# class_weights = {0: 1.0, 1: (n_neg/n_pos)}

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

# # Save outputs
# print("\nSaving model and mappings...")
# model.save("/data/chats/54qmdd/workspace/ncf_model_with_watchtime")

# mapping_data = {
#     'user2idx': user2idx,
#     'video2idx': video2idx,
#     'idx2video': idx2video,
#     'video_title_map': video_title_map,
#     'scaler': scaler,
#     'model_performance': history.history
# }

# with open("/data/chats/54qmdd/workspace/mappings_with_watchtime.pkl", "wb") as f:
#     pickle.dump(mapping_data, f)

# # Test recommendations
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
#     true_positive_videos = test_df[
#         (test_df["user"] == user_idx) & (test_df["label"] == 1)
#     ]["video"].unique()
#     return set(true_positive_videos)

# def evaluate_top_k(user_id, k=10):
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

# # Chọn ngẫu nhiên 1 người dùng từ tập test
# test_users = test_df[test_df["label"] == 1]["user"].unique()
# print('user', test_users)
# random_user_idx = random.choice(test_users)

# random_user_id = unique_users[random_user_idx]

# user_idt = "66fe788e1a8cb1d00d69c453"

# # Gọi hàm đánh giá
# evaluate_top_k(user_idt, k=10)


# # Show sample recommendations
# print("\nTesting recommendations...")
# test_user = list(user2idx.keys())[0]
# print(f"\nTop 5 recommendations for user {test_user}:")
# for rec in recommend_videos(test_user):
#     print(f"- {rec['title']} (score: {rec['score']:.3f})")

# --- IMPORTS ---
import json
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# --- LOAD WATCH HISTORIES ---
print("Loading data...")
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

print(f"Loaded {len(user_video_data)} interactions")

# --- MAPPINGS ---
unique_users = sorted(list(set(u for u, _, _ in user_video_data)))
unique_videos = sorted(list(set(v for _, v, _ in user_video_data)))

user2idx = {u: i for i, u in enumerate(unique_users)}
video2idx = {v: i for i, v in enumerate(unique_videos)}
idx2video = {i: v for v, i in video2idx.items()}

# --- DATAFRAME ---
df = pd.DataFrame(user_video_data, columns=["user_id", "video_id", "watch_time"])
df["user"] = df["user_id"].map(user2idx)
df["video"] = df["video_id"].map(video2idx)

scaler = MinMaxScaler()
df["watch_time_normalized"] = scaler.fit_transform(df[["watch_time"]])

# --- FILTER USERS WITH ENOUGH DATA ---
min_videos = 2
df_filtered = df[df.groupby("user")["video"].transform("count") >= min_videos]

# --- NEGATIVE SAMPLING FUNCTION ---
def generate_samples(df, neg_ratio=2):
    samples = []
    all_videos = set(df["video"].unique())

    for user, group in df.groupby("user"):
        pos_videos = set(group["video"])

        for _, row in group.iterrows():
            samples.append({
                "user": row["user"],
                "video": row["video"],
                "watch_time": row["watch_time_normalized"],
                "label": 1
            })

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

# --- GENERATE TRAINING DATA ---
print("Generating training data...")
train_df = generate_samples(df_filtered)
train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)

# --- PROCESS TITLES WITH TF-IDF ---
print("Processing titles with TF-IDF...")
ordered_titles = [video_title_map.get(idx2video[i], "") for i in range(len(video2idx))]
vectorizer = TfidfVectorizer(max_features=100)
title_features = vectorizer.fit_transform(ordered_titles).toarray()
print(f"Title TF-IDF shape: {title_features.shape}")

# --- BUILD MODEL ---
print("Building model...")
n_users = len(user2idx)
n_videos = len(video2idx)
emb_dim = 64

user_input = tf.keras.Input(shape=(1,), name='user')
video_input = tf.keras.Input(shape=(1,), name='video')
watch_time_input = tf.keras.Input(shape=(1,), name='watch_time')

user_embedding = tf.keras.layers.Embedding(n_users, emb_dim)(user_input)
video_embedding = tf.keras.layers.Embedding(n_videos, emb_dim)(video_input)

user_vec = tf.keras.layers.Flatten()(user_embedding)
video_vec = tf.keras.layers.Flatten()(video_embedding)

title_embedding_matrix = tf.constant(title_features, dtype=tf.float32)
title_vec = tf.keras.layers.Lambda(
    lambda x: tf.squeeze(tf.gather(title_embedding_matrix, tf.cast(x, tf.int32)), axis=1)
)(video_input)


watch_time_dense = tf.keras.layers.Dense(32, activation='selu')(watch_time_input)
watch_time_bn = tf.keras.layers.BatchNormalization()(watch_time_dense)

concat = tf.keras.layers.Concatenate()([user_vec, video_vec, title_vec, watch_time_bn])

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

output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=[user_input, video_input, watch_time_input], outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

# --- TRAINING ---
print("Training model...")
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

# --- SAVE MODEL ---
print("Saving model and mappings...")
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

# --- RECOMMENDATION FUNCTIONS ---
def recommend_videos(user_id, k=5):
    if user_id not in user2idx:
        return []

    user_idx = user2idx[user_id]
    video_indices = list(video2idx.values())

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
        return set()
    user_idx = user2idx[user_id]
    true_positive_videos = test_df[(test_df["user"] == user_idx) & (test_df["label"] == 1)]["video"].unique()
    return set(true_positive_videos)

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

# --- TEST SAMPLE ---
print("\nTesting recommendations...")
test_users = test_df[test_df["label"] == 1]["user"].unique()
random_user_idx = random.choice(test_users)
random_user_id = unique_users[random_user_idx]

user_idt = "66fe788e1a8cb1d00d69c453"

# Gọi hàm đánh giá
evaluate_top_k(user_idt, k=20)

test_user = list(user2idx.keys())[0]
print(f"\nTop 5 recommendations for user {test_user}:")
for rec in recommend_videos(test_user):
    print(f"- {rec['title']} (score: {rec['score']:.3f})")


