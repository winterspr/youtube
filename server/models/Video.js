import mongoose from "mongoose";

const videoSchema = new mongoose.Schema({ 
  _id: { type: String, required: true, unique: true},// ID của video từ YouTube, để tránh trùng lặp
  title: { type: String, required: true },
  description: { type: String, default: '' },
  channelTitle: { type: String, required: true },
  thumbnails: {
    default: {
      url: { type: String, default: '' },
      width: { type: Number, default: 120 },
      height: { type: Number, default: 90 }
    },
    medium: {
      url: { type: String, default: '' },
      width: { type: Number, default: 320 },
      height: { type: Number, default: 180 }
    },
    high: {
      url: { type: String, default: '' },
      width: { type: Number, default: 480 },
      height: { type: Number, default: 360 }
    }
  },
  publishedAt: { type: Date, required: true },
  viewCount: { type: Number, default: 0 },
  likeCount: { type: Number, default: 0 },
  likes: {
    type: [String],
    default: [],
  },
  dislikes: {
    type: [String],
    default: [],
  },
  userId: {
    type: String,
    required: true,
  },
},
{ timestamps: true }
);

export default mongoose.model('Video', videoSchema);