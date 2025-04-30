import mongoose, { Schema } from "mongoose";

const WatchhistorySchema = new mongoose.Schema({
    videoId: {type: String, required: true},
    userId: {type: String, required: true},
    watchedAt: {type: Date, default: Date.now},
    title: {type: String, required: true},
    channelTitle: {type: String, required: true},
    thumbnails:{
        default: {url: String, width: Number, height: Number},
        medium: {url: String, width: Number, height: Number},
        high: {url: String, width: Number, height: Number},
    },
    watchTime: { type: Number, default: 0 }
})

export default  mongoose.model('WatchHistory', WatchhistorySchema);
