import WatchHistory from '../models/History.js';
import User from '../models/User.js';
import Video from '../models/Video.js';

export const saveHistory = async(req, res, next) => {
    try{
        const {videoId, userId, title, thumbnails, watchedAt, channelTitle, watchTime} = req.body;
        const newHistory = new WatchHistory({
            videoId,
            userId,
            title,
            channelTitle,
            thumbnails,
            watchedAt: watchedAt || new Date(),
            watchTime: watchTime || 0,
        });
        await newHistory.save();
        res.status(201).json({message: 'History saved successfully'});
    } catch(err){
        console.error('lỗi khi lưu lịch sử', err);
        next(err);
    }
}
