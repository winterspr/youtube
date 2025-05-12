import User  from '../models/User.js';
import Video from '../models/Video.js';
import { createError } from '../error.js';

//add video
// export const saveVideo =  async (req, res, next) => {
//     const { video } = req.body;
//     if (!video || !video.videoId || !video.title) {
//         return res.status(400).json({ message: 'Dữ liệu video không hợp lệ' });
//     }
//     try {
        
    
//         // Tạo một đối tượng video mới từ dữ liệu frontend gửi lên
//         const save = new Video({
//           videoId: video.id,
//           title: video.snippet.title,
//           description: video.snippet.description,
//           channelTitle: video.snippet.channelTitle,
//           thumbnails: video.snippet.thumbnails,
//           publishedAt: video.snippet.publishedAt,
//           viewCount: video.statistics.viewCount,
//           likeCount: video.statistics.likeCount,
//         });
    
//         // Lưu video vào MongoDB
//         await save.save();
//         res.status(201).json({ message: 'Video đã được lưu thành công' });
//       } catch (error) {
//         res.status(500).json({ message: 'Lỗi khi lưu video', error });
//       }
// }


export const addVideo = async(req, res, next)=>{
    const newVideo = new Video({userId: req.user.id, ...req.body});
    try{
        const savevideo = await newVideo.save();
        res.status(200).json(savevideo);
    }
    catch(err){
        next(err);
    }

}


//update video
export const  updateVideo = async(req, res, next)=>{
    try{
        const  video = await Video.findById(req.params.id);
        if(!video) next(createError(403, "can't find video"));
        if(req.user.id === video.userId){
            const updateVideo = await Video.findByIdAndUpdate(req.params.id,{$set: req.body}, {new: true});
        }
        res.status(200).json(updateVideo);
    } catch(err){
        next(err);
    }
}



//delete video
export const deleteVideo = async(req, res, next)=>{
    try{
        const video = await Video.findById(req.params.id);
        if(!video) next(createError(403, "can't find video"));
        if(req.user.id === video.userId){
            await Video.findByIdAndDelete(req.params.id);
            res.status(200).json("Deleted Video");
        } else {
            next(createError(403, "You can only delete your video"))
        }
        
    } catch(err){
        next(err);
    }
}


//getvideo
export const getVideo = async(req, res, next)=>{
    try{
        const video = await Video.findById(req.params.id);
        if(!video) { return next(createError(403, "can't find video"))};
        return res.status(200).json(video);
    } catch(err){
        next(err);
    }
}


//add view
export const addView = async(req, res, next)=>{
    try{
        await Video.findByIdAndUpdate(req.params.id,{$inc: {views: 1}}, {new: true});
        res.status(200).json("View Added");
    } catch(err){
        next(err);
    }
}


//random
export const random = async(req, res, next)=>{
    try{
        const random = await  Video.aggregate({$sample:  {size: 40}});
        res.status(200).json(random);
    }  catch(err){
        next(err);
    }

}


//trend
export const trend  = async(req, res, next)=>{
    try{
        const trend = await Video.find().sort({view : -1}).limi(4);
        res.status(200).json(trend);
    } catch(err){
        next(err);
    }
}



//sub
export const sub = async(req, res, next)=>{
    try{
        const user = await User.findById(req.user.id);
        const subscribedChannel = user.subscribedUsers;
        const list = await Promise.all(
            subscribedChannel.map((channelId) => {
                return Video.find({userId : channelId});
            }
        ))
        res.status(200).json(list.flat().sort((a,b)=>b.createdAt - a.createdAt));
    } catch(err){
        next(err);
    }
    
}


//getbyTag
export  const getByTag = async(req, res, next)=>{
    const tags = req.query.tags.split(",");
    try {
        const videos = await Video.find({ tags: { $in: tags } }).limit(20);
        res.status(200).json(videos);
    } catch (err) {
        next(err);
    }
}


//search
export const search = async (req, res, next) => {
    const query = req.query.q;
    try {
      const videos = await Video.find({
        title: { $regex: query, $options: "i" },
      }).limit(40);
      res.status(200).json(videos);
    } catch (err) {
      next(err);
    }
};

export const updateWatchTime = async (req, res, next) => {
    const videoId = req.params.id;
    const userId = req.user.id; // Đã xác thực bằng middleware
    const { timeWatched } = req.body; // timeWatched: số giây người dùng vừa xem
  
    if (!timeWatched || typeof timeWatched !== 'number' || timeWatched <= 0) {
      return res.status(400).json({ message: 'Giá trị timeWatched không hợp lệ' });
    }
  
    try {
      const video = await Video.findById(videoId);
      if (!video) {
        return next(createError(404, "Không tìm thấy video"));
      }
  
      // Cập nhật tổng thời gian xem
      video.watchTime = (video.watchTime || 0) + timeWatched;
  
      // Cập nhật thời gian xem theo user
      const previousTime = video.userWatchTimes.get(userId) || 0;
      video.userWatchTimes.set(userId, previousTime + timeWatched);
  
      await video.save();
  
      res.status(200).json({
        message: "Đã cập nhật thời gian xem",
        watchTime: video.watchTime,
        userTime: video.userWatchTimes.get(userId)
      });
    } catch (err) {
      next(err);
    }
};

export const recommendVideo = async (req, res, next) => {
  try {
    // const { userId } = req.params;
    const { id: userId } = req.currentUser;

    const response = await axios.post("http://localhost:5000/recommend", {
      user_id: userId,
      k: 10,
    });
    res.json(response.data);
  } catch (err) {
    next(err);
  }
};