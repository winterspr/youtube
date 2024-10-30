import React, { useEffect, useState } from 'react'
import './PlayVideo.css'
import ThumbUpOutlinedIcon from '@mui/icons-material/ThumbUpOutlined';
import ThumbDownOffAltOutlinedIcon from '@mui/icons-material/ThumbDownOffAltOutlined';
import share from '../../assets/share.png'
import save from '../../assets/save.png'
import { API_KEY, value_converter } from '../../data'
import { formatDistance, formatDistanceToNow } from 'date-fns'
import { useParams } from 'react-router-dom'
import { Link } from 'react-router-dom'
import { useDispatch, useSelector } from "react-redux";
import { dislike, fetchSuccess, like } from "../../redux/videoSlice.js";
import ThumbUpIcon from '@mui/icons-material/ThumbUp';
import ThumbDownIcon from '@mui/icons-material/ThumbDown';
import axios from 'axios'
import { useLocation } from "react-router-dom";

const PlayVideo = () => {
  const {videoId} = useParams()
  const { currentUser } = useSelector((state) => state.user);
  const { currentVideo } = useSelector((state) => state.video);
  //console.log(currentVideo)
  const dispatch = useDispatch();
  const path = useLocation().pathname.split("/")[3];
  console.log(path)
  const [apiData, setApiData] = useState(null);
  const [channelData, setChannelData] = useState(null);
  const [commentData, setCommentData] = useState([]);
  const fetchVideoData = async()=>{
    const videoDetails_url = `https://youtube.googleapis.com/youtube/v3/videos?part=snippet%2CcontentDetails%2Cstatistics&id=${videoId}&key=${API_KEY}`
    await fetch(videoDetails_url).then(res=>res.json()).then(data=>setApiData(data.items[0]))
  }

  const fetchOtherData = async()=>{
    const channelData_url = `https://youtube.googleapis.com/youtube/v3/channels?part=snippet%2CcontentDetails%2Cstatistics&id=${apiData.snippet.channelId}&key=${API_KEY}`
    await fetch(channelData_url).then(res=>res.json()).then(data=>setChannelData(data.items[0]))

    const comment_url = `https://youtube.googleapis.com/youtube/v3/commentThreads?part=snippet%2Creplies&maxResults=50&videoId=${videoId}&key=${API_KEY}`
    await fetch(comment_url).then(res=>res.json()).then(data=>setCommentData(data.items))
  }

  useEffect(()=>{
    fetchVideoData();
  },[videoId])

  useEffect(()=>{
    fetchOtherData();
  },[apiData])

  useEffect(() => {
    const fetchData = async () => {
      try {
        const videoRes = await axios.get(`/api/videos/find/${path}`);
        console.log("Fetched video data:", videoRes.data);
        dispatch(fetchSuccess(videoRes.data));
      } catch (err) {
        console.error("Error fetching video data:", err);
      }
    };
    fetchData();
  }, [path, dispatch]);

  const handleLike = async () => {
    if (currentVideo && currentUser) {
      await axios.put(`/api/users/like/${currentVideo._id}`);
      dispatch(like(currentUser._id));
    }
  };
  const handleDislike = async () => {
    if (currentVideo && currentUser) {
      await axios.put(`/api/users/dislike/${currentVideo._id}`);
      dispatch(dislike(currentUser._id));
    }
  };
  return (
    
    <div className='play-video'>
        {/* <video src={video1} controls autoPlay muted></video> */}
        <iframe src={`https://www.youtube.com/embed/${videoId}?autoplay=1`}  frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin"></iframe>
        <h3>{apiData?apiData.snippet.title: "Title Here"}</h3>
        <div className="play-video-info">
          <p>{apiData?value_converter(apiData.statistics.viewCount):"16k"} Views &bull; {apiData?formatDistanceToNow(apiData.snippet.publishedAt, {addSuffix: true}).replace('about', ''):"0k"}</p>
          <div>
          <span>
  {currentVideo && currentVideo.likes ? (
    <button onClick={handleLike}>
      {currentVideo.likes.includes(currentUser?._id) ? (
        <ThumbUpIcon />
      ) : (
        <ThumbUpOutlinedIcon />
      )}
    </button>
  ) : (
    <button onClick={handleLike}>
      <ThumbUpOutlinedIcon />
    </button>
  )}
  {apiData ? value_converter(apiData.statistics.likeCount) : 155}
</span>

<span>
  {currentVideo && currentVideo.dislikes ? (
    <button onClick={handleDislike}>
      {currentVideo.dislikes.includes(currentUser?._id) ? (
        <ThumbDownIcon />
      ) : (
        <ThumbDownOffAltOutlinedIcon />
      )}
    </button>
  ) : (
    <button onClick={handleDislike}>
      <ThumbDownOffAltOutlinedIcon />
    </button>
  )}
  {apiData ? value_converter(apiData.statistics.dislikeCount) : 155}
</span>
            <span><img src={share} alt="" />Share</span>
            <span><img src={save} alt="" />Save</span>
          </div>
        </div>
        <hr />
        <div className="publisher">
          <Link to={`/channel/${channelData?channelData.id:""}`}>
            <img src={channelData?channelData.snippet.thumbnails.default.url:""} alt="" />
          </Link>
          <div>
            <p>{apiData?apiData.snippet.channelTitle:""}</p>
            <span>{value_converter(channelData?channelData.statistics.subscriberCount:"1M")} Subcribers</span>
          </div>
          <button>subscrible</button>
        </div>
        <div className='vid-dessciption'>
          <p>{apiData?apiData.snippet.description.slice(0,250):"Description Here"}</p>
          <hr />
          <h4>{apiData?value_converter(apiData.statistics.commentCount):102} comments</h4>
          {commentData.map((item,index)=>{
              return(
                <div key={index} className="comment">
                  <img src={item.snippet.topLevelComment.snippet.authorProfileImageUrl} alt="" />
                  <div>
                    <h3>{item.snippet.topLevelComment.snippet.authorDisplayName} <span>1 day ago</span></h3>
                    <p>{item.snippet.topLevelComment.snippet.textDisplay}</p>
                      <div className='comment-action'>
                        <img src={like} alt="" />
                        <span>{value_converter(item.snippet.topLevelComment.snippet.likeCount)}</span>
                        <img src={dislike} alt="" />
                      </div>
                  </div>
                </div>
              )
            })}
        </div>
    </div>
  )
}

export default PlayVideo