import React, { useEffect, useState, useRef } from 'react'
import './PlayVideo.css'
import ThumbUpOutlinedIcon from '@mui/icons-material/ThumbUpOutlined';
import ThumbDownOffAltOutlinedIcon from '@mui/icons-material/ThumbDownOffAltOutlined';
import share from '../../assets/share.png'
import save from '../../assets/save.png'
import { API_KEY, value_converter } from '../../data'
import { formatDistanceToNow } from 'date-fns'
import { useParams, Link } from 'react-router-dom'
import { useDispatch, useSelector } from "react-redux";
import { dislike, fetchSuccess, like } from "../../redux/videoSlice.js";
import ThumbUpIcon from '@mui/icons-material/ThumbUp';
import ThumbDownIcon from '@mui/icons-material/ThumbDown';
import axios from 'axios'

const PlayVideo = () => {
  const { videoId, categoryId } = useParams();
  const { currentUser } = useSelector((state) => state.user);
  const { currentVideo } = useSelector((state) => state.video);
  const dispatch = useDispatch();

  const [apiData, setApiData] = useState(null);
  const [channelData, setChannelData] = useState(null);
  const [commentData, setCommentData] = useState([]);

  // Bắt đầu đếm thời gian xem
  const startTimeRef = useRef(null);
  const hasSavedRef = useRef(false);

  const parseISODuration = (isoDuration) => {
    const regex = /PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?/;
    const matches = isoDuration.match(regex);
    const hours = parseInt(matches[1] || 0, 10);
    const minutes = parseInt(matches[2] || 0, 10);
    const seconds = parseInt(matches[3] || 0, 10);
    return hours * 3600 + minutes * 60 + seconds;
  };
  

useEffect(() => {
  if (apiData && currentUser) {
    startTimeRef.current = Date.now();
    hasSavedRef.current = false;
  }

  return () => {
    if (!hasSavedRef.current && startTimeRef.current && apiData && currentUser) {
      const endTime = Date.now();
      const duration = Math.floor((endTime - startTimeRef.current) / 1000);
      if (duration > 0) {
        saveWatchHistory(duration);
        hasSavedRef.current = true;
      }
    }
  };
}, [apiData, videoId]);

  const saveWatchHistory = async (watchTime) => {
    if (!apiData || !currentUser) return;
    const videoDuration = parseISODuration(apiData.contentDetails?.duration || 'PT0S');
    const actualWatchTime = Math.min(watchTime, videoDuration);
    try {
      const historyData = {
        videoId,
        userId: currentUser._id,
        title: apiData.snippet.title,
        channelTitle: apiData.snippet.channelTitle,
        watchedAt: new Date(),
        watchTime: actualWatchTime,
        thumbnails: {
          default: apiData.snippet.thumbnails.default,
          medium: apiData.snippet.thumbnails.medium,
          high: apiData.snippet.thumbnails.high,
        }
      };
      await axios.post("/api/history/add", historyData);
      console.log("✅ Watch history saved with watchTime:", duration);
    } catch (err) {
      console.error("❌ Error saving watch history:", err.response?.data || err.message);
    }
  };

  const fetchVideoData = async () => {
    const url = `https://youtube.googleapis.com/youtube/v3/videos?part=snippet%2CcontentDetails%2Cstatistics&id=${videoId}&key=${API_KEY}`
    const res = await fetch(url);
    const data = await res.json();
    setApiData(data.items[0]);
  }

  const fetchOtherData = async () => {
    if (!apiData) return;

    const channelRes = await fetch(`https://youtube.googleapis.com/youtube/v3/channels?part=snippet%2Cstatistics&id=${apiData.snippet.channelId}&key=${API_KEY}`);
    const channelData = await channelRes.json();
    setChannelData(channelData.items[0]);

    const commentRes = await fetch(`https://youtube.googleapis.com/youtube/v3/commentThreads?part=snippet%2Creplies&maxResults=50&videoId=${videoId}&key=${API_KEY}`);
    const commentData = await commentRes.json();
    setCommentData(commentData.items);
  }

  useEffect(() => {
    fetchVideoData();
  }, [videoId]);

  useEffect(() => {
    if (apiData && apiData.snippet) {
      fetchOtherData();
    }
  }, [apiData]);

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
      <iframe
        src={`https://www.youtube.com/embed/${videoId}?autoplay=1`}
        frameBorder="0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        referrerPolicy="strict-origin-when-cross-origin"
      ></iframe>

      <h3>{apiData ? apiData.snippet.title : "Title Here"}</h3>

      <div className="play-video-info">
        <p>{apiData ? value_converter(apiData.statistics.viewCount) : "0"} Views &bull; {apiData ? formatDistanceToNow(apiData.snippet.publishedAt, { addSuffix: true }).replace('about', '') : ""}</p>
        <div>
          <span>
            <button onClick={handleLike}>
              {currentVideo?.likes?.includes(currentUser?._id) ? <ThumbUpIcon /> : <ThumbUpOutlinedIcon />}
            </button>
            {apiData ? value_converter(apiData.statistics.likeCount) : 0}
          </span>
          <span>
            <button onClick={handleDislike}>
              {currentVideo?.dislikes?.includes(currentUser?._id) ? <ThumbDownIcon /> : <ThumbDownOffAltOutlinedIcon />}
            </button>
            {apiData ? value_converter(apiData.statistics.dislikeCount) : 0}
          </span>
          <span><img src={share} alt="" />Share</span>
          <span><img src={save} alt="" />Save</span>
        </div>
      </div>

      <hr />

      <div className="publisher">
        <Link to={`/channel/${channelData?.id}`}>
          <img src={channelData?.snippet?.thumbnails?.default?.url || ""} alt="" />
        </Link>
        <div>
          <p>{apiData?.snippet?.channelTitle || ""}</p>
          <span>{value_converter(channelData?.statistics?.subscriberCount || 0)} Subscribers</span>
        </div>
        <button>Subscribe</button>
      </div>

      <div className='vid-dessciption'>
        <p>{apiData ? apiData.snippet.description.slice(0, 250) : "Description Here"}</p>
        <hr />
        <h4>{apiData ? value_converter(apiData.statistics.commentCount) : "0"} Comments</h4>

        {commentData.map((item, index) => (
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
        ))}
      </div>
    </div>
  );
}

export default PlayVideo;
