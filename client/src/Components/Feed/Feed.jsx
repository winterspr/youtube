import React, { useState, useEffect } from 'react'
import './Feed.css'
import {Link} from 'react-router-dom'
import { API_KEY } from '../../data'
import { value_converter } from '../../data'
import { formatDistanceToNow } from 'date-fns'
import { useSelector } from 'react-redux'
import axios from 'axios'
const Feed = ({category}) => {
    const [data, setData] = useState([]);
    const { currentUser } = useSelector((state) => state.user);
//     const fetchData = async()=>{
//         const videoList_url = `https://youtube.googleapis.com/youtube/v3/videos?part=snippet%2CcontentDetails%2Cstatistics&chart=mostPopular&maxResults=50&regionCode=VN&videoCategoryId=${category}&key=${API_KEY}`
//         await fetch(videoList_url).then(response=>response.json()).then(data=>setData(data.items))
//     }
    const fetchData = async () => {
      try {
        const videoList_url = `https://youtube.googleapis.com/youtube/v3/videos?part=snippet,contentDetails,statistics&chart=mostPopular&maxResults=50&regionCode=VN&videoCategoryId=${category}&key=${API_KEY}`;
        const res = await axios.get(videoList_url);
        setData(res.data.items);
      } catch (err) {
        console.error('Fetch error:', err.message);
      }
    };

    const parseISODuration = (isoDuration) => {
        const regex = /PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?/;
        const matches = isoDuration.match(regex);
      
        const hours = parseInt(matches[1] || 0, 10);
        const minutes = parseInt(matches[2] || 0, 10);
        const seconds = parseInt(matches[3] || 0, 10);
      
        return hours * 3600 + minutes * 60 + seconds;
      };
      
    useEffect(()=>{
        fetchData();
    },[category])
    const saveVideoToDB = async (item) => {
        try {
            const videoData = {
            _id: item.id,
            title: item.snippet.title,
            description: item.snippet.description || '', 
            channelTitle: item.snippet.channelTitle,
            thumbnails: {
                default: {
                    url: item.snippet.thumbnails.default?.url || '', 
                    width: item.snippet.thumbnails.default?.width || 120,
                    height: item.snippet.thumbnails.default?.height || 90
                },
                medium: {
                    url: item.snippet.thumbnails.medium?.url || '',
                    width: item.snippet.thumbnails.medium?.width || 320,
                    height: item.snippet.thumbnails.medium?.height || 180
                },
                high: {
                    url: item.snippet.thumbnails.high?.url || '',
                    width: item.snippet.thumbnails.high?.width || 480,
                    height: item.snippet.thumbnails.high?.height || 360
                }
            },
            publishedAt: item.snippet.publishedAt,
            viewCount: parseInt(item.statistics.viewCount, 10) || 0, 
            likeCount: parseInt(item.statistics.likeCount, 10) || 0,
            duration: parseISODuration(item.contentDetails?.duration || "PT0S")
            };
            const response = await axios.post('/api/videos/add', videoData);
            if (response.status === 200) {
                alert('Saved Video Successful!');
            }
        } catch (error) {
            if (error.response && error.response.status === 409) {
                alert("This video Saved.");
            } 
            else if(item._id){
                console.log('Saved');
            }
            else {
                console.error('Error:', error.response ? error.response.data : error.message);
            }
        }
    }

    // const saveWatchhistory = async(item)=>{
    //     try{
    //         const historyData = {
    //             videoId: item.id,
    //             userId: currentUser._id,
    //             title: item.snippet.title,
    //             channelTitle: item.snippet.channelTitle,
    //             publishedAt: item.snippet.publishedAt,
    //             thumbnails:{
    //                 default: {
    //                     url: item.snippet.thumbnails.default?.url || '',
    //                     width: item.snippet.thumbnails.default?.width || 120,
    //                     height: item.snippet.thumbnails.default?.height || 90
    //                 },
    //                 medium: {
    //                     url: item.snippet.thumbnails.medium?.url || '',
    //                     width: item.snippet.thumbnails.medium?.width || 320,
    //                     height: item.snippet.thumbnails.medium?.height || 180
    //                 },
    //                 high: {
    //                     url: item.snippet.thumbnails.high?.url || '',
    //                     width: item.snippet.thumbnails.high?.width || 480,
    //                     height: item.snippet.thumbnails.high?.height || 360
    //                 }
    //             }
    //         };
    //         const response = await axios.post('/api/history/add', historyData);
    //         if(response.status == 201){
    //             console.log('Saved watch history successfully');
    //         }
    //     } catch(err){
    //         console.error('error', err.response ? err.response.data : err.message);
    //     }
    // }
  return (
    <div className='feed'>
        {data.map((item, index)=>{
            return (
            <Link to={`video/${item.snippet.categoryId}/${item.id}`} className="card" onClick={async () => {
                try {
                    await Promise.all([
                        saveVideoToDB(item),
                        //saveWatchhistory(item)
                    ]);
                } catch (err) {
                    console.error('error', err.message);
                }
            }}>
                <img src={item.snippet.thumbnails.medium.url} alt="" />
                <h2>{item.snippet.title}</h2>
                <h3>{item.snippet.channelTitle}</h3>
                <p>{value_converter(item.statistics.viewCount)} views &bull; {formatDistanceToNow(item.snippet.publishedAt, {addSuffix: true}).replace('about', '')}</p>
            </Link>
            )
        })}
    </div>  
  )
}
export default Feed