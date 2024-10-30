import React, { useState, useEffect } from 'react'
import './Feed.css'
import {Link} from 'react-router-dom'
import { API_KEY } from '../../data'
import { value_converter } from '../../data'
import { formatDistanceToNow } from 'date-fns'
import axios from 'axios'
const Feed = ({category}) => {
    const [data, setData] = useState([]);
    const fetchData = async()=>{
        const videoList_url = `https://youtube.googleapis.com/youtube/v3/videos?part=snippet%2CcontentDetails%2Cstatistics&chart=mostPopular&maxResults=50&regionCode=VN&videoCategoryId=${category}&key=${API_KEY}`
        await fetch(videoList_url).then(response=>response.json()).then(data=>setData(data.items))
    }

    useEffect(()=>{
        fetchData();
    },[category])
    const saveVideoToDB = async (item) => {
        try {
            const videoData = {
            _id: item.id,// Đặt _id là videoId từ YouTube
            title: item.snippet.title,
            description: item.snippet.description || '',  // Đặt mô tả mặc định nếu không có
            channelTitle: item.snippet.channelTitle,
            thumbnails: {
                default: {
                    url: item.snippet.thumbnails.default?.url || '',  // Kiểm tra null và đặt mặc định
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
            viewCount: parseInt(item.statistics.viewCount, 10) || 0,  // Đặt giá trị mặc định là 0
            likeCount: parseInt(item.statistics.likeCount, 10) || 0   // Đặt giá trị mặc định là 0 nếu không có
            };
            const response = await axios.post('/api/videos/add', videoData);
            // if (response.status === 200) {
            //     alert('Video đã được lưu vào database!');
            // }
        } catch (error) {
            if (error.response && error.response.status === 409) {  // Lỗi 409: Conflict
                alert("Video này đã tồn tại trong database.");
            } else {
                console.error('Lỗi khi lưu video:', error.response ? error.response.data : error.message);
            }
        }
    }
  return (
    <div className='feed'>
        {data.map((item, index)=>{
            return (
            <Link to={`video/${item.snippet.categoryId}/${item.id}`} className="card" onClick={()=>saveVideoToDB(item)}>
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