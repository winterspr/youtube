import React, { useEffect, useState } from 'react'
import './Recommended2.css'
import { API_KEY } from '../../data'
import { value_converter } from '../../data'
import { Link, useNavigate, useParams} from 'react-router-dom'
const Recommended2 = ({videoId}) => {
    const [apiData, setApiData] = useState([])
    const navigate = useNavigate();
    const fetchData = async()=>{
        const relatedVideo_url = `https://youtube.googleapis.com/youtube/v3/videos?part=snippet%2CcontentDetails%2Cstatistics&id=${videoId}&maxResults=50&key=${API_KEY}`
        await fetch(relatedVideo_url).then(res=>res.json()).then(data=>setApiData(data.items))
    }
    useEffect(()=>{
        fetchData()
    }, [videoId])

    const handleVideoClick = async (videoId) => {
  // Xóa video đã chọn ra khỏi danh sách
        setApiData(prevData => prevData.filter(video => video.id !== videoId));

        // Điều hướng đến trang video cụ thể
        navigate(`/video/${videoId}`);

        // Gọi API để lấy video liên quan
        useEffect(()=>{
            fetchData()
        }, [videoId])
    };

    return (
        <div className="recommended">
            {apiData.map((item, index) => {
                return (
                    <div key={index} className="side-video-list" onClick={() => handleVideoClick(item.id)}>
                        <img src={item.snippet.thumbnails.medium.url} alt="" />
                        <div className="vid-info">
                            <h4>{item.snippet.title}</h4>
                            <p>{item.snippet.channelTitle}</p>
                            <p>{value_converter(item.statistics.viewCount)} views</p>
                        </div>
                    </div>
                );
            })}
        </div>
    );
}

export default Recommended2