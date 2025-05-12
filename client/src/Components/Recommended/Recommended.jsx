import React, { useEffect, useState, useRef } from 'react'
import './Recommended.css'
import { API_KEY } from '../../data'
import { value_converter } from '../../data'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { useSelector } from 'react-redux'
import axios from 'axios'
const Recommended = ({categoryId}) => {
    const [apiData, setApiData] = useState([])
    const { videoId } = useParams();
    const navigate = useNavigate();
    const { currentUser } = useSelector((state) => state.user);
    const fetchData = async()=>{
        const relatedVideo_url = `https://youtube.googleapis.com/youtube/v3/videos?part=snippet%2CcontentDetails%2Cstatistics&chart=mostPopular&regionCode=VN&maxResults=45&videoCategoryId=${categoryId}&key=${API_KEY}`
        await fetch(relatedVideo_url).then(res=>res.json()).then(data=>setApiData(data.items))
    }
    useEffect(()=>{
        fetchData()
    }, [categoryId])

  //  const fetchData = async () => {
  //   try {
  //     const response = await axios.get('/recommend', {
  //       params: { user_id: currentUser._id, k: 30 }  // Gọi API với user_id và k = 5 video gợi ý
  //     });

  //     // Kiểm tra dữ liệu trả về có phải là mảng không
  //     if (Array.isArray(response.data)) {
  //       setApiData(response.data);  // Lưu mảng kết quả vào state
  //     } else {
  //       console.error('Dữ liệu trả về không phải là mảng:', response.data);
  //       alert('Lỗi: Dữ liệu không phải là mảng.');
  //     }
  //   } catch (error) {
  //     console.error('Lỗi khi gọi API gợi ý:', error.response ? error.response.data : error.message);
  //   }
  // };

  // useEffect(() => {
  //   if (currentUser) {
  //     fetchData();
  //   }
  // }, [currentUser]);
        
    const handleVideoClick = (videoId) => {
        setApiData(prevData => prevData.filter(video => video.id !== videoId));
        navigate(`/video/${categoryId}/${videoId}`);
    };
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
            likeCount: parseInt(item.statistics.likeCount, 10) || 0  
            };
            const response = await axios.post('/api/videos/add', videoData);
            if (response.status === 200) {
                alert('Video đã được lưu vào database!');
            }
        } catch (error) {
            if (error.response && error.response.status === 409) {
                alert("Video này đã tồn tại trong database.");
            } else {
                console.error('Lỗi khi lưu video:', error.response ? error.response.data : error.message);
            }
        }
    }

    return (
        <div className="recommended">
            {apiData.map((item, index) => {
                return (
                    <div key={index} className="side-video-list" onClick={async()=>{
                        try{
                            await Promise.all([
                                saveVideoToDB(item),
                                //saveWatchhistory(item)
                            ]  
                            );
                            handleVideoClick(item.id);
                        } catch(err){
                            console.error('Lỗi khi click video:', err.response ? err.response.data : err.message);
                        }
                    }}>
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
    // return (
    //     <div className="recommended">
    //         {apiData.length > 0 ? (
    //     apiData.map((item, index) => (
    //       <div
    //         key={index}
    //         className="side-video-list"
    //         onClick={() => handleVideoClick(item.video_id)}
    //       >
    //         <img src={`https://img.youtube.com/vi/${item.video_id}/mqdefault.jpg`} alt={item.title} />
    //         <div className="vid-info">
    //           <h4>{item.title}</h4>
    //           <p>{`Score: ${item.score.toFixed(2)}`}</p>
    //         </div>
    //       </div>
    //     ))
    //   ) : (
    //     <p>Không có video gợi ý.</p>
    //   )}
    //     </div>
    // );
    
}

export default Recommended