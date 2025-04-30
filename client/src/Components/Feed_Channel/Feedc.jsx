// 

import React, { useState, useEffect } from 'react'
import './Feedc.css'
import { Link, useParams } from 'react-router-dom'
import { API_KEY } from '../../data'
import { useSelector } from 'react-redux'
import axios from 'axios'
import Cookies from 'js-cookie';

const Feedc = ({channelId}) => {
    //const { channelId } = useParams();
    const token = Cookies.get('access_token');

    const [channelTitle, setChannelTitle] = useState(null);
    const [channelData, setChannelData] = useState([]);
    const { currentUser } = useSelector((state) => state.user);

    // Lấy dữ liệu kênh
    const fetchChannelData = async () => {
        const channelUrl = `https://youtube.googleapis.com/youtube/v3/channels?part=snippet%2CcontentDetails%2Cstatistics&id=${channelId}&key=${API_KEY}`;
        try {
            const response = await axios.get(channelUrl);
            setChannelTitle(response.data.items[0]);
        } catch (error) {
            console.error("Error fetching channel data:", error);
        }
    };

    // Lấy danh sách video từ playlist của kênh
    const fetchChannelVideos = async () => {
        if (!channelTitle?.contentDetails?.relatedPlaylists?.uploads) return;

        const playlistId = channelTitle.contentDetails.relatedPlaylists.uploads;
        const videoListUrl = `https://youtube.googleapis.com/youtube/v3/playlistItems?part=snippet%2CcontentDetails&playlistId=${playlistId}&key=${API_KEY}`;

        try {
            const response = await axios.get(videoListUrl);
            setChannelData(response.data.items);  // Lưu danh sách video
        } catch (error) {
            console.error("Error fetching channel videos:", error);
        }
    };

    // Gọi API khi thay đổi channelId hoặc kênh được lấy thành công
    useEffect(() => {
        fetchChannelData();
    }, []);

    useEffect(() => {
        if (channelTitle) {
            fetchChannelVideos();
        }
    }, [channelTitle]);

    // Lưu video vào cơ sở dữ liệu
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
    };

    // Lưu lịch sử xem video
    const saveWatchhistory = async (item) => {
        try {
            const historyData = {
                videoId: item.id,
                userId: currentUser._id,
                title: item.snippet.title,
                channelTitle: item.snippet.channelTitle,
                publishedAt: item.snippet.publishedAt,
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
                }
            };

            const response = await axios.post('/api/history/add', historyData);
            if (response.status === 201) {
                console.log('Saved watch history successfully');
            }
        } catch (error) {
            console.error('Error:', error.response ? error.response.data : error.message);
        }
    };

    return (
        <div className='feed'>
            {channelData.map((item) => (
                <Link
                    to={`/video/${item.contentDetails.videoId}`}
                    className="card"
                    key={item.snippet.resourceId.videoId}
                    onClick={async () => {
                        try {
                            await Promise.all([
                                saveVideoToDB(item),
                                saveWatchhistory(item)
                            ]);
                        } catch (err) {
                            console.error('Error:', err.message);
                        }
                    }}
                >
                    <img
                        src={item.snippet.thumbnails.default?.url || ''}
                        alt={item.snippet.title}
                    />
                    <h2>{item.snippet.title}</h2>
                    <h3>{item.snippet.channelTitle}</h3>
                </Link>
            ))}
        </div>
    );
};

export default Feedc;
