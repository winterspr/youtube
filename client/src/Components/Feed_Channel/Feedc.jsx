import React, { useState, useEffect } from 'react'
import './Feedc.css'
import {Link, useParams} from 'react-router-dom'
import { API_KEY } from '../../data'
import { value_converter } from '../../data'
import { formatDistanceToNow } from 'date-fns'

const Feedc = () => {
    const {channelId} = useParams();
    //const {category} = useParams();
    const [channelTitle, setChannelTitle] = useState([]);
    const [channelData, setChannelData] = useState([]);
    const [apiData, setApiData] = useState([]);
    
    const fetchData1 = async()=>{
        const channel = `https://youtube.googleapis.com/youtube/v3/channels?part=snippet%2CcontentDetails%2Cstatistics&id=${channelId}&key=${API_KEY}`
        await fetch(channel).then(response=>response.json()).then(data=>setChannelTitle(data.items[0]))
    }

    const fetchData = async()=>{
        const channelSections = `https://youtube.googleapis.com/youtube/v3/playlistItems?part=snippet%2CcontentDetails&playlistId=${channelTitle.contentDetails.relatedPlaylists.uploads}&key=${API_KEY}`
        await fetch(channelSections).then(response=>response.json()).then(data=>setChannelData(data.items))
    }

    // const fetchVideoData = async()=>{
    //     const videoDetails_url = `https://youtube.googleapis.com/youtube/v3/videos?part=snippet%2CcontentDetails%2Cstatistics&id=${channelData.snippet.resourceId.videoId}&key=${API_KEY}`
    //     await fetch(videoDetails_url).then(res=>res.json()).then(data=>setApiData(data.items[0]))
    // }

    useEffect(()=>{
        fetchData1();
    }, [channelId])

    useEffect(()=>{
        fetchData();
    }, [channelTitle])
    // useEffect(()=>{
    //     fetchVideoData();
    // },[videoId])
    
  return (
    <div className='feed'>
        {channelData.map((item, index)=>{
            return (
            <Link to={`/video/${item.snippet.resourceId.videoId}`} className="card">
                <img src={item.snippet.thumbnails.maxres.url} alt="" />
                <h2>{item.snippet.title}</h2>
                <h3>{item.snippet.channelTitle}</h3>
                {/* <p>{value_converter(item.statistics.viewCount)} views &bull; {formatDistanceToNow(item.snippet.publishedAt, {addSuffix: true}).replace('about', '')}</p> */}
            </Link>
            )
        })}
    </div>  
  )
}
export default Feedc