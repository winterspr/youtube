import React from 'react'
import './Video2.css'
import PlayVideo from '../../Components/PlayVideo/PlayVideo.jsx'
import Recommended2 from '../../Components/Recommended2/Recommended2.jsx'
import { useParams } from 'react-router-dom'
const Video2 = () => {
  const {videoId, categoryId} = useParams();
  return (
    <div className='play-container'>
      <PlayVideo videoId={videoId}/>
      <Recommended2 videoId={videoId}/>
    </div>
  )
}

export default Video2