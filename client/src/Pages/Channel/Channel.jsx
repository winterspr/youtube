import React, {useState} from 'react'
import './Channel.css'
import Sidebar from '../../Components/Sidebar/Sidebar.jsx'
import Feedc from "../../Components/Feed_Channel/Feedc.jsx"
import { useParams } from 'react-router-dom'
const Channel = ({sidebar}) => {
    const [category, setCategory] = useState([]);
    const {channelId} = useParams();
    return (
        <>
          <Sidebar sidebar={sidebar} category={category} setCategory={setCategory}></Sidebar>
          
          <div className={`container ${sidebar?"":"large-container"}`}>
            <Feedc channelId={channelId}/>
          </div>
          
        </>
      )
}

export default Channel