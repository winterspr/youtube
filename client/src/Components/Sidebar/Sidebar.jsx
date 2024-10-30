import React from 'react'
import './Sidebar.css'
import home from '../../assets/home.png'
import game_icon from '../../assets/game_icon.png'
import automobiles from '../../assets/automobiles.png'
import sports from '../../assets/sports.png'
import entertainment from '../../assets/entertainment.png'
import tech from '../../assets/tech.png'
import music from '../../assets/music.png'
import blogs from '../../assets/blogs.png'
import news from '../../assets/news.png'
import ArrowForwardIosIcon from '@mui/icons-material/ArrowForwardIos';
import AccountBoxOutlinedIcon from '@mui/icons-material/AccountBoxOutlined';
import HistoryOutlinedIcon from '@mui/icons-material/HistoryOutlined';
import PlaylistPlayOutlinedIcon from '@mui/icons-material/PlaylistPlayOutlined';
import SmartDisplayOutlinedIcon from '@mui/icons-material/SmartDisplayOutlined';
import WatchLaterOutlinedIcon from '@mui/icons-material/WatchLaterOutlined';
import ThumbUpAltOutlinedIcon from '@mui/icons-material/ThumbUpAltOutlined';
import AccountCircleOutlinedIcon from "@mui/icons-material/AccountCircleOutlined";
import SettingsOutlinedIcon from "@mui/icons-material/SettingsOutlined";
import FlagOutlinedIcon from "@mui/icons-material/FlagOutlined";
import HelpOutlineOutlinedIcon from "@mui/icons-material/HelpOutlineOutlined";
import { Link, Navigate } from 'react-router-dom'
import { useSelector } from 'react-redux'
const Sidebar = ({sidebar, category, setCategory}) => {
    const {currentUser} = useSelector((state)=> state.user)

  return (
    <div className={`sidebar ${sidebar?"":"small-sidebar"}`}>
        <div className="shortcut-links">
            <div className={`side-link ${category===0?"active":""}`} onClick={()=>setCategory(0)}>
                <img src={home} alt="" /><p>Home</p>
            </div>
            <div className={`side-link ${category===20?"active":""}`} onClick={()=>setCategory(20)}>
                <img src={game_icon} alt="" /><p>Gaming</p>
            </div>
            <div className={`side-link ${category===2?"active":""}`} onClick={()=>setCategory(2)}>
                <img src={automobiles} alt="" /><p>Automobiles</p>
            </div>
            <div className={`side-link ${category===17?"active":""}`} onClick={()=>setCategory(17)}>
                <img src={sports} alt="" /><p>Sports</p>
            </div>
            <div className={`side-link ${category===24?"active":""}`} onClick={()=>setCategory(24)}>
                <img src={entertainment} alt="" /><p>Entertainment</p>
            </div>
            <div className={`side-link ${category===28?"active":""}`} onClick={()=>setCategory(28)}>
                <img src={tech} alt="" /><p>tech</p>
            </div>
            <div className={`side-link ${category===10?"active":""}`} onClick={()=>setCategory(10)}>
                <img src={music} alt="" /><p>Music</p>
            </div>
            <div className={`side-link ${category===22?"active":""}`} onClick={()=>setCategory(22)}>
                <img src={blogs} alt="" /><p>Blogs</p>
            </div>
            <div className={`side-link ${category===25?"active":""}`} onClick={()=>setCategory(25)}>
                <img src={news} alt="" /><p>News</p>
            </div>
            <hr />
        </div>
        {!currentUser ?  (
            <div className="user">
                <p>Sign in to like videos, comment, and subscribe.</p>
                <Link to="/signin" className="user-signin flex-div">
                <AccountCircleOutlinedIcon />
                <p>SIGN IN</p>
                </Link>
                <hr />
            </div>
        ) : (
            <div className='yourchannel'>
                <div className="your_channel">
                    <p>Your</p>
                    <ArrowForwardIosIcon/>
                </div>
                <div className="yourchannel-icon">
                    <AccountBoxOutlinedIcon/>
                    <p>Your Channel</p>
                </div>
                <div className="yourchannel-icon">
                    <HistoryOutlinedIcon/>
                    <p>History</p>
                </div>
                <div className="yourchannel-icon">
                    <PlaylistPlayOutlinedIcon/>
                    <p>Playlists</p>
                </div>
                <div className="yourchannel-icon">
                    <SmartDisplayOutlinedIcon/>
                    <p>Your Videos</p>
                </div>
                <div className="yourchannel-icon">
                    <WatchLaterOutlinedIcon/>
                    <p>Watch Later</p>
                </div>
                <div className="yourchannel-icon">
                    <ThumbUpAltOutlinedIcon/>
                    <p>liked Video</p>
                </div>
                <hr />
            </div>
        )   
        }
        <div className='setting'>
            <div className="setting-icon">
                <SettingsOutlinedIcon/>
                <p>Settings</p>
            </div>
            <div className="setting-icon">
                <FlagOutlinedIcon />
                <p>Report</p>
            </div>
            <div className="setting-icon">
                <HelpOutlineOutlinedIcon />
                <p>Help</p>
            </div>
        </div>
    </div>
  )
}

export default Sidebar