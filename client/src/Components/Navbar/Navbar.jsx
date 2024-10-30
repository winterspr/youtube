import React from 'react'
import './Navbar.css'
import menu_icon from '../../assets/menu.png'
import logo from '../../assets/logo.png'
import search_icon from '../../assets/search.png'
import AccountCircleOutlinedIcon from "@mui/icons-material/AccountCircleOutlined";
import VideoCallOutLinedIcon from "@mui/icons-material/VideoCallOutlined"
import { Link, useNavigate } from "react-router-dom";
import { useSelector } from 'react-redux'

const Navbar = ({setSideBar}) => {
    const { currentUser } = useSelector((state) => state.user);
  return (
    <nav className='flex-div'>
        <div className='nav-left flex-div'>
            <img className='menu-icon' onClick={()=>setSideBar(prev=>prev===false?true:false)} src={menu_icon} alt="" />
            <Link to='/'><img className='logo' src={logo} alt="" /></Link>
        </div>
        <div className='nav-middle flex-div'>
            <div className='search-box flex-div'>
                <input type="text" placeholder='Search'/>
                <img src={search_icon} alt="" />
            </div> 
        </div>
        {currentUser ? (
            <div className='nav-user flex-div'>
                <VideoCallOutLinedIcon/>
                <img className="profile" src={currentUser.img}/>
                    {currentUser.name}
            </div>
        ) : (
            <Link to="signin" className="nav-right flex-div">
                <AccountCircleOutlinedIcon /><h3>SIGN IN</h3>
            </Link>
        )}
    </nav>
  )
}

export default Navbar 