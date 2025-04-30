import React, { useEffect, useState } from 'react'
import './Navbar.css'
import menu_icon from '../../assets/menu.png'
import logo from '../../assets/logo.png'
import search_icon from '../../assets/search.png'
import AccountCircleOutlinedIcon from "@mui/icons-material/AccountCircleOutlined";
import VideoCallOutLinedIcon from "@mui/icons-material/VideoCallOutlined"
import { Link, useNavigate } from "react-router-dom";
import { useSelector, useDispatch } from 'react-redux';
import { logout } from '../../redux/userSlice'

const Navbar = ({setSideBar}) => {
    const { currentUser } = useSelector((state) => state.user);
    const [showDropdown, setShowDropdown] = useState(false);
    const dispatch = useDispatch();
    const navigate = useNavigate();

    const handleProfileClick = () => {
      setShowDropdown(prev => !prev);
    };

    const handleSignOut = () => {
      dispatch(logout());
      setShowDropdown(false);
      navigate('/signin');
    };

    // const getCookie = (name) => {
    //     const cookie = document.cookie.split('; ').find(row => row.startsWith(name));
    //     if (cookie) {
    //       return cookie.split('=')[1];
    //     }
    //     return undefined;
    //   };
      
    //   useEffect(() => {
    //     const token = getCookie('access_token');
    //     if (!token) {
    //       dispatch(logout());
    //       navigate('/signin');
    //     }
    //   }, [dispatch, navigate]);
    
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
                <div>
                    <VideoCallOutLinedIcon/>
                </div>
                <div className="profile-container" onClick={handleProfileClick}>
                        <img className="profile" src={currentUser.img} alt="profile" />
                        <div>{currentUser.name}</div>
                    </div>
                    {showDropdown && (
                        <div className="profile-dropdown">
                            <Link to="/profile" className="dropdown-item">Profile Settings</Link>
                            <div className="dropdown-item" onClick={handleSignOut}>Sign Out</div>
                        </div>
                    )}
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