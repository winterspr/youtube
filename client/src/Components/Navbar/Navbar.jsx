import React, { useEffect, useState } from "react";
import "./Navbar.css";
import menu_icon from "../../assets/menu.png";
import logo from "../../assets/logo.png";
import search_icon from "../../assets/search.png";
import AccountCircleOutlinedIcon from "@mui/icons-material/AccountCircleOutlined";
import VideoCallOutLinedIcon from "@mui/icons-material/VideoCallOutlined";
import { Link, useNavigate } from "react-router-dom";
import { useSelector, useDispatch } from "react-redux";
import { logout } from "../../redux/userSlice";
import axios from "axios";

const Navbar = ({ setSideBar }) => {
  const { currentUser } = useSelector((state) => state.user);
  const [showDropdown, setShowDropdown] = useState(false);
  const [searchQuery, setSearchQuery] = useState(""); // State để lưu từ khóa tìm kiếm
  const [searchResults, setSearchResults] = useState([]);
  console.log('searchResults', searchResults);
  console.log('searchQuery', searchQuery);
  const dispatch = useDispatch();
  const navigate = useNavigate();

  const handleProfileClick = () => {
    setShowDropdown((prev) => !prev);
  };

  const handleSignOut = () => {
    dispatch(logout());
    setShowDropdown(false);
    navigate("/signin");
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

  const handleSearch = async () => {
    if (searchQuery.trim()) {
      try {
        const res = await axios.get(`/api/videos/search?q=${searchQuery}`); // Gọi API tìm kiếm
        setSearchResults(res.data); // Lưu kết quả tìm kiếm vào state
      } catch (err) {
        console.error("Error fetching search results:", err);
      }
    } else {
      setSearchResults([]); // Nếu không có từ khóa, xóa kết quả tìm kiếm
    }
  };

  // Hàm xử lý khi người dùng gõ vào ô tìm kiếm
  const handleSearchChange = (e) => {
    setSearchQuery(e.target.value); // Cập nhật từ khóa tìm kiếm
  };

  // Điều hướng đến trang video khi người dùng nhấn vào một video trong kết quả tìm kiếm
  const handleVideoClick = (videoId) => {
    navigate(`/video/${videoId}`); // Điều hướng đến trang video chi tiết
    setSearchResults([]); // Tắt kết quả tìm kiếm sau khi chọn video
  };

  return (
    // <nav className="flex-div">
    //   <div className="nav-left flex-div">
    //     <img
    //       className="menu-icon"
    //       onClick={() => setSideBar((prev) => (prev === false ? true : false))}
    //       src={menu_icon}
    //       alt=""
    //     />
    //     <Link to="/">
    //       <img className="logo" src={logo} alt="" />
    //     </Link>
    //   </div>
    //   <div className="nav-middle flex-div">
    //     <div className="search-box flex-div">
    //       <input type="text" placeholder="Search" />
    //       <img src={search_icon} alt="" />
    //     </div>
    //   </div>
    //   {currentUser ? (
    //     <div className="nav-user flex-div">
    //       <div>
    //         <VideoCallOutLinedIcon />
    //       </div>
    //       <div className="profile-container" onClick={handleProfileClick}>
    //         <img className="profile" src={currentUser.img} alt="profile" />
    //         <div>{currentUser.name}</div>
    //       </div>
    //       {showDropdown && (
    //         <div className="profile-dropdown">
    //           <Link to="/profile" className="dropdown-item">
    //             Profile Settings
    //           </Link>
    //           <div className="dropdown-item" onClick={handleSignOut}>
    //             Sign Out
    //           </div>
    //         </div>
    //       )}
    //     </div>
    //   ) : (
    //     <Link to="signin" className="nav-right flex-div">
    //       <AccountCircleOutlinedIcon />
    //       <h3>SIGN IN</h3>
    //     </Link>
    //   )}
    // </nav>
    <nav className='flex-div'>
      <div className='nav-left flex-div'>
        <img className='menu-icon' onClick={() => setSideBar(prev => !prev)} src={menu_icon} alt="" />
        <Link to='/'><img className='logo' src={logo} alt="" /></Link>
      </div>

      <div className='nav-middle flex-div'>
        <div className='search-box flex-div'>
          <input
            type="text"
            placeholder='Search'
            value={searchQuery} // Liên kết với state searchQuery
            onChange={handleSearchChange} // Cập nhật từ khóa tìm kiếm khi người dùng gõ
          />
          <img src={search_icon} alt="" onClick={handleSearch} /> {/* Khi nhấn vào icon, gọi handleSearch */}
        </div>

        {searchResults.length > 0 && (
          <div className="search-results-dropdown">
            {searchResults.map((result) => (
              <div
                key={result._id}
                className="search-result-item"
                onClick={() => handleVideoClick(result._id)} // Điều hướng đến video khi nhấn
              >
                <img src={result.thumbnails.medium.url} alt={result.title} />
                <div>{result.title}</div>
              </div>
            ))}
          </div>
        )}
      </div>

      {currentUser ? (
        <div className='nav-user flex-div'>
          <div>
            <VideoCallOutLinedIcon />
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
  );
};

export default Navbar;
