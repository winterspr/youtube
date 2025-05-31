import React, { useEffect, useRef, useState } from "react";
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
  const [showProfileDropdown, setShowProfileDropdown] = useState(false); // Dropdown cho profile
  const [showSearchResults, setShowSearchResults] = useState(false); // Kết quả tìm kiếm
  const [searchQuery, setSearchQuery] = useState(""); // State lưu từ khóa tìm kiếm
  const [searchResults, setSearchResults] = useState([]);
  const dispatch = useDispatch();
  const navigate = useNavigate();

  const dropdownRef = useRef(null);
  const searchBoxRef = useRef(null);

  const handleProfileClick = () => {
    setShowProfileDropdown((prev) => !prev);
    setShowSearchResults(false); // Đảm bảo khi mở profile dropdown thì đóng search results
  };

  const handleSignOut = () => {
    dispatch(logout());
    setShowProfileDropdown(false);
    navigate("/signin");
  };

  const handleSearch = async (query) => {
    if (query.trim()) {
      try {
        const res = await axios.get(`/api/videos/search?q=${query}`);
        setSearchResults(res.data);
        setShowSearchResults(true); // Hiển thị kết quả tìm kiếm
      } catch (err) {
        console.error("Error fetching search results:", err);
      }
    } else {
      setSearchResults([]);
      setShowSearchResults(false); // Ẩn kết quả tìm kiếm khi không có query
    }
  };

  const handleSearchChange = (e) => {
    const query = e.target.value;
    setSearchQuery(query);
    handleSearch(query);
  };

  const handleVideoClick = (videoId) => {
    navigate(`/video/${videoId}`);
    setSearchResults([]);
    setShowSearchResults(false); // Ẩn kết quả tìm kiếm khi click vào video
  };

  const handleClickOutside = (e) => {
    if (searchBoxRef.current && !searchBoxRef.current.contains(e.target)) {
      setShowSearchResults(false);
    }
    if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
      setShowProfileDropdown(false);
    }
  };

  useEffect(() => {
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  return (
    <nav className="flex-div">
      <div className="nav-left flex-div">
        <img
          className="menu-icon"
          onClick={() => setSideBar((prev) => !prev)}
          src={menu_icon}
          alt=""
        />
        <Link to="/">
          <img className="logo" src={logo} alt="" />
        </Link>
      </div>

      <div className="nav-middle flex-div">
        <div className="search-area" ref={searchBoxRef}>
          <div className="search-box flex-div">
            <input
              type="text"
              placeholder="Search"
              value={searchQuery}
              onChange={handleSearchChange}
            />
            <img src={search_icon} alt="" />
          </div>

          {showSearchResults && searchResults.length > 0 && (
            <div className="search-results-dropdown">
              {searchResults.map((result) => (
                <div
                  key={result._id}
                  className="search-result-item"
                  onClick={(e) => {
                    e.stopPropagation(); // Ngăn event "click outside"
                    handleVideoClick(result._id);
                  }}
                >
                  <img src={result.thumbnails.medium.url} alt={result.title} />
                  <div>{result.title}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {currentUser ? (
        <div className="nav-user flex-div">
          <div onClick={() => navigate("/upload-video")} className="upload-icon">
            <VideoCallOutLinedIcon />
          </div>
          <div className="profile-container" onClick={handleProfileClick}>
            <img className="profile" src={currentUser.img} alt="profile" />
            <div>{currentUser.name}</div>
          </div>
          {showProfileDropdown && (
            <div className="profile-dropdown" ref={dropdownRef}>
              <Link to="/profile/settings" className="dropdown-item">
                Profile Settings
              </Link>
              <Link to="/video/settings" className="dropdown-item">
                My videos
              </Link>
              <div className="dropdown-item" onClick={handleSignOut}>
                Sign Out
              </div>
            </div>
          )}
        </div>
      ) : (
        <Link to="/signin" className="nav-right flex-div">
          <AccountCircleOutlinedIcon />
          <h3>SIGN IN</h3>
        </Link>
      )}
    </nav>
  );
};

export default Navbar;
