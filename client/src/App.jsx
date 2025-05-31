import React, { useState } from 'react'
import { ThemeProvider } from 'styled-components'
import  styled  from 'styled-components'
import Navbar from './Components/Navbar/Navbar.jsx'
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from './Pages/Home/Home.jsx'
import Video from './Pages/Video/Video.jsx'
import { darkTheme, lightTheme } from './utils/Theme.js';
import Signin from './Pages/Signin/Signin.jsx';
import Channel from './Pages/Channel/Channel.jsx';
import Video2 from './Pages/Video2/Video2.jsx';
import ProfileSettings from './Components/Profile_Setting/ProfileSetting.jsx';
import UploadVideo from './Components/Upload_Video/Upload_Video.jsx';

function App(){
  const [sidebar, setSideBar] = useState(true);
  const [darkMode, setDarkMode] = useState(true);
  return (
    <ThemeProvider theme={darkMode ? darkTheme : lightTheme}>
      <BrowserRouter>
        <Navbar darkMode={darkMode} setDarkMode={setDarkMode} setSideBar={setSideBar}/>
          <Routes>
            <Route path='/' element={<Home sidebar={sidebar}/>}/>
            <Route path='/video/:categoryId/:videoId' element={<Video/>}/>
            <Route path='/video/:videoId' element={<Video2/>}/>
            <Route path='/signin' element={<Signin/>}/>
            <Route path='/channel/:channelId' element={<Channel/>}></Route>
            <Route path='/profile/settings' element={<ProfileSettings/>}/>
            <Route path="/upload-video" element={<UploadVideo />} />
          </Routes>
      </BrowserRouter>
    </ThemeProvider>
  )
}

export default App