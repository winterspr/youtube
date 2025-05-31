// import React, { useState } from "react";
// import axios from "axios";
// import { useDispatch, useSelector } from "react-redux";
// import Cookies from 'js-cookie';
// import "./Upload_Video.css";
// import { v4 as uuidv4 } from "uuid";

// const UploadVideo = () => {
//   const [title, setTitle] = useState("");
//   const [desc, setDesc] = useState("");
//   const [tags, setTags] = useState("");
//   const [videoFile, setVideoFile] = useState(null);
//   const [thumbnailFile, setThumbnailFile] = useState(null);
//   const { currentUser } = useSelector((state) => state.user);
//   const token = Cookies.get('access_token');
//   console.log('token', token);

// const handleUpload = async (e) => {
//   e.preventDefault();
//   try {
//     if (!videoFile || !thumbnailFile) {
//       alert("Vui lòng chọn video và ảnh thumbnail");
//       return;
//     }

//     // 1. Upload video file -> nhận url video
//     const videoForm = new FormData();
//     videoForm.append("file", videoFile);
//     const videoRes = await axios.post("/api/upload", videoForm);
//     const videoUrl = videoRes.data.url; // Phải là string

//     // 2. Upload thumbnail file -> nhận url ảnh thumbnail
//     const thumbForm = new FormData();
//     thumbForm.append("file", thumbnailFile);
//     const thumbRes = await axios.post("/api/upload", thumbForm);
//     const imgUrl = thumbRes.data.url; // Phải là string

//     console.log("Video URL:", videoUrl);
//     console.log("Thumbnail URL:", imgUrl);

//     // 3. Gửi dữ liệu video (metadata) lên server
//     const newVideoId = uuidv4();
//     const res = await axios.post(
//       "/api/videos/add",
//       {
//         _id: newVideoId,
//         title,
//         description: desc,
//         tags: tags ? tags.split(",").map((t) => t.trim()) : [],
//         thumbnails: {
//           default: { url: imgUrl, width: 120, height: 90 },
//           medium: { url: imgUrl, width: 320, height: 180 },
//           high: { url: imgUrl, width: 480, height: 360 },
//         },
//         videoUrl, // bạn cần chắc chắn schema backend có trường videoUrl hoặc đổi tên theo schema
//       },
//       {
//         headers: {
//           Authorization: `Bearer ${token}`,
//         },
//       }
//     );

//     alert("Upload video thành công!");
//   } catch (err) {
//     console.error("Upload failed:", err);
//     alert("Upload thất bại!");
//   }
// };


//   return (
//     <div className="upload-video-page">
//       <h2>Upload Video</h2>
//       <form onSubmit={handleUpload} className="upload-form">
//         <input
//           type="text"
//           placeholder="Tiêu đề"
//           value={title}
//           onChange={(e) => setTitle(e.target.value)}
//           required
//         />
//         <textarea
//           placeholder="Mô tả"
//           value={desc}
//           onChange={(e) => setDesc(e.target.value)}
//           required
//         />
//         <input
//           type="text"
//           placeholder="Tags (cách nhau bằng dấu phẩy)"
//           value={tags}
//           onChange={(e) => setTags(e.target.value)}
//         />
//         <label>Chọn video (.mp4):</label>
//         <input
//           type="file"
//           accept="video/mp4,video/mkv,video/webm"
//           onChange={(e) => setVideoFile(e.target.files[0])}
//           required
//         />
//         <label>Chọn ảnh thumbnail:</label>
//         <input
//           type="file"
//           accept="image/*"
//           onChange={(e) => setThumbnailFile(e.target.files[0])}
//           required
//         />
//         <button type="submit">Tải lên</button>
//       </form>
//     </div>
//   );
// };

// export default UploadVideo;

import React, { useState, useEffect } from "react";
import axios from "axios";
import { useSelector } from "react-redux";
import Cookies from 'js-cookie';
import "./Upload_Video.css";

const UploadVideo = () => {
  const [showUploadForm, setShowUploadForm] = useState(false);
  const [videos, setVideos] = useState([]);
  const [title, setTitle] = useState("");
  const [desc, setDesc] = useState("");
  const [tags, setTags] = useState("");
  const [videoFile, setVideoFile] = useState(null);
  const [thumbnailFile, setThumbnailFile] = useState(null);

  const { currentUser } = useSelector((state) => state.user);
  const token = Cookies.get('access_token');

  // Lấy danh sách video của user
  const fetchUserVideos = async () => {
    try {
    const res = await axios.get(`/api/videos/myvideos`, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    setVideos(res.data);
  } catch (err) {
    console.error("Lỗi khi lấy video:", err);
  }
  };

  useEffect(() => {
    if (currentUser?._id) {
      fetchUserVideos();
    }
  }, [currentUser]);

  const handleUpload = async (e) => {
    e.preventDefault();
    try {
      if (!videoFile || !thumbnailFile) {
        alert("Vui lòng chọn video và ảnh thumbnail");
        return;
      }

      // Upload video file
      const videoForm = new FormData();
      videoForm.append("file", videoFile);
      const videoRes = await axios.post("/api/upload", videoForm);
      const videoUrl = videoRes.data.url;

      // Upload thumbnail
      const thumbForm = new FormData();
      thumbForm.append("file", thumbnailFile);
      const thumbRes = await axios.post("/api/upload", thumbForm);
      const imgUrl = thumbRes.data.url;

      // Gửi metadata video
      await axios.post(
        "/api/videos/add",
        {
          title,
          description: desc,
          tags: tags ? tags.split(",").map((t) => t.trim()) : [],
          thumbnails: {
            default: { url: imgUrl, width: 120, height: 90 },
            medium: { url: imgUrl, width: 320, height: 180 },
            high: { url: imgUrl, width: 480, height: 360 },
          },
          videoUrl,
        },
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      alert("Tải video thành công!");
      setShowUploadForm(false);
      fetchUserVideos(); // cập nhật lại danh sách video
    } catch (err) {
      console.error("Upload failed:", err);
      alert("Upload thất bại!");
    }
  };

  return (
    <div className="upload-video-page">
      <h2>Video của bạn</h2>
      <button onClick={() => setShowUploadForm(!showUploadForm)}>
        {showUploadForm ? "Đóng" : "Tải video lên"}
      </button>

      {showUploadForm && (
        <form onSubmit={handleUpload} className="upload-form">
          <input
            type="text"
            placeholder="Tiêu đề"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            required
          />
          <textarea
            placeholder="Mô tả"
            value={desc}
            onChange={(e) => setDesc(e.target.value)}
            required
          />
          <input
            type="text"
            placeholder="Tags (cách nhau bằng dấu phẩy)"
            value={tags}
            onChange={(e) => setTags(e.target.value)}
          />
          <label>Chọn video (.mp4):</label>
          <input
            type="file"
            accept="video/*"
            onChange={(e) => setVideoFile(e.target.files[0])}
            required
          />
          <label>Chọn ảnh thumbnail:</label>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setThumbnailFile(e.target.files[0])}
            required
          />
          <button type="submit">Tải lên</button>
        </form>
      )}

      <div className="video-list">
        {videos.length === 0 ? (
          <p>Bạn chưa có video nào.</p>
        ) : (
          videos.map((video) => (
            <div key={video._id} className="video-item">
              <img src={video.thumbnails?.medium?.url} alt="thumbnail" />
              <h4>{video.title}</h4>
              <p>{video.description}</p>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default UploadVideo;

