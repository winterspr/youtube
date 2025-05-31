import React, { useState } from 'react';
import './ProfileSetting.css';
import { useSelector, useDispatch } from 'react-redux';
import axios from 'axios';
import { updateSuccess } from '../../redux/userSlice';

const ProfileSettings = () => {
  const dispatch = useDispatch();
  const { currentUser } = useSelector((state) => state.user);

  const [name, setName] = useState(currentUser?.name || '');
  const [email, setEmail] = useState(currentUser?.email || '');
  const [avatar, setAvatar] = useState(currentUser?.img || '');
  const [uploading, setUploading] = useState(false);

  // Hàm xử lý upload ảnh file
  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);

    try {
      setUploading(true);
      // Gửi file lên API upload, backend trả về URL ảnh
      const res = await axios.post('/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setAvatar(res.data.url); // Giả sử backend trả về { url: 'link_anh_upload' }
      setUploading(false);
    } catch (err) {
      console.error('Upload thất bại:', err);
      alert('Upload ảnh thất bại!');
      setUploading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await axios.put(`/api/users/${currentUser._id}`, {
        name,
        email,
        img: avatar,
      });
      dispatch(updateSuccess(res.data));
      alert('Cập nhật thành công!');
    } catch (err) {
      alert('Cập nhật thất bại!');
      console.error(err);
    }
  };

  return (
    <div className="profile-settings">
      <h2>Cài đặt hồ sơ</h2>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label>Ảnh đại diện:</label>
          <img src={avatar} alt="avatar" className="avatar-preview" />
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
          />
          {uploading && <p>Đang tải ảnh lên...</p>}
          <p>Hoặc nhập URL ảnh đại diện:</p>
          <input
            type="text"
            value={avatar}
            onChange={(e) => setAvatar(e.target.value)}
            placeholder="URL ảnh đại diện"
          />
        </div>

        <div className="form-group">
          <label>Tên hiển thị:</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Tên của bạn"
          />
        </div>

        <div className="form-group">
          <label>Email:</label>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Email của bạn"
          />
        </div>

        <button type="submit" className="save-btn">Lưu thay đổi</button>
      </form>
    </div>
  );
};

export default ProfileSettings;
