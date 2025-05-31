// import express from "express";
// import multer from "multer";
// import path from "path";

// const router = express.Router();

// // Cấu hình multer lưu file vào thư mục 'uploads'
// const storage = multer.diskStorage({
//   destination: function (req, file, cb) {
//     cb(null, "uploads/");
//   },
//   filename: function (req, file, cb) {
//     cb(null, Date.now() + path.extname(file.originalname)); // Tạo tên file mới
//   },
// });
// const upload = multer({
//   storage: storage,
//   fileFilter: (req, file, cb) => {
//     const allowedTypes = [
//       "image/jpeg",
//       "image/png",
//       "image/jpg",
//       "application/pdf",
//     ];
//     if (!allowedTypes.includes(file.mimetype)) {
//       return cb(new Error("Invalid file type"), false);
//     }
//     cb(null, true);
//   },
//   limits: { fileSize: 5 * 1024 * 1024 }, // 5MB
// });

// router.post("/", upload.single("file"), (req, res) => {
//   console.log("Received file:", req.file);
//   if (!req.file) return res.status(400).json({ message: "No file uploaded" });
//   const fullUrl =
//     req.protocol + "://" + req.get("host") + "/uploads/" + req.file.filename;
//   res.status(200).json({ url: fullUrl });
// });

// export default router;

import express from "express";
import multer from "multer";
import path from "path";
import fs from "fs";

const router = express.Router();

// Tạo folder nếu chưa tồn tại
const ensureDir = (dir) => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
};

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    let folder = "";
    if (file.mimetype.startsWith("image/")) {
      folder = "uploads/images";
    } else if (file.mimetype.startsWith("video/")) {
      folder = "uploads/videos";
    } else {
      return cb(new Error("Invalid file type"), false);
    }

    ensureDir(folder);
    cb(null, folder);
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + path.extname(file.originalname));
  },
});

const upload = multer({
  storage,
  fileFilter: (req, file, cb) => {
    const allowedTypes = [
      "image/jpeg",
      "image/png",
      "image/jpg",
      "video/mp4",
      "video/mkv",
      "video/webm",
    ];
    if (!allowedTypes.includes(file.mimetype)) {
      return cb(new Error("Invalid file type"), false);
    }
    cb(null, true);
  },
  limits: { fileSize: 200 * 1024 * 1024 }, // 200MB max
});

// Upload ảnh hoặc video
router.post("/", upload.single("file"), (req, res) => {
  if (!req.file) return res.status(400).json({ message: "No file uploaded" });

  const fileUrl = `${req.protocol}://${req.get("host")}/${req.file.path.replace(/\\/g, "/")}`;
  res.status(200).json({ url: fileUrl });
});

export default router;
