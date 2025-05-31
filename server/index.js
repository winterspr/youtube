import express from "express"
import mongoose from "mongoose";
import dotenv from "dotenv"
import userRoutes from "./routes/users.js";
import videoRoutes from "./routes/videos.js";
import commentRoutes from "./routes/comments.js";
import uploadRoutes from './routes/uploads.js';
import authRoutes from "./routes/auth.js";
import HistoryRouter from "./routes/History.js";
import cookieParser from "cookie-parser"
import cors from  "cors"
import fs from 'fs';

if (!fs.existsSync('uploads')) {
  fs.mkdirSync('uploads');
}

const app = express();
dotenv.config()
const connect = ()=>{
    mongoose.connect(process.env.MONGO).then(()=>{
        console.log("Connected to DB")
    }).catch((err)=>{
        throw err;
    })
}

app.use(cors())
app.use("/uploads/images", express.static("uploads/images"));
app.use("/uploads/videos", express.static("uploads/videos"));
app.use(cookieParser())
app.use(express.json())
app.use("/api/auth", authRoutes);
app.use("/api/users", userRoutes);
app.use("/api/videos", videoRoutes);
app.use("/api/comments", commentRoutes);
app.use("/api/history", HistoryRouter);
app.use("/api/upload", uploadRoutes);

app.use((err, req, res, next)=>{
    const status = err.status || 500;
    const message = err.message || "something went wrong!"
    return res.status(status).json({
        success: false,
        status,
        message,
    })
})

app.listen('8000',()=> {
    connect();
    console.log("Connected!")
})