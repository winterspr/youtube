import express from "express";
import {addVideo,updateVideo, deleteVideo, addView, getByTag, getVideo, getMyVideos, random, search, sub, trend, updateWatchTime, recommendVideo, likeVideo, dislikeVideo } from "../controllers/video.js";
import { verifyToken } from "../verifyToken.js";

const router = express.Router();

//create a video
// router.post("/save", saveVideo)
router.post("/add", verifyToken, addVideo)
router.put("/:id", verifyToken, updateVideo)
router.delete("/:id", verifyToken, deleteVideo)
router.get("/find/:id", getVideo)
router.get("/myvideos", verifyToken, getMyVideos);
router.put("/view/:id", addView)
router.put("/watchtime/:id", verifyToken, updateWatchTime);
router.get("/trend", trend)
router.get("/random", random)
router.get("/sub",verifyToken, sub)
router.get("/tags", getByTag)
router.get("/search", search)
router.get("/recommend", verifyToken, recommendVideo);
router.put("/like/:id", verifyToken, likeVideo);
router.put("/dislike/:id", verifyToken, dislikeVideo);

export default router;