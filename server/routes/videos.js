import express from "express";
import {addVideo,updateVideo, deleteVideo, addView, getByTag, getVideo, random, search, sub, trend, updateWatchTime } from "../controllers/video.js";
import { verifyToken } from "../verifyToken.js";

const router = express.Router();

//create a video
// router.post("/save", saveVideo)
router.post("/add", verifyToken, addVideo)
router.put("/:id", verifyToken, updateVideo)
router.delete("/:id", verifyToken, deleteVideo)
router.get("/find/:id", getVideo)
router.put("/view/:id", addView)
router.put("/watchtime/:id", updateWatchTime);
router.get("/trend", trend)
router.get("/random", random)
router.get("/sub",verifyToken, sub)
router.get("/tags", getByTag)
router.get("/search", search)

export default router;