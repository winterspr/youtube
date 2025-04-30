import express from "express";
import { googleAuth, signup, signin } from "../controllers/auth.js";
import {verifyToken} from "../verifyToken.js"

const router = express.Router();

//CREATE A USER
router.post("/signup", signup);

//SIGN IN
router.post("/signin", signin);

// //GOOGLE AUTH
router.post("/google", googleAuth);

export default router;