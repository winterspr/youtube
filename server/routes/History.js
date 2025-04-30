import express from 'express';
import { saveHistory } from '../controllers/History.js';
import { verifyToken } from '../verifyToken.js';

const router = express.Router();

router.post('/add', verifyToken, saveHistory);

export default router;