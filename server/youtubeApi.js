import axios from 'axios';
import dotenv from "dotenv"
dotenv.config()

async function fetchVideosFromSearch(videoId) {
    const API_KEY = process.env.API_KEY;
    const url = `https://youtube.googleapis.com/youtube/v3/videos?part=snippet%2CcontentDetails%2Cstatistics&id=${videoId}&maxResults=50&key=${API_KEY}`;
  
    try {
      const response = await axios.get(url);
      return response.data.items;
    } catch (error) {
      console.error('Error fetching videos: ', error);
      return [];
    }
  }
module.exports = {fetchVideosFromSearch}