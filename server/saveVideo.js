import Video from './models/Video.js';

async function saveVideoDetails(videoDetails) {
    for (const videoData of videoDetails) {
      const { id: videoId, snippet, contentDetails, statistics } = videoData;
  
      const video = new Video({
        videoId: videoId,
        title: snippet.title,
        description: snippet.description,
        publishedAt: snippet.publishedAt,
        channelId: snippet.channelId,
        channelTitle: snippet.channelTitle,
        thumbnails: {
          default: snippet.thumbnails.default,
          medium: snippet.thumbnails.medium,
          high: snippet.thumbnails.high,
        },
        viewCount: statistics.viewCount || 0,
        likeCount: statistics.likeCount || 0,
        commentCount: statistics.commentCount || 0,
        duration: contentDetails.duration,  // Thời lượng video (ISO 8601)
      });
  
      try {
        await video.save();
        console.log(`Video ${videoId} saved successfully!`);
      } catch (error) {
        console.error(`Error saving video ${videoId}: `, error);
      }
    }
  }

module.exports = {saveVideoDetails}