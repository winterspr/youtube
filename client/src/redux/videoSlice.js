// import { createSlice } from "@reduxjs/toolkit";

// const initialState = {
//     currentVideo: null,
//     loading: false,
//     error:  false,
// };

// export const videoSlice = createSlice({
//     name: 'video',
//     initialState,
//     reducers: {
//         fetchStart: (start)=>{
//             start.loading = true;
//         },

//         fetchSuccess: (state,  action) => {
//             state.loading =  false;
//             state.currentVideo = action.payload;
//         },

//         fetchFailure: (state)=>{
//             state.loading = false;
//             state.error = true;
//         },

//         like: (state,  action) => {
//             if (!state.currentVideo.likes.includes(action.payload)) {
//                 state.currentVideo.likes.push(action.payload);
//                 state.currentVideo.dislikes.splice(
//                   state.currentVideo.dislikes.findIndex(
//                     (userId) => userId === action.payload
//                   ),
//                   1
//                 );
//               }
//         },

//         dislike:  (state,  action) => {
//             if(!state.currentVideo.dislikes.includes(action.payload)){
//                 state.currentVideo.dislikes.push(action.payload);
//                 state.currentVideo.likes.splice(
//                     state.currentVideo.likes.findIndex(
//                         (userId)=>userId === action.payload
//                     ),
//                     1
//                 )
//             }
//         }

//     }
// })

// export const {fetchStart, fetchFailure,  fetchSuccess, like, dislike} = videoSlice.actions;
// export  default videoSlice.reducer;

import { createSlice } from "@reduxjs/toolkit";

const initialState = {
  currentVideo: null,
  loading: false,
  error: false,
};

export const videoSlice = createSlice({
  name: "video",
  initialState,
  reducers: {
    fetchStart: (start) => {
      start.loading = true;
    },

    fetchSuccess: (state, action) => {
      state.loading = false;
      state.currentVideo = action.payload;
    },

    fetchFailure: (state) => {
      state.loading = false;
      state.error = true;
    },

    like: (state, action) => {
      const userId = action.payload;
      const { likes, dislikes } = state.currentVideo;

      if (likes.includes(userId)) {
        // Toggle off like
        state.currentVideo.likes = likes.filter((id) => id !== userId);
      } else {
        // Like và loại bỏ dislike nếu có
        state.currentVideo.likes.push(userId);
        state.currentVideo.dislikes = dislikes.filter((id) => id !== userId);
      }
    },

    dislike: (state, action) => {
      const userId = action.payload;
      const { likes, dislikes } = state.currentVideo;

      if (dislikes.includes(userId)) {
        // Toggle off dislike
        state.currentVideo.dislikes = dislikes.filter((id) => id !== userId);
      } else {
        // Dislike và loại bỏ like nếu có
        state.currentVideo.dislikes.push(userId);
        state.currentVideo.likes = likes.filter((id) => id !== userId);
      }
    },
  },
});

export const { fetchStart, fetchFailure, fetchSuccess, like, dislike } =
  videoSlice.actions;
export default videoSlice.reducer;
