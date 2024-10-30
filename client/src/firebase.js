import { initializeApp } from "firebase/app";
// import { getAnalytics } from "firebase/analytics";
import {getAuth ,GoogleAuthProvider} from "firebase/auth";
const firebaseConfig = {
  apiKey: "AIzaSyDvLIIIDm0fDpD0KVPwrfWDcPwSUG8OIeg",
  authDomain: "video-d8e19.firebaseapp.com",
  projectId: "video-d8e19",
  storageBucket: "video-d8e19.appspot.com",
  messagingSenderId: "647254606632",
  appId: "1:647254606632:web:d2c873d62bae4261db461d",
  measurementId: "G-PNXD92EXWX"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
// const analytics = getAnalytics(app);
export const auth = getAuth();
export const provider = new GoogleAuthProvider();

export default app;