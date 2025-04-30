import React from 'react'
import { useState } from 'react'
import './Signin.css'
import axios  from 'axios'
import { auth, provider } from "../../firebase.js";
import {useDispatch} from  'react-redux'
import {useNavigate} from 'react-router-dom'
import { loginFailure, loginStart, loginSuccess } from "../../redux/userSlice.js";
import { signInWithPopup } from "firebase/auth";

const Signin = () => {
    const dispatch = useDispatch()
    const navigate = useNavigate()
    const [name, setName] = useState("")
    const [email, setEmail] = useState("")
    const [password, setPassword] = useState("")
    const [state, setState] = useState("Login")
    const [isActive, setIsActive] = useState(false)
    const handleRegisterClick = ()=>{
        setIsActive(true)
        setState("Sign Up")
    }
    const handleLoginClick = ()=>{
        setIsActive(false)
        setState("Login")    
    }

    const handleLogin = async(e)=>{
        e.preventDefault();
        dispatch(loginStart());
        try{
            const res = await axios.post("api/auth/signin", {email,password});
            dispatch(loginSuccess(res.data));
            navigate("/")
        } catch(err){
            dispatch(loginFailure());
        }
    }

    const handleRegister = async (e) => {
        e.preventDefault();
        dispatch(loginStart());
        try {
            const res = await axios.post("api/auth/signup", {
                name,
                email,
                password
            });
            dispatch(loginSuccess(res.data));
            navigate("/")
        } catch (err) {
            dispatch(loginFailure());
        }
    }
      

    const signInWithGoogle = async()=>{
        dispatch(loginStart());
        signInWithPopup(auth, provider)
        .then((result)=>{
            axios
                .post("api/auth/google", {
                    name:  result.user.displayName,
                    email: result.user.email,
                    img: result.user.photoURL,
                })
                .then((res)=>{
                    console.log('res', res);
                    dispatch(loginSuccess(res.data));
                    navigate("/")
                });
        })
        .catch((error)=>{
            dispatch(loginFailure());
        })
    }
  return (
    <div className="total">
                <div className={`loginsignup-in ${isActive ? 'active' : ''}`}>
                    <div className="form-container sign-up">
                        <form action="">
                            <h1>{state}</h1>
                            <div className="social-icons">
                                <a onClick={signInWithGoogle} className="icon"><i className="fa-brands fa-google-plus-g"></i></a>
                                <a href="#" className="icon"><i className="fa-brands fa-facebook-f"></i></a>
                                <a href="#" className="icon"><i className="fa-brands fa-github"></i></a>
                                <a href="#" className="icon"><i className="fa-brands fa-instagram"></i></a>
                            </div>
                            <span>or use your email for registertion</span>
                            <input name='username'  type="text" placeholder="Name" onChange={(e)=> setName(e.target.value)}/>
                            <input name="email"  type="email" placeholder="Email" onChange={(e)=> setEmail(e.target.value)}/>
                            <input name="password"  type="password" placeholder="Password" onChange={(e)=> setPassword(e.target.value)}/>
                            <button onClick={handleRegister}>Sign Up </button>
                        </form>
                    </div>

                    <div className="form-container sign-in">
                        <form action="">
                            <h1>{state}</h1>
                            <div className="social-icons">
                                <a onClick={signInWithGoogle} className="icon"><i className="fa-brands fa-google-plus-g"></i></a>
                                <a href="#" className="icon"><i className="fa-brands fa-facebook-f"></i></a>
                                <a href="#" className="icon"><i className="fa-brands fa-github"></i></a>
                                <a href="#" className="icon"><i className="fa-brands fa-instagram"></i></a>
                            </div>
                            <span>or use your email password</span>
                            <input name="name"  type="email" placeholder="Email" onChange={(e)=> setEmail(e.target.value)}/>
                            <input name="password" type="password" placeholder="Password" onChange={(e)=> setPassword(e.target.value)}/>
                            <a href="#">Forget Your Password?</a>
                            <button onClick={handleLogin}>Sign In </button>
                        </form>
                    </div>
                    <div className="toggle-container">
                        <div className="toggle">
                            <div className="toggle-panel toggle-left">
                                <h1>Welcome Back!</h1>
                                <p>Enter your personal details to use all of site features</p>
                                <button className="hidden" id="login" onClick={handleLoginClick}>Sign In</button>
                            </div>
                            <div className="toggle-panel toggle-right">
                                <h1>Hello, Friends!</h1>
                                <p>Register to continue with Youtube</p>
                                <button className="hidden" id="register" onClick={handleRegisterClick}>Sign Up</button>
                            </div>
                        </div>
                     </div>
                </div>
        </div>
  )
}

export default Signin