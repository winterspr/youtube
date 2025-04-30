import mongoose from "mongoose"
import User from "../models/User.js"
import bcrypt from "bcryptjs"
import { createError } from "../error.js";
import jwt from "jsonwebtoken"

export const signup = async(req, res, next)=>{
    try{
        const salt = bcrypt.genSaltSync(10);
        const hash = bcrypt.hashSync(req.body.password, salt);
        const newUser = new User({...req.body, password: hash})

        await newUser.save();
        const token = jwt.sign({ id: newUser._id }, process.env.JWT);
        const { password, ...others } = newUser._doc;
        
        res
          .cookie("access_token", token, {
            httpOnly: true,
            //maxAge: 1000 * 60 * 60 * 24 * 365,
          })
          .status(200)
          .json(others);
    } catch(err){
        next(err)
    }
}   

export const signin = async (req, res, next) => {
    try {
      const user = await User.findOne({ email: req.body.email });
      if (!user) return next(createError(404, "User not found!"));
  
      const isCorrect = await bcrypt.compare(req.body.password, user.password);
  
      if (!isCorrect) return next(createError(400, "Wrong Credentials!"));
  
      const token = jwt.sign({ id: user._id }, process.env.JWT);
      console.log('token', token);
      const { password, ...others } = user._doc;
      res.cookie("access_token", token, {
          httpOnly: false,
          secure: true,
          //maxAge: 1000 * 60 * 60 * 24 * 365,
        })
        .status(200)
        .json(others);
    } catch (err) {
      next(err);
    }
  };

  export const logout = (req, res) => {
    res.clearCookie("access_token", {
      httpOnly: true,
    });
    res.status(200).json("User has been logged out");
  };

  export const googleAuth = async (req, res, next) => {
    try {
      const user = await User.findOne({ email: req.body.email });
      if (user) {
        const token = jwt.sign({ id: user._id }, process.env.JWT);
        console.log('token', token);
        res
          .cookie("access_token", token, {
            httpOnly: false,
            secure: true,
          })
          .status(200)
          .json(user._doc);
      } else {
        const newUser = new User({
          ...req.body,
          fromGoogle: true,
        });
        const savedUser = await newUser.save();
        const token = jwt.sign({ id: savedUser._id }, process.env.JWT);
        res
          .cookie("access_token", token, {
            httpOnly: true,
          })
          .status(200)
          .json(savedUser._doc);
      }
    } catch (err) {
      next(err);
    }
  };