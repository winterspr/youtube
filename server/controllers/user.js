

import {createError} from "../error.js"
import User from "../models/User.js"
import Video from "../models/Video.js"

//update user
export const update = async(req, res ,next)=>{
  if(req.params.id === req.user.id){
    try{
      const updateUser = await User.findByIdAndUpdate(
        req.params.id,
        {
          $set: req.body,
        },
        {
          new: true,
        }
      );
      res.status(200).json(updateUser)
    } catch(err){
      next(err)
    }
  } else {
    return next(createError(403, "You can update only your account"))
  }
}

//delete user
export const deleteUser = async(req, res, next)=>{
  if(req.params.id === req.user.id){
    try{
      await User.findByIdAndDelete(req.params.id)
      res.status(200).json("Deleted")
    } catch(err){
      next(err);
    }
  } else {
    return next(createError(403, "can't find user to delete"))
  }
}
//get user
export const getUser = async(req, res, next)=>{
  try{
    const user = await User.findById(req.params.id)
    res.status(200).json(user)
  } catch(err){
    next(err)
  }
}
//subcribe
export const Subscribe = async(req, res, next)=>{
    try{
      await User.findByIdAndUpdate(req.user.id,{
          $push: {subscribedUsers: req.params.id}
        }
      )
      await User.findByIdAndUpdate(req.params.id,{
        $inc: {subscribers: 1}
      })
      res.status(200).json("SubCription successfull")
    } catch(err){
      next(err)
    }
}
//unsubcriber
export const unSubscribe = async(req, res, next)=>{
  try{
    await User.findByIdAndUpdate(req.user.id,{
      $pull: {subscribedUsers: req.params.id}
    })
    await User.findByIdAndUpdate(req.params.id, {
      $inc: {subscribers: -1}
    })
    res.status(200).json("UnSubCription successfull")
  } catch(err){
    next(err)
  }
}

//like
  export const like = async(req, res, next)=>{
    const id = req.user.id
    const videoId = req.params.videoId
    try{
      await Video.findByIdAndUpdate(videoId, 
      {
        $addToSet: {likes: id}, //$addToSet để đảm bảo chi like một lần
        $pull: {dislikes: id},
      },
      {
        new: true,
      }
    )
      res.status(200).json("The video has been Liked")
    } catch(err){
      next(err)
    }
  }

//dislike
export const dislike = async(req, res, next)=>{
  const id = req.user.id
  const videoId = req.params.videoId
  try{
    await Video.findByIdAndUpdate(videoId, {
      $addToSet: {dislikes: id}, //$addToSet để đảm bảo chi like một lần
      $pull: {likes: id},
    })
    res.status(200).json("The video has been disliked")
  } catch(err){
    next(err)
  }
}