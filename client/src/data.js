export const API_KEY = 'AIzaSyByKtk-vj4Lkx-3bDZ2mVBNxFLiShi2pG8';

export const value_converter = (value)=>{
    if(value>=1000000000){
        return (Math.floor(value/1000000000*10)/10+"T").toString().replace('.', ',');
    }
    else if(value>=1000000){
        return (Math.floor(value/1000000*10)/10+"Tr").toString().replace('.', ',');
    }
    else if(value>=1000){
        return (Math.floor(value/1000*10)/10+"K").toString().replace('.', ',');
    }
    else{
        return value;
    }
}