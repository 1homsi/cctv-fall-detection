import React, { useState, useEffect } from "react"
import { motion, MotionConfig, useScroll } from "framer-motion"
import { useAnimatedText } from "./use-animated-text"
import "./products.css"
import { transition } from "./transition"
import { useRef } from "react";
import face from "../../assets/product1.png"
import fall from "../../assets/fall2.png"
import all from "../../assets/fallD2.png"
import ButtonGroup from '@mui/material/ButtonGroup';
import Button from '@mui/material/Button';


export default function Products() {
  const [isOn, setOn] = useState(true)
  const headerRef = useAnimatedText(isOn ? 8 : 9, transition)
  const toggleSwitch = () => setOn(!isOn);
  const [monthly, setBilled] = useState(false)

  const [screen, setScreen] = useState(1)


  const [isSmallScreen, setIsSmallScreen] = useState(false);

  useEffect(() => {
    setIsSmallScreen(window.innerWidth >= 1100 ? false : true);
  }, []);

  const contentBasedOnScreen =()=>{
    if(screen === 1){
      return(
        <div className="itemInMobile">
          <h1>
            Fall Detection
          </h1>
          <div className="blackline"></div>
          <p>An AI can detect and analyze customer emotions, helping businesses to measure satisfaction and improve customer service using facial expressions, tone of voice, and body language.</p>
          <div className="blackline"></div>
          <h3>{monthly ? "100$ / month" : "1150$ / year"}</h3>
          <button>
            Get Now!
          </button>

            <img src={fall} alt=""  
             
             style={{
              width: "80%",
              height: "300px",
              borderRadius: "20px",
             }}
             
             />

        </div>
      )
    }
    if(screen === 2){
      return(
        <div className="itemInMobile">
          <h1>
            Emotion Detection
          </h1>
          <div className="blackline"></div>
          <p>Imagine having an AI that can read people's emotions with remarkable accuracy, like a mind-reading machine. This AI is designed to detect and analyze the emotional state of customers, allowing businesses to measure their satisfaction levels and respond proactively to their needs. With its advanced algorithms and machine learning capabilities, this emotion detection AI can pick up on subtle cues like facial expressions, tone of voice, and body language, providing real-time feedback to help businesses improve their customer service. It's like having a secret weapon for customer satisfaction, giving businesses the power to connect with their customers on a deeper level and build long-lasting relationships.</p>
          <div className="blackline"></div>
          <h3>{monthly ? "150$ / month" : "1700$ / year"}</h3>
          <button>
            Get Now!
          </button>
          <img src={face} alt=""  
             
             style={{
              width: "80%",
              height: "300px",
              borderRadius: "20px",
             }}
             
             />
          
        </div>
      )
    }
    if(screen === 3){
      return(
        <div className="itemInMobile">
          <h1>
            The Best
          </h1>
          <div className="blackline"></div>
          <p>This advanced AI not only protects you from falls but also detects your emotional state to gauge your satisfaction as a customer. It can pick up on subtle cues like facial expressions, tone of voice, and body language, providing businesses with real-time feedback to help them improve customer service. It's like having a personal guardian angel and a secret weapon for customer satisfaction rolled into one.</p>
          <div className="blackline"></div>
          <h3>{monthly ? "230$ / month" : "2700$ / year"}</h3>
          <button>
            Get Now!
          </button>

          <img src={all} alt=""  
             
             style={{
              width: "80%",
              height: "300px",
              borderRadius: "20px",
             }}
             
             />
        </div>
      )
    }
  }



  return (
    !isSmallScreen ? 
    <div className="mainProduct" >
    <MotionConfig transition={transition}>
      <motion.div
        className="containerProducts"
        initial={false}
        animate={{
          backgroundColor: isOn ? "#c9ffed" : "#ff2558",
          color: isOn ? "#7fffd4" : "#c70f46"
        }}
      >
        <h1 className="open" children="<AI>" />
        <h1 className="close" children="</AI>" />

        <motion.h1 ref={headerRef} />
        <div className="productsContent">
            <h2>Switch to {isOn? "Pro?" : "Basic?"}</h2>
          <div className="switch" data-isOn={!isOn} onClick={toggleSwitch}>
            <motion.div className="handle" layout transition={spring} />
            
          </div>
         
          <div style={{
            margin:"1%",
          }} >
                  <ButtonGroup
              disableElevation
              variant="contained"
              aria-label="Disabled elevation buttons"
              
            >
              <Button style={{
                backgroundColor: "#333",
              }} 
              
              onClick={() => setBilled(true)}
              >Monthly</Button>
              <Button style={{
                backgroundColor: "#333",
              }} 
              onClick={() => setBilled(false)}
              >Yearly</Button>
            </ButtonGroup>
          </div>

          { isOn ?
          <div className="displayProductsNormal" >
            <div className="product">
              <div className="product-right" >
                <h3>Fall Detection</h3>
                <div className="blackline">{".     "}</div>
                <p>Imagine having an AI that watches over you, with every step you take, every move you make. This AI is always on guard, scanning your movements for signs of imbalance or instability. It's your guardian angel, ready to leap into action at a moment's notice if you start to fall. With advanced sensors and algorithms, this AI can deploy safety features, call for emergency assistance, and even analyze your movements to identify areas where you may be at risk of falling. It's like having a superhero by your side, always watching over you and protecting you from harm.</p>
                <div className="buy">
                  <h3>{monthly ? "100$ / month" : "1150$ / year"}</h3>
                  <button>
                    Get Now!
                  </button>
                </div>
              </div>
              <div className="product-left">
                <img src={fall} alt="fall-detection" border="0" />
              </div>
            </div>
            <div className="product">
              <div className="product-right">
                  <h3>Emotion Detection</h3>
                  <div className="blackline">{".     "}</div>
                  <p>Imagine having an AI that can read people's emotions with remarkable accuracy, like a mind-reading machine. This AI is designed to detect and analyze the emotional state of customers, allowing businesses to measure their satisfaction levels and respond proactively to their needs. With its advanced algorithms and machine learning capabilities, this emotion detection AI can pick up on subtle cues like facial expressions, tone of voice, and body language, providing real-time feedback to help businesses improve their customer service. It's like having a secret weapon for customer satisfaction, giving businesses the power to connect with their customers on a deeper level and build long-lasting relationships.</p>
                  <div className="buy">
                    <h3>{monthly ? "150$ / month" : "1700$ / year"}</h3>
                    <button>
                      Get Now!
                    </button>
                  </div>
                </div>
                <div className="product-left">
                <img src={face} alt="" />
              </div>
              </div>
              
          </div>
          :
          <div className="displayProductsPro" >
            <div className="product">
              <div className="product-right">
                  <h3>The Best</h3>
                  <div className="blackline"></div>
                  <p>This advanced AI not only protects you from falls but also detects your emotional state to gauge your satisfaction as a customer. It can pick up on subtle cues like facial expressions, tone of voice, and body language, providing businesses with real-time feedback to help them improve customer service. It's like having a personal guardian angel and a secret weapon for customer satisfaction rolled into one.</p>
                  <div className="buy">
                    <h3>{monthly ? "230$ / month" : "2700$ / year"}</h3>
                    <button>
                      Get Now!
                    </button>
                  </div>
              </div>
              <div className="product-left">
                <img src={fall} alt="" /> 
              </div>
            </div>

          </div>
          }
        </div>
      </motion.div>
    </MotionConfig>
    </div>
    :
    <div className="mobileProducts">
      <div className="miniNav">
        <h4
        onClick={()=>{
          setScreen(1)
        }}
        >Fall</h4>
        <h4
        onClick={()=>{
          setScreen(2)
        }}
        >Emotion</h4>
        <h4
        onClick={()=>{
          setScreen(3)
        }}
        >Both</h4>
      </div>
      <div style={{
            margin:"1%",
          }} >
                  <ButtonGroup
              disableElevation
              variant="contained"
              aria-label="Disabled elevation buttons"
              
            >
              <Button style={{
                backgroundColor: "#333",
              }} 
              
              onClick={() => setBilled(true)}
              >Monthly</Button>
              <Button style={{
                backgroundColor: "#333",
              }} 
              onClick={() => setBilled(false)}
              >Yearly</Button>
            </ButtonGroup>
          </div>
      {contentBasedOnScreen()}
    </div>
  )
}

const spring = {
  type: "spring",
  stiffness: 700,
  damping: 30
};