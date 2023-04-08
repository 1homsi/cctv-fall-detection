import "./styles.css";
import React from "react";
import "./landing.css"
import { Suspense, useState, useEffect } from "react";
import { motion, MotionConfig, useMotionValue } from "framer-motion";
import { Shapes } from "./Shapes";
import { transition } from "./settings";
import useMeasure from "react-use-measure";
import landing from "../../assets/landing4.svg"
import sec2 from "../../assets/faceD1.png"
import sec3 from "../../assets/fallD2.png"



export default function Landing({navigate}) {
  const [ref, bounds] = useMeasure({ scroll: false });
  const [isHover, setIsHover] = useState(false);
  const [isPress, setIsPress] = useState(false);
  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);

  const resetMousePosition = () => {
    mouseX.set(0);
    mouseY.set(0);
  };

    useEffect(() => {
        if (isPress) {
            navigate('/products')
        }        
    }, [isPress])



  return (
    <div className="ContainerLanding">
        <div className="BannerLanding" >
            <div className="landing-text">
                <h1>
                Always Watching, Always Protecting
                </h1>
                <h4>
                WatchGuard AI is an intelligent and reliable AI system that constantly monitors your surroundings to ensure your safety and protection. 
                </h4>
                <div className="startBTN" >
                    <button className="startBTN" onMouseEnter={() => setIsHover(true)} onMouseLeave={() => setIsHover(false)} onMouseDown={() => setIsPress(true)} onMouseUp={() => setIsPress(false)} >
                        Start
                    </button>
                       
                </div>
            </div>
            <div className="right" >
                <img src={landing} alt="landing" className="landing-img"/>
            </div>
            </div>
            <div className="sec2InLanding" >
                <div className="sec2InLanding-left" >
                    <img src={sec2} alt="sec2" className="sec2-img" />
                </div>
                <div className="sec2InLanding-right" >
                    <h1>Emotion Detection</h1>
                    <h4>WatchGuard AI is able to detect your emotions that can later be used to measure customer satisfaction etc... </h4>
                </div>
            </div>
            <div className="blackline"></div>
            <div className="sec2InLanding" >
                <div className="sec2InLanding-left" >
                <h1>Fall Detection</h1>
                    <h4>
                    WatchGuard AI is able to detect if customers have fallen and will save the time and date of the event
                       </h4>
                    
                </div>
                <div className="sec2InLanding-right" >
                <img src={sec3} alt="sec3" className="sec3-img" />
                </div>
            </div>
    </div>
  );
}
