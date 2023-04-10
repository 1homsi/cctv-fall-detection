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
                WatchGuard is an intelligent and reliable system that constantly monitors your surroundings to ensure your safety and protection. 
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
                    <h4 style={{fontWeight: "400"}} >
                    WatchGuard emotion detection is an advanced artificial intelligence system that uses cutting-edge technology to accurately detect and interpret human emotions. With its powerful algorithms and advanced machine learning capabilities, WatchGuard AI is capable of identifying a wide range of emotions, including joy, sadness, anger, fear, and more. Whether you are looking to improve your customer service, enhance your marketing campaigns, or simply gain a better understanding of your audience, WatchGuard AI can provide you with the insights you need to succeed. So why wait? Try WatchGuard AI emotion detection today and start unlocking the power of emotional intelligence!
                    </h4>
                </div>
            </div>
            <div className="blackline"></div>
            <div className="sec2InLanding" >
                <div className="sec2InLanding-left" >
                <h1>Fall Detection</h1>
                    <h4 style={{fontWeight: "400"}} >
                    WatchGuard Fall Detection is an advanced artificial intelligence system that is designed to detect and alert caregivers and staff to falls that have already occurred. Using sophisticated algorithms and machine learning techniques, it can analyze movements and behaviors in real-time and alert the appropriate personnel if a fall is detected. While it does not prevent falls from happening, WatchGuard AI Fall Detection can provide a critical response in the event of a fall, allowing for timely assistance and potentially minimizing the severity of injuries. Whether you're a caregiver or a staff member in a public space, WatchGuard AI Fall Detection can provide an added layer of safety and security for those who may be at risk of falling.
                    </h4>
                    
                </div>
                <div className="sec2InLanding-right" >
                <img src={sec3} alt="sec3" className="sec3-img" />
                </div>
            </div>
    </div>
  );
}
