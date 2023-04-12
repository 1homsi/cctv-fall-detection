import "./styles.css";
import React from "react";
import "./landing.css"
import * as THREE from 'three';
import EyeAnimation from "../../components/eye"
// import Matrix from "../../components/timeline/matrix";

// stuff 
import { Suspense, useState, useEffect, useRef } from "react";
import { motion, MotionConfig, useMotionValue } from "framer-motion";
import { Shapes } from "./Shapes";
import { transition } from "./settings";
import useMeasure from "react-use-measure";

// for the videos and images
import landing from "../../assets/landing4.svg"
import sec2 from "../../assets/faceD1.png"
import sec3 from "../../assets/fallD2.png"
import why from "../../assets/why.jpeg"
import vidbanner from "../../assets/vidbanner.mp4"
import world from "../../assets/world.jpeg"

// for the typing animation
import { TypeAnimation } from 'react-type-animation';

// for the parallax effect
import Tilt from 'react-parallax-tilt';





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
            <div className="BannerLanding">
            <video src={vidbanner} autoPlay loop muted></video>
                <div className="landing-text">
                    <TypeAnimation
                        sequence={[
                        'Always Watching',
                        1500,
                        'Always Protecting',
                        1500,
                        ]}
                        speed={50} // Custom Speed from 1-99 - Default Speed: 40
                        style={{ fontSize: '3em' }}
                        wrapper="h1" // Animation will be rendered as a <span>
                        repeat={Infinity} // Repeat this Animation Sequence infinitely
                    />
                    <h4>
                    Sentinel watchgaurd is an intelligent and reliable system that constantly monitors your surroundings to ensure your safety and protection. 
                    </h4>
                    
                    <div className="startBTN">
                    <button className="startBTN" onMouseEnter={() => setIsHover(true)} onMouseLeave={() => setIsHover(false)} onMouseDown={() => setIsPress(true)} onMouseUp={() => setIsPress(false)}>
                        Start
                    </button>
                    </div>
                </div>
            </div>
            <div className="sec2InLanding" >
                <div className="sec2InLanding-left" >
                <Tilt
                    
                    tiltMaxAngleX={10}
                    tiltMaxAngleY={10}
                    perspective={1000}
                    transitionSpeed={2000}
                    scale={1.1}
                    gyroscope={true}
                    gyroscopeMinAngleX={-10}
                    gyroscopeMaxAngleX={10}
                    gyroscopeMinAngleY={-10}
                    gyroscopeMaxAngleY={10}
                    >
                    <img src={sec2} alt="sec2" className="sec2-img" />
                    </Tilt>
                </div>
                <div className="sec2InLanding-right" >
                    <h1>Emotion Detection</h1>
                    <h4 style={{fontWeight: "400"}} >
                    Sentinel watchgaurd's emotion detection is an advanced artificial intelligence system that uses cutting-edge technology to accurately detect and interpret human emotions. With its powerful algorithms and advanced machine learning capabilities, Sentinel watchgaurd is capable of identifying a wide range of emotions, including joy, sadness, anger, fear, and more. Whether you are looking to improve your customer service, enhance your marketing campaigns, or simply gain a better understanding of your audience, Sentinel watchgaurd can provide you with the insights you need to succeed. So why wait? Try Sentinel watchgaurd's emotion detection today and start unlocking the power of emotional intelligence!
                    </h4>
                </div>
            </div>
            <div className="blackline"></div>
            <div className="sec2InLanding"
            style={{
                backgroundColor: "#636262",
                color: "#fff"
            }}
            >
                <div className="sec2InLanding-left" >
                <h1>Fall Detection</h1>
                    <h4 style={{fontWeight: "400"}} >
                    Sentinel watchgaurd's Fall Detection is an advanced artificial intelligence system that is designed to detect and alert caregivers and staff to falls that have already occurred. Using sophisticated algorithms and machine learning techniques, it can analyze movements and behaviors in real-time and alert the appropriate personnel if a fall is detected. While it does not prevent falls from happening, Sentinel watchgaurd's Fall Detection can provide a critical response in the event of a fall, allowing for timely assistance and potentially minimizing the severity of injuries. Whether you're a caregiver or a staff member in a public space, Sentinel watchgaurd's Fall Detection can provide an added layer of safety and security for those who may be at risk of falling.
                    </h4>
                    
                </div>
                <div className="sec2InLanding-right" >
                <Tilt
                    
                    tiltMaxAngleX={10}
                    tiltMaxAngleY={10}
                    perspective={1000}
                    transitionSpeed={2000}
                    scale={1.1}
                    gyroscope={true}
                    gyroscopeMinAngleX={-10}
                    gyroscopeMaxAngleX={10}
                    gyroscopeMinAngleY={-10}
                    gyroscopeMaxAngleY={10}
                    >
                <img src={sec3} alt="sec3" className="sec3-img" />
                </Tilt>
                </div>
            </div>

            <div>
                {/* <EyeAnimation />
                <Matrix /> */}
            </div>
            <div className="blackline"></div>
            <div className="sec2InLanding" >
                <div className="sec2InLanding-left" >
                <Tilt
                    
                    tiltMaxAngleX={10}
                    tiltMaxAngleY={10}
                    perspective={1000}
                    transitionSpeed={2000}
                    scale={1.1}
                    gyroscope={true}
                    gyroscopeMinAngleX={-10}
                    gyroscopeMaxAngleX={10}
                    gyroscopeMinAngleY={-10}
                    gyroscopeMaxAngleY={10}
                    >
                    <img src={why} alt="sec2" className="sec2-img" />
                    </Tilt>
                </div>
                <div className="sec2InLanding-right" >
                    <h1>Why Sentinel watchgaurd?</h1>
                    <h4 style={{fontWeight: "400"}} >
                    Sentinel watchgaurd is the ideal solution for public areas such as malls and schools, where safety and security are of the utmost importance. Our advanced technology and sophisticated algorithms are designed to quickly and accurately detect falls and other incidents, allowing you to respond quickly and effectively in the event of an emergency. With Sentinel watchgaurd, you can monitor public spaces with confidence, knowing that you have the tools you need to keep people safe and secure. Plus, our emotion detection technology can help you better understand the needs and preferences of your customers or students, allowing you to provide the best possible service and support. So whether you're looking to improve safety and security, enhance customer or student experience, or simply gain a better understanding of your audience, Sentinel watchgaurd is the smart choice for public areas.
                    </h4>
                </div>
            </div>
    </div>
  );
}
