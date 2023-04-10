import React from 'react'
import ab from "../../assets/about3.svg"
import "./about.css"

function About() {
  return (
    <div className="ContainerAbout">
        <div className='AboutUsContainer'>
            <div className='AboutUsOne'>
                <img src="https://cdn.dribbble.com/users/4107928/screenshots/16332316/computer_science_4x.jpg" alt="about us image" />
            </div>
            <div className='AboutUsTwo'>
                <h1>About Us</h1>
                <p>
                    Welcome to WatchGuard! We are a cutting-edge technology company dedicated to developing innovative solutions for fall detection and emotion detection. <br/>
                    At WatchGuard, we understand the importance of maintaining independence and quality of life as we age. That's why we have developed state-of-the-art fall detection technology that utilizes artificial intelligence and machine learning algorithms to detect falls with high accuracy. We are committed to excellence in innovation, reliability, and customer satisfaction. Our team of dedicated engineers and researchers are at the forefront of technological advancements in the field of fall detection and emotion detection. We strive to continually improve and expand our products and services to meet the evolving needs of our customers and partners <br />
                    <b>Mission:</b> Enhance safety and well-being for individuals of all ages by providing advanced technology that can accurately detect falls and monitor emotional states in real-time. <br />
                </p>
            </div>
            {/* <h1>About Us</h1>
            <p>
                Welcome to WatchGuard! We are a cutting-edge technology company dedicated to developing innovative solutions for fall detection and emotion detection. Our mission is to enhance safety and well-being for individuals of all ages by providing advanced technology that can accurately detect falls and monitor emotional states in real-time. <br />
                At WatchGuard, we understand the importance of maintaining independence and quality of life as we age. Falls are a leading cause of injuries in seniors, often resulting in serious consequences such as fractures and hospitalizations. That's why we have developed state-of-the-art fall detection technology that utilizes artificial intelligence and machine learning algorithms to detect falls with high accuracy. Our fall detection system is designed to be non-intrusive and can be seamlessly integrated into various settings, including homes, healthcare facilities, and assisted living communities. <br />
                In addition to fall detection, we also specialize in emotion detection technology. We recognize that emotional well-being is a crucial aspect of overall health and happiness. Our emotion detection system uses sophisticated algorithms to analyze facial expressions, voice patterns, and physiological responses to accurately determine emotional states in real-time. This technology has wide-ranging applications, including mental health monitoring, mood tracking, and personalized emotional support.
                At WatchGuard, we are committed to excellence in innovation, reliability, and customer satisfaction. Our team of dedicated engineers and researchers are at the forefront of technological advancements in the field of fall detection and emotion detection. We strive to continually improve and expand our products and services to meet the evolving needs of our customers and partners.
            </p> */}
        </div>
        {/* <div className="ContainerAbout-image">
            <div className="ContainerAbout-text">
                <h1>About Us</h1>
                <h1>Watch Guard AI</h1>
=                <p>
                    Welcome to FallSense! We are a cutting-edge technology company dedicated to developing innovative solutions for fall detection and emotion detection. Our mission is to enhance safety and well-being for individuals of all ages by providing advanced technology that can accurately detect falls and monitor emotional states in real-time.
                </p>
            </div>
            <div className="ContainerAbout-right">
                <img src={ab}
                alt=""
                style={{
                    width: "70%",
                    height: "70%",
                }}
                />
            </div>
        </div> */}
    </div>
  )
}

export default About