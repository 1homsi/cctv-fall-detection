import React from 'react'
import ab from "../../assets/about3.svg"
import "./about.css"

function About() {
  return (
    <div className="ContainerAbout">
        <div class="custom-shape-divider-top-1680771511">
            <svg data-name="Layer 1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 120" preserveAspectRatio="none">
                <path d="M0,0V46.29c47.79,22.2,103.59,32.17,158,28,70.36-5.37,136.33-33.31,206.8-37.5C438.64,32.43,512.34,53.67,583,72.05c69.27,18,138.3,24.88,209.4,13.08,36.15-6,69.85-17.84,104.45-29.34C989.49,25,1113-14.29,1200,52.47V0Z" opacity=".25" class="shape-fill"></path>
                <path d="M0,0V15.81C13,36.92,27.64,56.86,47.69,72.05,99.41,111.27,165,111,224.58,91.58c31.15-10.15,60.09-26.07,89.67-39.8,40.92-19,84.73-46,130.83-49.67,36.26-2.85,70.9,9.42,98.6,31.56,31.77,25.39,62.32,62,103.63,73,40.44,10.79,81.35-6.69,119.13-24.28s75.16-39,116.92-43.05c59.73-5.85,113.28,22.88,168.9,38.84,30.2,8.66,59,6.17,87.09-7.5,22.43-10.89,48-26.93,60.65-49.24V0Z" opacity=".5" class="shape-fill"></path>
                <path d="M0,0V5.63C149.93,59,314.09,71.32,475.83,42.57c43-7.64,84.23-20.12,127.61-26.46,59-8.63,112.48,12.24,165.56,35.4C827.93,77.22,886,95.24,951.2,90c86.53-7,172.46-45.71,248.8-84.81V0Z" class="shape-fill"></path>
            </svg>
        </div>
        <div className='AboutUsText'>
            <h1>About Us</h1>
            <p>
                Welcome to WatchGuard! We are a cutting-edge technology company dedicated to developing innovative solutions for fall detection and emotion detection. Our mission is to enhance safety and well-being for individuals of all ages by providing advanced technology that can accurately detect falls and monitor emotional states in real-time. <br />
                At WatchGuard, we understand the importance of maintaining independence and quality of life as we age. Falls are a leading cause of injuries in seniors, often resulting in serious consequences such as fractures and hospitalizations. That's why we have developed state-of-the-art fall detection technology that utilizes artificial intelligence and machine learning algorithms to detect falls with high accuracy. Our fall detection system is designed to be non-intrusive and can be seamlessly integrated into various settings, including homes, healthcare facilities, and assisted living communities. <br />
                In addition to fall detection, we also specialize in emotion detection technology. We recognize that emotional well-being is a crucial aspect of overall health and happiness. Our emotion detection system uses sophisticated algorithms to analyze facial expressions, voice patterns, and physiological responses to accurately determine emotional states in real-time. This technology has wide-ranging applications, including mental health monitoring, mood tracking, and personalized emotional support.
                At WatchGuard, we are committed to excellence in innovation, reliability, and customer satisfaction. Our team of dedicated engineers and researchers are at the forefront of technological advancements in the field of fall detection and emotion detection. We strive to continually improve and expand our products and services to meet the evolving needs of our customers and partners.
            </p>
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