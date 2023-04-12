import React from 'react';
import Anime from 'animejs/lib/anime.es.js';
import './about.css';
import Matrix from "../../components/timeline/matrix";

//team images
import teamMember1 from '../../assets/jalal.jpeg';
import teamMember2 from '../../assets/fawzy.jpeg';




const About = () => {

  // animate page elements on load
  const animateOnLoad = () => {
    Anime.timeline({loop: false})
      .add({
        targets: '.page-title',
        opacity: [0,1],
        translateY: [-50, 0],
        easing: "easeOutExpo",
        duration: 2000
      })
      .add({
        targets: '.who-we-are, .mission, .vision',
        opacity: [0,1],
        translateY: [50, 0],
        easing: "easeOutExpo",
        duration: 2000,
        delay: (el, i) => i * 250
      })
  }

  // call animation on page load
  React.useEffect(() => {
    animateOnLoad();
  }, []);

  return (
    <div className="about-page">
      <Matrix />
      <h1 className="page-title">WatchGuard AI</h1>
      <p className="who-we-are">We are WatchGuard AI, a leading provider of AI-powered CCTV cameras that help businesses and organizations detect and analyze human emotions in real-time. By using our cutting-edge technology, you can monitor customer satisfaction, reduce wait times, and improve overall customer experience.</p>
      <p className="mission">Our mission is to help businesses and organizations create safe and secure environments for their customers and employees. We believe that by using AI technology to analyze human behavior, we can prevent accidents and reduce crime, making public spaces safer for everyone.</p>
      <p className="vision">Our vision is to create a world where people feel safe and secure wherever they go. We believe that technology can be used for good, and we are committed to creating products that make a positive impact on society. We envision a future where AI technology is integrated seamlessly into public spaces, improving people's lives and making the world a better place.</p>
      <p className="contact-us">Contact us today to learn more about our AI-powered CCTV cameras and how they can benefit your business or organization.</p>
      <div className="team-section">
        <h2 className="team-title">Our Team</h2>
        <div className="team-member">
          <img src={teamMember1} alt="Team Member 1" />
          <h3 className="team-member-name">Jalal Ghannam</h3>
          <p className="team-member-position">Developer & Student at Rafik Hariri University</p>
          <p className="team-member-bio">Jalal Ghannam is a visionary entrepreneur with a passion for technology and innovation. He and his team founded WatchGuard AI with the goal of using AI technology to improve public safety and security.</p>
        </div>
        <div className="team-member">
          <img src={teamMember2} alt="Team Member 2" />
          <h3 className="team-member-name">Fawzy El Mozayen</h3>
          <p className="team-member-position">Developer & Student at Rafik Hariri University</p>
          <p className="team-member-bio">Fawzy El Mozayen is not only a developer but oversees WatchGuard AI's day-to-day operations and ensures that the company runs smoothly.</p>
        </div>
        <div className="team-member">
          <img src="/images/team-member-3.png" alt="Team Member 3" />
          <h3 className="team-member-name">David Lee</h3>
          <p className="team-member-position">Chief Technology Officer</p>
          <p className="team-member-bio">David Lee is a highly skilled engineer with expertise in AI and computer vision. He leads WatchGuard AI's technology development and ensures that the company stays at the forefront of the industry.</p>
        </div>
        <div className="team-member">
          <img src="/images/team-member-4.png" alt="Team Member 4" />
          <h3 className="team-member-name">Lisa Chen</h3>
          <p className="team-member-position">Chief Marketing Officer</p>
          <p className="team-member-bio">Lisa Chen is a marketing expert with a passion for driving business growth. She oversees WatchGuard AI's marketing strategy and helps to communicate the value of our products to customers.</p>
        </div>
        <div className="team-member">
          <img src="/images/team-member-4.png" alt="Team Member 4" />
          <h3 className="team-member-name">Lisa Chen</h3>
          <p className="team-member-position">Chief Marketing Officer</p>
          <p className="team-member-bio">Lisa Chen is a marketing expert with a passion for driving business growth. She oversees WatchGuard AI's marketing strategy and helps to communicate the value of our products to customers.</p>
        </div>
      </div>
    </div>
  );
}
export default About;