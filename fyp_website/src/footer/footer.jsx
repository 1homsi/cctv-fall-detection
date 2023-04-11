import React from "react";
import "./footer.css"

const Footer = () => {
  return (
    <footer className="footer">
      <div className="container">
        <div className="row">
          <div className="col-sm-6">
            <h4>About WatchGuard AI</h4>
            <p>
              WatchGuard AI is an intelligent and reliable AI system that constantly monitors your surroundings to ensure your safety and protection.
            </p>
          </div>
          <div className="col-sm-6">
            <h4>Contact Us</h4>
            <p>Lebanon</p>
            <p>Rafik Hariri University</p>
            <p>Phone: 555-123-4567</p>
            <p>Email: info@watchguardai.com</p>
          </div>
        </div>
        <hr />
        <p>&copy; 2023 WatchGuard AI</p>
      </div>
    </footer>
  );
};

export default Footer;
