import React from "react";
import "./footer.css"

const Footer = () => {
  return (
    <footer className="footer">
      <div className="container">
        <div className="row">
          <div className="col-sm-6">
            <h4>About Sentinel watchgaurd</h4>
            <p>
              WatchGuard is an intelligent and reliable system that constantly monitors your surroundings to ensure your safety and protection.
            </p>
          </div>
          <div className="col-sm-6">
            <p>Email: info@watchguard.com</p>
          </div>
        </div>
        <hr />
        <p>&copy; 2023 WatchGuard</p>
      </div>
    </footer>
  );
};

export default Footer;
