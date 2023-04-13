import React, { useState, useEffect } from "react";
import { color, motion, MotionConfig, useScroll } from "framer-motion";
import { useAnimatedText } from "./use-animated-text";
import "./products.css";
import { transition } from "./transition";
import { useRef } from "react";
import face from "../../assets/product1.png";
import fall from "../../assets/fall2.png";
import all from "../../assets/fallD2.png";
import ButtonGroup from "@mui/material/ButtonGroup";
import Button from "@mui/material/Button";

export default function Products() {
  const [isOn, setOn] = useState(true);
  const headerRef = useAnimatedText(isOn ? 8 : 9, transition);
  const toggleSwitch = () => setOn(!isOn);
  const [monthly, setBilled] = useState(true);

  const [screen, setScreen] = useState(1);
  const [selected, setSelected] = useState(0);

  useEffect(() => {
    // setIsSmallScreen(window.innerWidth >= 1100 ? false : true);
  }, []);

  return (
    <div className="mainProductsCont">
      <div className="buttonsProd">
        <button
          style={{
            backgroundColor: selected === 0 && "#fff",
            color: selected === 0 && "rgb(47, 45, 45)",
            border:selected === 0 && "rgb(47, 45, 45) solid 1.5px"
          }}

onClick={() => {
            setSelected(0);
          }}
        >
          Fall Detection
        </button>
        <button
          style={{
            backgroundColor: selected === 1 && "#fff",
            color: selected === 1 && "rgb(47, 45, 45)",
            border:selected === 1 && "rgb(47, 45, 45) solid 1.5px"
          }}
              onClick={() => {
                setSelected(1);
              }} 
        >
          Face Recognition
        </button>
        <button
          style={{
            backgroundColor: selected === 2 && "#fff",
            color: selected === 2 && "rgb(47, 45, 45)",
            border:selected === 2 && "rgb(47, 45, 45) solid 1.5px"
          }}
             onClick={() => {
              setSelected(2);
            }}
        >
          Sentinel
        </button>
      </div>
      {
        selected === 0 ? (

      <div className="productsInner">
        <div className="itemCards">
          <h1>1 month </h1>
          <h3>$99.99</h3>
          <button>Get Now!</button>
        </div>
        <div className="itemCards">
          <h1>3 months</h1>
          <h3>$249.99</h3>
          <button>Get Now!</button>
        </div>
        <div className="itemCards">
          <h1>1 year</h1>
          <h3>$1,099.99</h3>
          <button>Get Now!</button>
        </div>
      </div>
        ):
        selected === 1 ? (
          
      <div className="productsInner">
      <div className="itemCards">
        <h1>1 month</h1>
        <h3>$89.99</h3>
        <button>Get Now!</button>
      </div>
      <div className="itemCards">
        <h1>3 months</h1>
        <h3>$179.99</h3>
        <button>Get Now!</button>
      </div>
      <div className="itemCards">
        <h1>1 year</h1>
        <h3>$1,079.99</h3>
        <button>Get Now!</button>
      </div>
    </div>
        ):  selected === 2 &&  (
          
      <div className="productsInner">
      <div className="itemCards">
        <h1>1 month</h1>
        <h3>$199.99</h3>
        <button>Get Now!</button>
      </div>
      <div className="itemCards">
        <h1>3 months</h1>
        <h3>$399.99</h3>
        <button>Get Now!</button>
      </div>
      <div className="itemCards">
        <h1>1 year</h1>
        <h3>$2,399.99</h3>
        <button>Get Now!</button>
      </div>
    </div>
        )}
    </div>
  );
}
