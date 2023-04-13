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
          <h3>$100</h3>
          <button>Get Now!</button>
        </div>
        <div className="itemCards">
          <h1>3 months</h1>
          <h3>$250</h3>
          <button>Get Now!</button>
        </div>
        <div className="itemCards">
          <h1>1 year</h1>
          <h3>$1000</h3>
          <button>Get Now!</button>
        </div>
      </div>
        ):
        selected === 1 ? (
          
      <div className="productsInner">
      <div className="itemCards">
        <h1>temp </h1>
        <p>
          Imagine having an AI that watches over you, with every step you
          take, every move you make. This AI is always on guard, scanning your
          movements for signs of imbalance or instability. It's your guardian
          angel, ready to leap into action at a moment's notice if you start
          to fall. With advanced sensors and algorithms, this AI can deploy
          safety features, call for emergency assistance, and even analyze
          your movements to identify areas where you may be at risk of
          falling. It's like having a superhero by your side, always watching
          over you and protecting you from harm.
        </p>
        <h3>{monthly ? "100$ / month" : "1150$ / year"}</h3>
        <button>Get Now!</button>
      </div>
      <div className="itemCards">
        <h1>Face Recognition</h1>
        <p>
          Imagine having an AI that can read people's emotions with remarkable
          accuracy, like a mind-reading machine. This AI is designed to detect
          and analyze the emotional state of customers, allowing businesses to
          measure their satisfaction levels and respond proactively to their
          needs. With its advanced algorithms and machine learning
          capabilities, this emotion detection AI can pick up on subtle cues
          like facial expressions, tone of voice, and body language, providing
          real-time feedback to help businesses improve their customer
          service. It's like having a secret weapon for customer satisfaction,
          giving businesses the power to connect with their customers on a
          deeper level and build long-lasting relationships.
        </p>
        <h3>{monthly ? "100$ / month" : "1150$ / year"}</h3>
        <button>Get Now!</button>
      </div>
      <div className="itemCards">
        <h1>The Best</h1>
        <p>
          This advanced AI not only protects you from falls but also detects
          your emotional state to gauge your satisfaction as a customer. It
          can pick up on subtle cues like facial expressions, tone of voice,
          and body language, providing businesses with real-time feedback to
          help them improve customer service. It's like having a personal
          guardian angel and a secret weapon for customer satisfaction rolled
          into one.
        </p>
        <h3>{monthly ? "150$ / month" : "1800$ / year"}</h3>
        <button>Get Now!</button>
      </div>
    </div>
        ):  selected === 2 &&  (
          
      <div className="productsInner">
      <div className="itemCards">
        <h1>fuck </h1>
        <p>
          Imagine having an AI that watches over you, with every step you
          take, every move you make. This AI is always on guard, scanning your
          movements for signs of imbalance or instability. It's your guardian
          angel, ready to leap into action at a moment's notice if you start
          to fall. With advanced sensors and algorithms, this AI can deploy
          safety features, call for emergency assistance, and even analyze
          your movements to identify areas where you may be at risk of
          falling. It's like having a superhero by your side, always watching
          over you and protecting you from harm.
        </p>
        <h3>{monthly ? "100$ / month" : "1150$ / year"}</h3>
        <button>Get Now!</button>
      </div>
      <div className="itemCards">
        <h1>Face Recognition</h1>
        <p>
          Imagine having an AI that can read people's emotions with remarkable
          accuracy, like a mind-reading machine. This AI is designed to detect
          and analyze the emotional state of customers, allowing businesses to
          measure their satisfaction levels and respond proactively to their
          needs. With its advanced algorithms and machine learning
          capabilities, this emotion detection AI can pick up on subtle cues
          like facial expressions, tone of voice, and body language, providing
          real-time feedback to help businesses improve their customer
          service. It's like having a secret weapon for customer satisfaction,
          giving businesses the power to connect with their customers on a
          deeper level and build long-lasting relationships.
        </p>
        <h3>{monthly ? "100$ / month" : "1150$ / year"}</h3>
        <button>Get Now!</button>
      </div>
      <div className="itemCards">
        <h1>Fuck Fukc</h1>
        <p>
          This advanced AI not only protects you from falls but also detects
          your emotional state to gauge your satisfaction as a customer. It
          can pick up on subtle cues like facial expressions, tone of voice,
          and body language, providing businesses with real-time feedback to
          help them improve customer service. It's like having a personal
          guardian angel and a secret weapon for customer satisfaction rolled
          into one.
        </p>
        <h3>{monthly ? "150$ / month" : "1800$ / year"}</h3>
        <button>Get Now!</button>
      </div>
    </div>
        )}
    </div>
  );
}
