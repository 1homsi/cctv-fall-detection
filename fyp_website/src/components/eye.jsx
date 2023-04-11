import React, { useRef } from 'react';
import anime from 'animejs';
import { InView } from 'react-intersection-observer';

const EyeAnimation = () => {
  const eyeRef = useRef(null);
  const pupilRef = useRef(null);

  const handleIntersection = (inView) => {
    if (inView) {
      anime({
        targets: pupilRef.current,
        translateX: '+=30',
        translateY: '+=30',
        duration: 500
      });

      anime({
        targets: eyeRef.current,
        rotate: 5,
        duration: 500,
        easing: 'easeInOutQuad'
      });

      // add mousemove event listener to parent div of eye
      eyeRef.current.parentElement.addEventListener('mousemove', handleMouseMove);
    } else {
      anime({
        targets: [pupilRef.current, eyeRef.current],
        translateX: 0,
        translateY: 0,
        rotate: 0,
        duration: 500,
        easing: 'easeInOutQuad'
      });

      // remove mousemove event listener when eye is out of view
      eyeRef.current.parentElement.removeEventListener('mousemove', handleMouseMove);
    }
  };

  const handleMouseMove = (e) => {
    const eyeRect = eyeRef.current.getBoundingClientRect();
    const pupilRect = pupilRef.current.getBoundingClientRect();

    // calculate position of cursor relative to center of eye
    const deltaX = e.clientX - (eyeRect.left + eyeRect.width / 2);
    const deltaY = e.clientY - (eyeRect.top + eyeRect.height / 2);

    // limit maximum displacement of pupil
    const maxDelta = Math.min(eyeRect.width, eyeRect.height) / 4;
    const displacement = Math.min(Math.sqrt(deltaX ** 2 + deltaY ** 2), maxDelta);

    // calculate angle of displacement relative to center of eye
    const angle = Math.atan2(deltaY, deltaX);

    // calculate new position of pupil
    const newX = eyeRect.left + eyeRect.width / 2 + displacement * Math.cos(angle) - pupilRect.width / 2;
    const newY = eyeRect.top + eyeRect.height / 2 + displacement * Math.sin(angle) - pupilRect.height / 2;

    // update position of pupil
    anime({
      targets: pupilRef.current,
      translateX: newX - pupilRect.left,
      translateY: newY - pupilRect.top,
      duration: 50,
      easing: 'linear'
    });
  };

  return (
    <InView onChange={handleIntersection}>
      <div style={{ width: '100%', height: '20vh', position: 'relative' }}>
        <div
          ref={eyeRef}
          style={{
            width: '150px',
            height: '150px',
            borderRadius: '50%',
            backgroundColor: '#fff',
            position: 'absolute',
            top: 'calc(50% - 75px)',
            left: 'calc(50% - 75px)',
            boxShadow: '0 5px 10px rgba(0, 0, 0, 0.1)'
          }}
        >
          <div
            ref={pupilRef}
            style={{
              width: '50px',
              height: '50px',
              borderRadius: '50%',
              backgroundColor: '#000',
              position: 'absolute',
              top: 'calc(50% - 25px)',
              left: 'calc(50% - 25px)',
              boxShadow: '0 2px 5px rgba(0, 0, 0, 0.1)'
            }}
          />
        </div>
      </div>
    </InView>
    );
};

export default EyeAnimation;
    
