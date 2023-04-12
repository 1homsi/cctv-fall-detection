import React, { useEffect, useRef } from 'react';

const NUM_COLUMNS = 80;
const NUM_ROWS = 30;
const NUM_VALUES = 10;
const TICK_MS = 50;
const COLORS = ['green', 'white', 'gray', 'lime', 'chartreuse', 'springgreen'];
const MOUSE_MOVE_SPEED = 0.3;
const MAX_DISTANCE = 100;

const Matrix = () => {
  const canvasRef = useRef(null);
  const valuesRef = useRef([]);
  const mouseRef = useRef({ x: 0, y: 0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const { width, height } = canvas;
    const columnWidth = width / NUM_COLUMNS;
    const rowHeight = height / NUM_ROWS;

    // Initialize the values
    for (let i = 0; i < NUM_COLUMNS; i++) {
      valuesRef.current[i] = new Array(NUM_ROWS).fill(0);
    }

    // Update the mouse position
    const handleMouseMove = (event) => {
      mouseRef.current.x = event.clientX;
      mouseRef.current.y = event.clientY;
    };
    document.addEventListener('mousemove', handleMouseMove);

    // Start the animation loop
    const intervalId = setInterval(() => {
      // Clear the canvas
      ctx.fillStyle = '#334443';
      ctx.fillRect(0, 0, width, height);

      // Move each value down one row
      for (let i = NUM_COLUMNS - 1; i >= 0; i--) {
        for (let j = NUM_ROWS - 1; j >= 1; j--) {
          valuesRef.current[i][j] = valuesRef.current[i][j - 1];
        }
      }

      // Add a new random value to the top row of each column
      for (let i = 0; i < NUM_COLUMNS; i++) {
        valuesRef.current[i][0] = Math.floor(Math.random() * NUM_VALUES);
      }

      // Draw the values, moving them away from the mouse
      for (let i = 0; i < NUM_COLUMNS; i++) {
        for (let j = 0; j < NUM_ROWS; j++) {
          const x = i * columnWidth;
          const y = j * rowHeight;
          const dx = x - mouseRef.current.x;
          const dy = y - mouseRef.current.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          if (distance < MAX_DISTANCE) {
            const directionX = dx / distance;
            const directionY = dy / distance;
            const speed = MOUSE_MOVE_SPEED * (MAX_DISTANCE - distance) / MAX_DISTANCE;
            const newX = x + directionX * speed;
            const newY = y + directionY * speed;
            ctx.fillStyle = COLORS[valuesRef.current[i][j]];
            ctx.fillText(valuesRef.current[i][j], newX, newY);
          }
        }
      }
    }, TICK_MS);

    // Stop the animation loop and remove the mouse listener when the component unmounts
    return () => {
      clearInterval(intervalId);
      document.removeEventListener('mousemove', handleMouseMove);
    };
  }, []);

  return (
    <div style={{
      position: 'absolute',
      top: 0,
      left: 0,
      width: '100%',
      height: '50%',
      overflow: 'hidden',
      zIndex: 0,
    }} >
    <canvas
      ref={canvasRef}
      width={window.innerWidth}
      height={window.innerHeight}
      style={{ background: 'black',}}
    />
    </div>
  );
};

export default Matrix;