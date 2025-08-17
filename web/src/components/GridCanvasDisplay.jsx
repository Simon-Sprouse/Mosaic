import React, { useEffect } from 'react';

function GridCanvasDisplay({ canvasRefs, size, gridLayout }) {


    // note gridLayout is not being implemented yet, this is OK !

  useEffect(() => {
    canvasRefs.forEach(ref => {
      if (ref.current) {
        ref.current.width = size.w;
        ref.current.height = size.h;
        ref.current.style.width = `${size.w}px`;
        ref.current.style.height = `${size.h}px`;
      }
    });
  }, [canvasRefs, size]);

  return (
    <>
      {canvasRefs.map((ref, i) => (
        <canvas
          key={i}
          ref={ref}
          style={{ border: '1px solid white', margin: '5px' }}
        />
      ))}
    </>
  );
}

export default GridCanvasDisplay;
