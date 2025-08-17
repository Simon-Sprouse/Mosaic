import React, { forwardRef, useEffect, useRef } from 'react';

const CanvasDisplay = forwardRef(({ width, height, style }, ref) => {
    const internalRef = useRef(null);

    // If no ref is passed, fall back to internalRef (for flexible usage)
    const canvasRef = ref || internalRef;

    useEffect(() => {
        const canvas = canvasRef.current;
        if (canvas && width && height) {
            canvas.width = width;
            canvas.height = height;
            canvas.style.width = `${width}px`;
            canvas.style.height = `${height}px`;
        }
    }, [width, height]);

    return (
        <canvas
            ref={canvasRef}
            style={{
                display: 'block',
                backgroundColor: '#000',
                ...style,
            }}
        />
    );
});

export default CanvasDisplay;
