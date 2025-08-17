import React, { forwardRef, useEffect, useRef } from 'react';

const CanvasDisplay = forwardRef(({ size, style }, ref) => {
    const internalRef = useRef(null);

    // If no ref is passed, fall back to internalRef (for flexible usage)
    const canvasRef = ref || internalRef;

    useEffect(() => {
        const canvas = canvasRef.current;
        if (canvas && size.w && size.h) {
            canvas.width = size.w;
            canvas.height = size.h;
            canvas.style.width = `${size.w}px`;
            canvas.style.height = `${size.h}px`;
        }
    }, [size]);

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
