import React, { useEffect } from 'react';

function GridCanvasDisplay({ canvasRefs, size, gridLayout }) {

    console.log("gridLayout: ", gridLayout); // {rows: 2, cols: 2}

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

    const gridStyle = {
        display: 'grid',
        gridTemplateRows: `repeat(${gridLayout.rows}, ${size.h}px)`,
        gridTemplateColumns: `repeat(${gridLayout.cols}, ${size.w}px)`,
        gap: '0px',
    };

    return (
        <div style={gridStyle}>
            {canvasRefs.map((ref, i) => (
                <canvas
                    key={i}
                    ref={ref}
                    style={{ border: '1px solid white' }} // remove border if you want tighter packing
                />
            ))}
        </div>
    );
}

export default GridCanvasDisplay;
