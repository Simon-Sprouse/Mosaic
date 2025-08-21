import React, { useEffect, useState } from 'react';

function GridCanvasDisplay({ canvasRefs, size, gridLayout }) {

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

    const canvasWrapperStyle = {
        position: 'relative',
        width: `${size.w}px`,
        height: `${size.h}px`,
    };

    const overlayStyle = {
        position: 'absolute',
        bottom: 0,
        right: 0,
        background: 'rgba(0, 0, 0, 0.6)',
        color: 'white',
        padding: '2px 6px',
        fontSize: '20px',
        cursor: 'pointer',
        zIndex: 1,
        display: 'none',
    };

    const wrapperHoverStyle = {
        ...overlayStyle,
        display: 'block',
    };

    const saveCanvasAsImage = (ref, index) => {
        if (!ref.current) return;
        const link = document.createElement('a');
        link.download = `canvas-${index}.png`;
        link.href = ref.current.toDataURL();
        link.click();
    };

    return (
        <div style={gridStyle}>
            {canvasRefs.map((ref, i) => (
                <div
                    key={i}
                    style={canvasWrapperStyle}
                    onMouseEnter={e => {
                        const overlay = e.currentTarget.querySelector('.save-overlay');
                        if (overlay) overlay.style.display = 'block';
                    }}
                    onMouseLeave={e => {
                        const overlay = e.currentTarget.querySelector('.save-overlay');
                        if (overlay) overlay.style.display = 'none';
                    }}
                >
                    <canvas
                        ref={ref}
                        style={{ display: 'block' }}
                    />
                    <div
                        className="save-overlay"
                        style={overlayStyle}
                        onClick={() => saveCanvasAsImage(ref, i)}
                    >
                        Save
                    </div>
                </div>
            ))}
        </div>
    );
}

export default GridCanvasDisplay;
