import React, { useState, useEffect, useRef } from 'react';

function SliderInput({ label, value, min, max, step, onChange }) {
  const [inputValue, setInputValue] = useState(value.toString());
  const sliderRef = useRef(null);

  // Sync external changes into the input box
  useEffect(() => {
    setInputValue(value.toString());
  }, [value]);

//   useEffect(() => {
//     if (sliderRef.current) {
//       const percent = ((value - min) / (max - min)) * 100;
//       sliderRef.current.style.setProperty('--progress', `${percent}%`);
//     }
//   }, [value, min, max]);

  // On blur, correct or reset input if needed
  const handleBlur = () => {
    const num = parseFloat(inputValue);
    if (isNaN(num)) {
      setInputValue(value.toString()); // revert
    } else {
      // clamp to range
      const clamped = Math.max(min, Math.min(max, num));
      onChange(clamped);
      setInputValue(clamped.toString());
    }
  };

  return (
    <div className="slider-control">
      <div className="slider-header">
        <label className="slider-label">{label}</label>
        <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onBlur={handleBlur}
            onKeyDown={(e) => {
                if (e.key === 'Enter') {
                e.target.blur();  // Triggers onBlur immediately
                }
            }}
        />
      </div>
      <div className="slider-track">
        {/* <span className="slider-min">{min}</span> */}
        <input
          className="slider-input"
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          style={{ '--progress': `${((value - min) / (max - min)) * 100}%` }}
          onChange={(e) => {
            const val = parseFloat(e.target.value);
            const minVal = parseFloat(e.target.min);
            const maxVal = parseFloat(e.target.max);
            const percent = ((val - minVal) / (maxVal - minVal)) * 100;
            e.target.style.setProperty('--progress', `${percent}%`);
            onChange(val);
          }}
        />
        {/* <span className="slider-max">{max}</span> */}
      </div>
    </div>
  );
}

export default SliderInput;
