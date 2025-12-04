import React from "react";
import react from "../../../../assets/react.svg";
export default function D_CardComponent() {
  return (
    <div className="max-w-sm mx-auto bg-white rounded-2xl shadow-lg overflow-hidden hover:shadow-xl transition-shadow duration-300">
      <img className="w-full h-48 object-contain" src={react} alt="Nature" />
      <div className="p-4">
        <h2 className="text-lg font-semibold text-gray-800">Beautiful React</h2>
        <p className="text-gray-600 mt-2">
          Explore the wonders of the React world.
        </p>
        <button className="mt-4 px-4 py-2 bg-indigo-500 text-white rounded-lg hover:bg-indigo-600 transition-colors duration-300">
          Learn More
        </button>
      </div>
    </div>
  );
}
