import React from "react";

export default function D_NavBarComponent() {
  return (
    <>
      <nav className="flex items-center justify-between px-6 py-4 bg-lime-50 shadow-md rounded">
        <div className="text-xl font-bold text-indigo-600">MyLogo</div>
        <ul className="flex space-x-6 list-none">
          <li className="text-gray-600 hover:text-indigo-500 transition-colors">
            메인화면
          </li>
          <li className="text-gray-600 hover:text-indigo-500 transition-colors">
            내정보
          </li>
          <li className="text-gray-600 hover:text-indigo-500 transition-colors">
            기타
          </li>
        </ul>
      </nav>
    </>
  );
}
