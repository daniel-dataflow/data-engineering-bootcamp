import React from "react";

export default function CardComponent({ title, imgSrc, hightlight, children }) {
  //스타일 적용 변수 선언하기
  const baseClass =
    "bg-white rounded-lg shadow-md overflow-hidden transition-all duration-300 hover:shadow-xl";
  const highlightedClasses = "border-2 border-blue-500 shadow-blue-500/20";
  const cardCss = `${baseClass} ${
    hightlight ? highlightedClasses : "border border-gray-200"
  }`;

  return (
    <div className={cardCss}>
      {/* 이미지 데이터가 전달되면 이미지 출력 */}
      {imgSrc && (
        <img src={imgSrc} alt={title} className="w-full h-48 object-cover" />
      )}
      <div className="p-4">
        <h2 className="text-xl font-bold mb-2">{title}</h2>
        <div className="text-gray-700">{children}</div>
      </div>
    </div>
  );
}
