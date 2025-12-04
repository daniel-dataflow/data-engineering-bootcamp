import React from "react";

export default function ModalComponent({
  isOpen,
  onClose,
  children,
  variant = "primary",
}) {
  if (!isOpen) {
    return null;
  }
  const textbg =
    variant == "primary"
      ? "bg-blue-100"
      : variant == "warn"
      ? "bg-yellow-100"
      : variant == "error"
      ? "bg-red-200"
      : "";
  return (
    // 오버레이: 화면 전체를 덮는 반투명 배경
    <div
      className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50"
      onClick={onClose}
    >
      {/* 모달 컨텐츠: 실제 내용이 들어가는 흰색 박스 */}
      <div
        className={
          "rounded-lg shadow-xl p-6 relative max-w-md w-full " + textbg
        }
        onClick={(e) => e.stopPropagation()} // 컨텐츠 클릭 시 닫히지 않도록 이벤트 전파 방지
      >
        {/* 닫기 버튼 */}
        <button
          className="absolute top-2 right-2 text-2xl text-gray-500 hover:text-gray-800 rounded-lg"
          onClick={onClose}
        >
          &times;
        </button>
        {children}
      </div>
    </div>
  );
}
