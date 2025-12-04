import React from "react";

export default function D_SectionContainer() {
  return (
    <div>
      <section className="flex items-center justify-center h-screen bg-gradient-to-br from-blue-500 to-teal-400">
        <div className="text-center text-white">
          <h1 className="text-5xl font-bold">나의 홈페이지</h1>
          <p className="mt-4 text-lg">재미있는 리액트 컴포넌트</p>
          <button className="mt-6 px-6 py-3 bg-white text-blue-500 rounded-lg font-semibold shadow-lg hover:bg-gray-100 transition">
            시작하기
          </button>
        </div>
      </section>
    </div>
  );
}
