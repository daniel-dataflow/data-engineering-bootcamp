import React from "react";

export default function D_LayoutComponent() {
  return (
    <div>
      <h2>레이아웃 설정 클래스</h2>
      <div>
        <p>inline, block, line-block 설정</p>
        <span className="block">block설정 span태그</span>
        <h3 className="inline">inline h3</h3>
        <h3 className="inline">inline h3</h3>
        <div
          type="text"
          className="inline-block border border-red-300 w-50 h-40"
        >
          inline-block
        </div>
        <div
          type="text"
          className="inline-block border border-red-300  w-50 h-60"
        >
          inline-block
        </div>
      </div>
    </div>
  );
}
