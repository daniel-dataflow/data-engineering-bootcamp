import React from "react";
import A_Child2 from "./A_Child2";

export default function A_Child({ data }) {
  return (
    <div className="border border-red-500 p-5">
      <h4>A_Child</h4>
      <h4>데이터를 활용하지 않는 컴포넌트</h4>
      <h4>데이터 전달</h4>
      <A_Child2 data={data}>데이터 전달</A_Child2>
      <h4>데이터 전달하지않음</h4>
      <A_Child2>데이터 전달하지 않음</A_Child2>
    </div>
  );
}
