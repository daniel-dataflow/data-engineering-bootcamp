import React, { useContext } from "react";
import { ContextTest } from "./resources/Context";
export default function B_UseObject() {
  const objContext = useContext(ContextTest);
  return (
    <div>
      <h4>객체 데이터 출력하기</h4>
      {Object.entries(objContext).map((data, i) => (
        <p key={i}>
          {data[0]} {data[1]}
        </p>
      ))}
    </div>
  );
}
