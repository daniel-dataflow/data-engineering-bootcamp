import React, { useContext } from "react";
import { ContextTest } from "../resources/Context";

export default function B_Child() {
  const contextData = useContext(ContextTest);
  return (
    <div>
      <h3>자식컴포넌트</h3>
      <p>props을 전달받지 않고 전역의 Context를 이용할 수 있음</p>
      <h4>데이터 이용 basicData:{contextData}</h4>
    </div>
  );
}
