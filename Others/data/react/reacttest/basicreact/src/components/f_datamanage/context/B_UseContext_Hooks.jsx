import React, { useContext } from "react";
import { ContextTest } from "./resources/Context";
import B_Child from "./child/B_Child";

export default function B_UseContext() {
  //context데이터 가져오기
  //useContext hooks를 이용하기
  const contextData = useContext(ContextTest);
  return (
    <div>
      <h3>hooks를 이용해서 Context이용하기</h3>
      <p>useContext() hooks를 이용해서 context값을 가져올 수 있음</p>
      <h4>contextData : {contextData}</h4>
      <h3>Hooks사용한 태그의 자식 컴포넌트도 전달 없이 이용가능</h3>
      <B_Child />
    </div>
  );
}
