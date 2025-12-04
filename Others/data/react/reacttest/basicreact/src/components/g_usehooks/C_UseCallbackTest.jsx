import React from "react";
import C_CallbackWithEffect from "./child/C_CallbackWithEffect";

export default function C_UseCallbackTest() {
  return (
    <div>
      <h3>useCallback()활용하기</h3>
      <p>
        데이터(값)이 아닌 전달되는 함수를 메모이제이션하는 hook 의존성이 바뀌지
        않는 이상 같은 함수를 계속 전달함.
        <br />
        주로 함수를 props로 전달할때, useEffect, useMemo 등 다른 hook의
        의존성으로 함수를 넣고 싶을때, 랜더일 최적화할때
      </p>
      <C_BasicCallback></C_BasicCallback>
      <C_CallbackWithEffect></C_CallbackWithEffect>
      <p>
        모든 함수에 무조건 적용하는것 보다 성능에 문제가 생기는 부분만 선별해서
        적용하는것이 좋음 -> 공식문서
      </p>
      <p>너무 많은 useCallback은 가독성을 떨어트림.</p>
    </div>
  );
}
