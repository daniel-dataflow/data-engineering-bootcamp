import React from "react";

import A_UserRefTest from "./A_UserRefTest";
import B_UseMemoTest from "./B_UseMemoTest";
import C_BasicCallback from "./child/C_BasicCallback";
import C_CallbackWithEffect from "./child/C_CallbackWithEffect";
import D_UseReducerContainer from "./D_UseReducerContainer";

export default function HooksContainer() {
  return (
    <div>
      <h3>Hooks활용하기</h3>
      <p>리액트의 기능을 손쉽게 이용할 수 있게 해주는 함수</p>
      {/* useRef() 이용하기 */}
      <A_UserRefTest></A_UserRefTest>
      {/* useMemo() 이용하기
      <B_UseMemoTest></B_UseMemoTest> */}
      {/* useCallback() 이용하기 */}
      <C_BasicCallback></C_BasicCallback>
      {/* useEffect으로 페이지 로딩시 특정 함수에 요청을 보내는 로직에서 useCallback이용하기 */}
      <C_CallbackWithEffect />
      {/* useReducer() 활용하기 */}
      <D_UseReducerContainer />
    </div>
  );
}
