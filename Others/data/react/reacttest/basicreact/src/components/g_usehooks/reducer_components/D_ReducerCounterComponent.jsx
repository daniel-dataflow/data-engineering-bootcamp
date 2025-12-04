import React, { useReducer } from "react";
import { counterReducer } from "../reducers/reducer";
export default function D_ReducerCounterComponent() {
  const [state, dispatch] = useReducer(counterReducer, 0);
  return (
    <div>
      <h3>카운터 컴포넌트 useReducer로 구현하기</h3>
      <h4>현재값 : {state}</h4>
      <button onClick={() => dispatch({ type: "INCREASE" })}>증가</button>
      <button onClick={() => dispatch({ type: "DECREASE" })}>감소</button>
    </div>
  );
}
