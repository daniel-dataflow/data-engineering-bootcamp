import React, { useCallback, useState } from "react";

import C_ChildButton from "./C_ChildButton";

export default function C_BasicCallback() {
  const [count, setCount] = useState(0);
  const [value, setValue] = useState();

  // 선언된 함수는 랜더링 될때마다 새로운 함수객체를 생성해버림.
  const increase = () => {
    setCount((prev) => prev + 1);
  };
  const decrease = () => {
    setCount((prev) => prev - 1);
  };
  const callbackIncrease = useCallback(() => {
    setCount((prev) => prev + 1);
  }, []);
  const callbackDecrease = useCallback(() => {
    setCount((prev) => prev - 1);
  }, []);
  return (
    <div>
      <h4>props로 전달되는 함수적용하기</h4>
      <p>counf : {count}</p>
      <p>
        부모컴포넌트가 리랜더링 되면 함수객체를 계속 생성하는 현상이 발생함 ->
        Props으로 함수를 전달받는 컴포넌트는 React.memo를 설정해야함 하지만
        usecallback()을 하지 않고 적용하면 함수를 계속 생성함.
      </p>
      <p></p>
      <input type="text" onChange={(e) => setValue(e.target.value)} />
      <h4>함수를 계속 생성하는 버튼</h4>
      <C_ChildButton
        label="nocallback"
        title="증가"
        onClick={increase}
      ></C_ChildButton>
      <C_ChildButton
        label="nocallback"
        title="감소"
        onClick={decrease}
      ></C_ChildButton>
      <h4>useCallback을 적용해서 호출하기</h4>
      <p>함수를 한번만 호출하는 버튼</p>
      <C_ChildButton
        label="callback적용"
        title="증가"
        onClick={callbackIncrease}
      />
      <C_ChildButton
        label="callback적용"
        title="감소"
        onClick={callbackDecrease}
      />
    </div>
  );
}
