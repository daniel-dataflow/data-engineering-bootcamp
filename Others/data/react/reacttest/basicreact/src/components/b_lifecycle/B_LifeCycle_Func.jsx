import React, { useState, useEffect } from "react";

export default function B_LifeCycle_Func() {
  const [data, setData] = useState("초기값");
  const [data1, setData1] = useState("변경하지 않는값");
  useEffect(() => {
    console.log("mount에서 실행");
  }, []);
  useEffect(() => {
    console.log(`data1값이 수정되면 실행 ${data1}`);
    console.log(data);
    console.log(data1);
  }, [data1]);
  useEffect(() => {
    console.log("unmount에서 실행");
    return () => {
      console.log("unmount설정");
      //clearup함수처리시 interval함수를 종료할때 사용
    };
  }, []);
  return (
    <div>
      <h3>함수형 컴포넌트에서 lifecycle이용하기</h3>
      <p>함수형 컴포넌트에서는 useEffect()함수를 이용함</p>
      <button
        onClick={() => {
          setData((pre) => {
            return pre + 1;
          });
        }}
      >
        state data변경하기
      </button>
      <button
        onClick={() => {
          setData1((pre) => {
            return pre + 1;
          });
        }}
      >
        state data1변경하기
      </button>
    </div>
  );
}
