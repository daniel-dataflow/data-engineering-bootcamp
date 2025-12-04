import React, { useState } from "react";

export default function A_StateBasicFunction() {
  const [strData, setStrData] = useState("초기값");
  const [numData, setNumData] = useState(19);
  const strDataChange = (e) => {
    setStrData(e.target.value);
  };
  return (
    <div>
      <h3>함수형 컴포넌트에서 state이용하기</h3>
      <p>
        useState() hooks를 이용해서 state생성 개별로 생성해서 필요한 데이터에
        따라 useState()를 이용해서 별도 생성 관리해야함. useState() hooks를
        이용하면 배열로 value, setState함수를 반환하기 때문에
        배열구조분해할당으로 저장해서 이용 인수로는 초기값을 설정함.
      </p>
      <h4>state 값 출력하기</h4>
      <p>
        변수로 설정해서 변수명을 호출하면 state에 저장된 값을 출력할 수 있음
      </p>
      <p>strData : {strData}</p>
      <p>numData : {numData}</p>
      <h4>state 수정하기</h4>
      <p>
        반환받은 set메소드를 이용해서 수정함 매개변수로 전달함 값으로 value를
        덮어쓰기함 -> 일반변수에 값을 대입한것과 동일함.
      </p>
      <input type="text" onChange={strDataChange} />
      <button
        onClick={() => {
          setNumData(10);
        }}
      >
        10
      </button>
      <button
        onClick={() => {
          setNumData(20);
        }}
      >
        20
      </button>
    </div>
  );
}
