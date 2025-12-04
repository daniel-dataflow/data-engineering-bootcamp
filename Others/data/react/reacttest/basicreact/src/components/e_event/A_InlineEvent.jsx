import React from "react";

export default function A_InlineEvent() {
  return (
    <div className="flex flex-col space-y-5 items-center">
      <h2>inline으로 이벤트 설정하고 이벤트객체 이용하기</h2>
      <button
        className="max-w-fit"
        onClick={() => {
          alert("클릭했다");
        }}
      >
        클릭해봐
      </button>
      <input
        className="max-w-fit"
        onChange={() => {
          console.log("입력했다");
        }}
      />
      <h3>이벤트객체 이용하기</h3>
      <p>이벤트 객체를 활용하는건 js에서 활용한 것과 동일함 </p>
      <input
        className="max-w-fit"
        onChange={(e) => {
          console.log(e);
          //리액트에서 이 방법을 이용하지 않음 -> state을 이용해서 데이터를 처리함.
          console.log(e.target.value);
          //리액트에서 권장하지 않는 방식 화면에 출력되는 데이터는 state, props를 이용해서 처리해야함. 나중에 배움
          e.target.nextElementSibling.innerText = e.target.value;
          console.log("입력했다");
        }}
      />
      <span></span>
    </div>
  );
}
