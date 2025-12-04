import React from "react";
import { ContextTest } from "./resources/Context";
export default function B_UseContext_Tag() {
  return (
    <div>
      <h3>태그 방식으로 가져오기</h3>
      <p>
        Context명.Consumer태그를 이용해서 가져오기 태그내부에 value를 매개변수를
        받는 함수를 설정해서 활용함
      </p>
      <ContextTest.Consumer>
        {(value) => <h4>basicData : {value}</h4>}
      </ContextTest.Consumer>
    </div>
  );
}
