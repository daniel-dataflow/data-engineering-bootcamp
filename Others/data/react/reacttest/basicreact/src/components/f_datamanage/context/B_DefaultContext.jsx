import React, { useContext } from "react";
// 1. context가져오기
import { ContextDefault } from "./resources/Context";
export default function B_DefaultContext() {
  //hooks를 이용해서 context데이터를 사용
  const defaultContext = useContext(ContextDefault);
  return (
    <div>
      <h4>Hooks로 context생성시 설정한 값을 가져오기</h4>
      {Object.entries(defaultContext).map((data, i) => (
        <p key={i}>
          {data[0]} : {data[1]}
        </p>
      ))}
      <h4>태그로 default값 가져오기</h4>
      <ContextDefault.Consumer>
        {(value) =>
          Object.entries(value).map((data, i) => (
            <p key={i}>
              {data[0]} : {data[1]}
            </p>
          ))
        }
      </ContextDefault.Consumer>
    </div>
  );
}
