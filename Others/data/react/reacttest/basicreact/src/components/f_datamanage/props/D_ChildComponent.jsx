import React from "react";

export default function D_ChildComponent(props) {
  return (
    <div>
      <h4>props.children값 가져오기</h4>
      <div>
        <h5>데이터 및 타입확인하기</h5>
        {typeof props.children == "object"
          ? Object.keys(props.children)
          : props.children}
        &nbsp;
        {typeof props.children}
      </div>
    </div>
  );
}
