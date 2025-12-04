import React from "react";
import A_InlineEvent from "./A_InlineEvent";
import B_FunctionEvent from "./B_FunctionEvent";

export default function EventContainer() {
  return (
    <div>
      <h2>이벤트 설정하기</h2>
      <A_InlineEvent></A_InlineEvent>
      <B_FunctionEvent></B_FunctionEvent>
    </div>
  );
}
