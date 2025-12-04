import React, { useContext, useRef } from "react";
import { ContextTest } from "../resources/Context";

function useRenderCount(){
  const ref=useRef(0);
  ref.current+=1
  return ref.current
}
export default function D_Child() {
  const myContext = useContext(ContextTest);
  // console.log("D_Child 랜더링됨!");
  const renders=useRenderCount()

  return (
    <div>
      <h4>contextdata출력 : {myContext.data}</h4>
      <p>랜더링 수 : {renders}</p>
    </div>
  );
}
