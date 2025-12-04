import React, { useContext } from "react";
import { ChangeContext } from "./resources/Context";
export default function C_ChangeContextValue2() {
  const data = useContext(ChangeContext);
  return (
    <div>
      <h4>ChangeContextValue2</h4>
      <h5>context값 출력하기</h5>
      <p>변경되나 확인하기</p>
      <p>ChangeContext값 : {typeof data == "string" ? data : data.data}</p>
      {typeof data == "object" && (
        <input
          type="text"
          onChange={(e) => {
            data.setData(e.target.value);
          }}
        />
      )}
    </div>
  );
}
