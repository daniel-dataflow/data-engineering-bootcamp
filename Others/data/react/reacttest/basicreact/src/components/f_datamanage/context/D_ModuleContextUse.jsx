import React from "react";
import { useMyContext } from "./resources/myprovider";
export default function D_ModuleContextUse({ children="", isUpdate = false }) {
  const contextData = useMyContext();
  return (
    <div>
      <h4>모듈 provider데이터 출력하기 {children!=''?`/ isUpdate ${children}`:`` }</h4>
      <p>provider 데이터 출력하기</p>
      {/* key/value를 순차적으로 출력 : id:bslove -> pw:bs1234 */}
      {Object.entries(contextData.data).map((d, i) => (
        <>
          <p key={i}>
            {d[0]} {d[1]}
          </p>
          <input key={`${d[0]}_${d[1]}`} type="text" onChange={(e)=>contextData.setData(pre=>({...pre,[d[0]]:e.target.value}))}></input>
          {/* context의 데이터를 수정하는 태그 만들기 props(isUpdate)값에 따라 출력 결정 */}
          {isUpdate && (
            <input
              type="text"
              name={d[0]}
              // value={d[1]}
              onChange={(e) => {
                contextData.setData((prev) => ({
                  ...prev,
                  [d[0]]: e.target.value,
                }));
              }}
            />
          )}
        </>
      ))}
    </div>
  );
}
