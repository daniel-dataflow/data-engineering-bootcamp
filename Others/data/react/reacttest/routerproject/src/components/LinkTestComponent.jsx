import React from "react";
import { createSearchParams, Link } from "react-router-dom";

export default function LinkTestComponent() {
  return (
    <div>
      <h2>Link컴포넌트</h2>
      <p>
        컴포넌트로 링크를 설정할 때 사용하는 컴포넌트 -> a태그로 전환됨. 필수
        속성 to : 이동한 route경로를 설정, 문자열 / 객체 객체 구조 :{" "}
        {`{pathname:"",searh:"?key=value",hash:"#문자열"}`}
      </p>
      <div style={{ display: "flex", justifyContent: "space-evenly" }}>
        <Link to="/about">기본연결</Link>
        <Link
          to={{
            pathname: "/about",
            // search: "?name=test&value=데이터",
            //객체 데이터를 querystring으로 변환해서 전달하기
            search: `?${createSearchParams({ name: "test", value: "데이터" })}`,
            hash: "#8282",
          }}
        >
          객체로 연결하기
        </Link>
        <Link to="/about" state={{ id: "admin", key: "123#111" }}>
          state-표현되지않는데이터 전송
        </Link>
        <Link to="/about" replace={true}>
          History를 남기지않기 - -1로 되돌아갈 수 없음
        </Link>
      </div>
    </div>
  );
}
