import React from "react";
import { Link } from "react-router-dom";
import { headernav } from "../../resources/commondata";
const style = {
  display: "flex",
  margin: 0,
  padding: 0,
  width: "100%",
  justifyContent: "space-evenly",
  flexWrap: "wrap",
  listStyle: "none",
};
export default function HeaderComponent() {
  return (
    <ul style={style}>
      {headernav.map((v, i) => {
        return (
          <li key={`${v.path}_${i}`}>
            {/* 
            Link컴포넌트를 이용해서 페이지를 전환할 수 있음 -> a태그로 변환  
              속성은 to, replace, state를 설정할 수 있음
            */}
            <Link to={v.path}>{v.label}</Link>
          </li>
        );
      })}
    </ul>
  );
}
